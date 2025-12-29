#!/bin/bash
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script runs the full pipeline of Flow2GAN on LibriTTS dataset to reproduce the results in the paper:
# "Flow2GAN: Hybrid Flow Matching and GAN with Multi-Resolution Network for Few-step High-Fidelity Audio Generation"
# ()
# You can extend it to your own dataset with necessary modifications.

set -euo pipefail

# Control which part to run
# -1: data preparation
# 0: Inference with the HuggingFace model
# 1: Flow Matching pretraining
# 2: Save averaged model for GAN generator initialization
# 3: GAN finetuning
# 4: Inference
# 5: Compute objective metrics
# 6: Save averaged GAN generator model for final deployment
# 
# After data preparation in stage -1, could either
# a) set stage=0 and stop_stage=0 to directly inference with the HuggingFace model,
#    then jump to stage 5 to compute objective metrics; 
# or
# b) set stage=1 and stop_stage=6 to run the full pipeline.

stage=0
stop_stage=0

root_dir=download/libritts/LibriTTS/  # Modify this path to your LibriTTS root directory
manifests=data/manifests/libritts
train_recordings=$manifests/recordings_train-all-shuf.jsonl.gz
valid_recordings=$manifests/recordings_valid.jsonl.gz
test_recordings=$manifests/recordings_test.jsonl.gz
test_recordings_small=$manifests/recordings_test_10.jsonl.gz
filelists=data/wav_list/libritts
valid_filelist=$filelists/filelist_valid.txt
test_filelist=$filelists/filelist_test.txt

model_name=mel_24k_base
# Pretrain settings
pt_exp_dir=./output/exp-pretrain
pt_epoch=200  
pt_avg=40
pt_batch_size=256  # Could adjust based on your GPU memory
pt_num_gpus=2
# Finetune settings
# Could set step to 1,2,4, would construct a GAN generator 
# by forwarding the Flow Matching model for 1,2,4 steps respectively
step=4
generator_path=$pt_exp_dir/epoch-${pt_epoch}-avg-${pt_avg}-use-avg-model.pt  
ft_exp_dir=./output/exp-finetune-step-${step}
ft_epoch=20  
ft_avg=4
ft_batch_size=64  # Could adjust based on your GPU memory
ft_num_gpus=1
pred_wav_dir=$ft_exp_dir/wav-epoch-${ft_epoch}-avg-${ft_avg}-use-avg-model-pred-step-${step} 

# Prepare data
if [ $stage -le -1 ] && [ $stop_stage -ge -1 ]; then
  echo "Download LibriTTS data"
  if [ ! -d "$root_dir" ]; then
    echo "LibriTTS root not found at $root_dir."
    echo "Please download and extract LibriTTS into this path"
    exit 1
  fi

  echo "Prepare recording manifests"
  python ./scripts/prepare_recordings_libritts.py \
    --root-dir $root_dir \
    --out-dir $manifests \
    --num-jobs 20

  echo "Prepare wav file lists for metrics evaluation"
  python ./scripts/prepare_test_list_libritts.py \
    --root-dir $root_dir \
    --out-dir $filelists

  echo "Combine & shuffle manifests"
  if [ ! -f $train_recordings ]; then
    cat <(gunzip -c $manifests/recordings_train-clean-100.jsonl.gz) \
      <(gunzip -c $manifests/recordings_train-clean-360.jsonl.gz) \
      <(gunzip -c $manifests/recordings_train-other-500.jsonl.gz) \
      | shuf | gzip -c > $train_recordings
  fi
  if [ ! -f $valid_recordings ]; then
    cat <(gunzip -c $manifests/recordings_dev-clean.jsonl.gz) \
      <(gunzip -c $manifests/recordings_dev-other.jsonl.gz) \
      | shuf | gzip -c > $valid_recordings
  fi
  if [ ! -f $test_recordings ]; then
    cat <(gunzip -c $manifests/recordings_test-clean.jsonl.gz) \
      <(gunzip -c $manifests/recordings_test-other.jsonl.gz) \
      | gzip -c > $test_recordings
  fi

  # Create small subsets for tensorboard visualization in training
  gunzip -c $test_recordings | shuf | head -n 10 | \
    gzip -c > $test_recordings_small

  # Merge wav file lists for computing objective metrics
  cat $filelists/dev-clean.txt $filelists/dev-other.txt > $valid_filelist
  cat $filelists/test-clean.txt $filelists/test-other.txt > $test_filelist

  echo "Data preparation done."
fi

# Inference with the HuggingFace model
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
  echo "Inference with the HuggingFace model"
  pred_wav_dir=./output-libritts-mel-${step}-step
  # The inferred wavs would be saved to ./output-libritts-mel-${step}-step
  for rec in $valid_recordings $test_recordings; do
      echo "Inference on ${rec}"
      python -m flow2gan.bin.infer \
        --hf-model-name libritts-mel-${step}-step \
        --output-dir $pred_wav_dir \
        --load-gan False \
        --model-name $model_name \
        --n-timesteps $step \
        --root-path $root_dir \
        --test-recordings $rec \
        --batch-size 64
  done
  
  echo "Now jump to stage 5 to compute objective metrics on the inference results."
  stage=5
  stop_stage=5
fi

# Flow Matching pretraining
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
  echo "Start Flow Matching pretraining"
  python -m flow2gan.bin.pretrain \
    --world-size $pt_num_gpus \
    --num-epochs $pt_epoch \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir $pt_exp_dir \
    --model-name $model_name \
    --train-recordings $train_recordings \
    --valid-recordings $valid_recordings \
    --test-recordings $test_recordings_small \
    --batch-size $pt_batch_size \
    --save-infer-steps "2,4,8" \
    --save-every-n 20 \
    --master 12345
fi

if [ $stage -le 2 ] && [ $stop_stage -ge 2 ]; then
  echo "Save averaged model for GAN generator initialization"
  # Would get $pt_exp_dir/epoch-${pt_epoch}-avg-${pt_avg}-use-avg-model.pt
  python -m flow2gan.bin.save_averaged_model \
    --epoch $pt_epoch \
    --avg $pt_avg \
    --use-averaged-model True \
    --exp-dir $pt_exp_dir \
    --load-gan False \
    --model-name $model_name 
fi

# GAN finetuning
if [ $stage -le 3 ] && [ $stop_stage -ge 3 ]; then
  echo "Start GAN finetuning"
  python -m flow2gan.bin.finetune \
    --world-size $ft_num_gpus \
    --num-epochs $ft_epoch \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir $ft_exp_dir \
    --model-name $model_name \
    --generator-model-path $generator_path \
    --gen-start-batch-idx 1000 \
    --n-timesteps $step \
    --train-recordings $train_recordings \
    --valid-recordings $valid_recordings \
    --test-recordings $test_recordings_small \
    --batch-size $ft_batch_size \
    --save-every-n 2 \
    --master 12345
fi

# Inference
if [ $stage -le 4 ] && [ $stop_stage -ge 4 ]; then
  echo "Inference with the GAN finetuned model"
  # The inferred wavs would be saved to $ft_exp_dir/wav-epoch-${ft_epoch}-avg-${ft_avg}-use-avg-model-pred-step-${step}
  for rec in $valid_recordings $test_recordings; do
      echo "Inference on ${rec}"
      python -m flow2gan.bin.infer \
        --epoch $ft_epoch \
        --avg $ft_avg \
        --use-averaged-model True \
        --exp-dir $ft_exp_dir \
        --load-gan True \
        --model-name $model_name \
        --n-timesteps $step \
        --root-path $root_dir \
        --test-recordings $rec \
        --batch-size 64
  done
fi

# Compute objective metrics
if [ $stage -le 5 ] && [ $stop_stage -ge 5 ]; then
  # Should first install packages in requirements_eval.txt
  # For ViSQOL, please follow instructions at https://github.com/google/visqol
  echo "Compute objective metrics"
  for f in $valid_filelist $test_filelist; do
      # Should first download a Wav2Vec2 model from https://huggingface.co/facebook/wav2vec2-base 
      # and save to download/wav2vec2_base
      echo "Compute FSD on $f"
      python ./scripts/compute_fsd.py \
        --model-path download/wav2vec2_base \
        --real-path $root_dir \
        --eval-path $pred_wav_dir \
        --wav-list-file $f

      echo "Compute PESQ and ViSQOL on $f"
      python ./scripts/compute_pesq_visqol.py \
        --gt-wav-dir $root_dir \
        --pred-wav-dir $pred_wav_dir \
        --wav-list-file $f \
        --use-visqol True \
        --n-proc 20
      
      echo "Compute V/UV F1 and Periodicity on $f"
      python ./scripts/compute_pitch_periodicity.py \
        --gt-wav-dir $root_dir \
        --pred-wav-dir $pred_wav_dir \
        --wav-list-file $f
  done
fi

if [ $stage -le 6 ] && [ $stop_stage -ge 6 ]; then
  echo "Save averaged model (GAN generator) for final deployment"
  # Would get $ft_exp_dir/epoch-${ft_epoch}-avg-${ft_avg}-use-avg-model-only-gen.pt
  python -m flow2gan.bin.save_averaged_model \
    --epoch $ft_epoch \
    --avg $ft_avg \
    --use-averaged-model True \
    --exp-dir $ft_exp_dir \
    --load-gan True \
    --model-name $model_name 
fi
