#!/bin/bash
set -euo pipefail

# Set stage to control which part to run
# 0: data preparation
# 1: Flow Matching pretraining
# 2: GAN finetuning
# 3: Inference
# 4: Compute objective metrics
stage=0

root_dir=download/libritts/LibriTTS  # Modify this path to your LibriTTS root directory

manifests=data/manifests/libritts
train_recordings=$manifests/recordings_train-all-shuf.jsonl.gz
valid_recordings=$manifests/recordings_valid.jsonl.gz
test_recordings=$manifests/recordings_test.jsonl.gz
test_recordings_small=$manifests/recordings_test_10.jsonl.gz

filelists=data/wav_list/libritts
valid_filelist=$filelists/filelist_valid.txt
test_filelist=$filelists/filelist_test.txt

# Prepare data
if [ $stage -eq 0 ]; then
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

  # Merge wav file lists for computing metrics
  cat $filelists/dev-clean.txt $filelists/dev-other.txt \
    > $valid_filelist
  cat $filelists/test-clean.txt $filelists/test-other.txt \
    > $test_filelist
fi

# Flow Matching pretraining
model_name=mel_24k_base
pretrain_exp_dir=./exp-pretrain
if [ $stage -eq 1 ]; then
  export CUDA_VISIBLE_DEVICES=0,1
  echo "Start Flow Matching pretraining"
  python ./pretrain.py \
    --world-size 2 \
    --num-epochs 200 \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir $pretrain_exp_dir \
    --model-name $model_name \
    --train-recordings $train_recordings \
    --valid-recordings $valid_recordings \
    --test-recordings $test_recordings_small \
    --batch-size 256 \
    --save-infer-steps "2,4,8" \
    --save-every-n 20 \
    --master 12345
fi

# GAN finetuning
generator_path=$pretrain_exp_dir/epoch-200-avg-40-use-avg-model.pt  # Modify this if needed
finetune_exp_dir=./exp-finetune
# Could set step to 1,2,4, would construct a GAN generator 
# by forwarding the Flow Matching model for 1,2,4 steps respectively
step=4  
if [ $stage -eq 2 ]; then
  # Save averaged model over last 40 epochs, would get $pretrain_exp_dir/epoch-200-avg-40-use-avg-model.pt
  python ./save_averaged_model.py \
    --epoch 200 \
    --avg 40 \
    --use-averaged-model True \
    --exp-dir $pretrain_exp_dir \
    --model-name $model_name 
  
  export CUDA_VISIBLE_DEVICES=0
  echo "Start GAN finetuning"
  python ./finetune.py \
    --world-size 1 \
    --num-epochs 20 \
    --start-epoch 1 \
    --use-fp16 0 \
    --exp-dir $finetune_exp_dir \
    --model-name $model_name \
    --generator-model-path $generator_path \
    --n-timesteps $step \
    --train-recordings $train_recordings \
    --valid-recordings $valid_recordings \
    --test-recordings $test_recordings_small \
    --batch-size 64 \
    --save-every-n 2 \
    --master 12345
fi

# Inference
if [ $stage -eq 3 ]; then
  # Inference with the GAN finetuned model
  # The inferred wavs would be saved to $finetune_exp_dir/wav-epoch-20-avg-4-use-avg-model-pred-step-${step}
  export CUDA_VISIBLE_DEVICES=0
  for rec in $valid_recordings $test_recordings; do
      echo "Inference on ${rec}"
      python ./infer.py \
        --epoch 20 \
        --avg 4 \
        --use-averaged-model True \
        --exp-dir $finetune_exp_dir \
        --infer-gan True \
        --model-name $model_name \
        --n-timesteps $step \
        --root-path $root_dir \
        --test-recordings $rec \
        --batch-size 64
  done
fi

# Compute objective metrics
pred_wav_dir=$finetune_exp_dir/wav-epoch-20-avg-4-use-avg-model-pred-step-${step}  # Modify this if needed
if [ $stage -eq 4 ]; then
  for f in $valid_filelist $test_filelist; do
      # Should first download a Wav2Vec2 model from https://huggingface.co/facebook/wav2vec2-base 
      # and save to download/huggingface/wav2vec2_base
      echo "Compute FSD on $f"
      export CUDA_VISIBLE_DEVICES=0
      python ./compute_fsd.py \
        --model-path download/huggingface/wav2vec2_base \
        --real-path $root_dir \
        --eval-path $pred_wav_dir \
        --wav-list-file $f

      echo "Compute PESQ and ViSQOL on $f"
      python ./compute_pesq_visqol.py \
        --gt-wav-dir $root_dir \
        --pred-wav-dir $pred_wav_dir \
        --wav-list-file $f \
        --use-visqol True \
        --n-proc 20
      
      echo "Compute V/UV F1 and Periodicity on $f"
      python ./compute_pitch_periodicity.py \
        --gt-wav-dir $root_dir \
        --pred-wav-dir $pred_wav_dir \
        --wav-list-file $f
  done
fi


