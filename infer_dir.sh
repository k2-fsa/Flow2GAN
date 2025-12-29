
#!/bin/bash

set -euo pipefail

step=4  # Could be 1,2,4
model_name=mel_24k_base
hf_model_name=libritts-mel-${step}-step

# Required model will be downloaded from HuggingFace Hub automatically

# Infer from a directory of audio files (*.wav)
python -m flow2gan.bin.infer_dir \
    --model-name $model_name \
    --n-timesteps $step \
    --hf-model-name $hf_model_name \
    --input-type audio \
    --input-dir ./test_data/wav/ \
    --output-dir ./output_from_wav/

# Infer from a directory of mel files (*.pt)
python -m flow2gan.bin.infer_dir \
    --model-name $model_name \
    --n-timesteps $step \
    --hf-model-name $hf_model_name \
    --input-type mel \
    --input-dir ./test_data/mel/ \
    --output-dir ./output_from_mel/
