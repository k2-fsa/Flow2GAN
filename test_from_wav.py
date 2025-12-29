#!/usr/bin/env python3
# Copyright         2025  Xiaomi Corp.        (authors: Zengwei Yao)
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

import os

import soundfile as sf
import torch
import torchaudio

from flow2gan import get_model
from flow2gan.models.modules import LogMelSpectrogram


step = 4  # Could set step to 1,2,4
model_name = "mel_24k_base"
hf_model_name = f"libritts-mel-{step}-step"

# Required model will be downloaded from HuggingFace Hub automatically
model, model_cfg = get_model(model_name=model_name, hf_model_name=hf_model_name)

cond_module = LogMelSpectrogram(
    sampling_rate=model_cfg.sampling_rate,
    n_fft=model_cfg.mel_n_fft,
    hop_length=model_cfg.mel_hop_length,
    n_mels=model_cfg.n_mels,
    center=True,
    power=1,
)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

model.to(device)
model.eval()
cond_module.to(device)
cond_module.eval()

input_path = "./test_data/wav/1089_134686_000002_000000.wav"
output_path = "output.wav"
audio, sr = torchaudio.load(input_path)  # (1, time)
assert sr == model_cfg.sampling_rate
audio = audio.to(device)  

with torch.inference_mode():
    mel_spec = cond_module(audio)  # (1, n_mels, frames)
    pred_audio = model.infer(cond=mel_spec, n_timesteps=step, clamp_pred=True)

sf.write(output_path, pred_audio.cpu().squeeze(0).numpy(), sr)
print(f"Wrote output to {output_path}")

