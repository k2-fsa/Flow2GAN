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
from typing import Optional

import soundfile as sf
import torch

from flow2gan import get_model


step = 4  # Could set step to 1,2,4
model_name = "mel_24k_base"
hf_model_name = f"libritts-mel-{step}-step"

model, model_cfg = get_model(model_name=model_name, hf_model_name=hf_model_name)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda", 0)

model.to(device)
model.eval()

input_path = "./test_data/mel/1089_134686_000002_000000.pt"
output_path = "output.wav"
mel_spec = torch.load(input_path)  # (1, n_mels, frames)
mel_spec = mel_spec.to(device)  

with torch.inference_mode():
    pred_audio = model.infer(cond=mel_spec, n_timesteps=step, clamp_pred=True)

sf.write(output_path, pred_audio.cpu().squeeze(0).numpy(), model_cfg.sampling_rate)
print(f"Wrote output to {output_path}")

