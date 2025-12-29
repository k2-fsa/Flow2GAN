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

from typing import Optional, Tuple

from huggingface_hub import hf_hub_download

from flow2gan.checkpoint import load_checkpoint
from flow2gan.models.config import HF_MODEL_NAMES, HF_REPO, get_generator_config
from flow2gan.models.generator import MelAudioGenerator
from flow2gan.models.modules import LogMelSpectrogram
from flow2gan.utils import AttributeDict


def get_model(
    model_name: str = "mel_24k_base", 
    hf_model_name: Optional[str] = "libritts-mel-4-step",
    checkpoint: Optional[str] = None,
) -> Tuple[MelAudioGenerator, AttributeDict]: 
    assert (checkpoint is not None) or (hf_model_name is not None), \
        "Either checkpoint or hf_model_name must be provided."

    model_cfg = get_generator_config(model_name)
    model = MelAudioGenerator(**model_cfg)

    if checkpoint is not None:
        print(f"Using local checkpoint: {checkpoint}")
    else:
        print("Using checkpoint from HF hub")
        assert hf_model_name in HF_MODEL_NAMES, "Supported names are " + ", ".join(HF_MODEL_NAMES.keys())
        checkpoint = hf_hub_download(HF_REPO, filename=hf_model_name + ".pt")
    load_checkpoint(checkpoint, model)

    return model, model_cfg