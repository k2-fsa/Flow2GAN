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

import argparse
import logging
import os
from typing import Optional

import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download

from flow2gan.bin.pretrain import get_cond_module_and_generator
from flow2gan.checkpoint import load_checkpoint
from flow2gan.models.config import HF_MODEL_NAMES, HF_REPO
from flow2gan.utils import AttributeDict, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Local path to the checkpoint to load.",
    )

    parser.add_argument(
        "--hf-model-name",
        type=str,
        default=None,
        help="""Name of the model file in HF hub. Supported names are 
        `libritts-mel-1-step`, `libritts-mel-2-step`, `libritts-mel-4-step`.""",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mel_24k_base",
        help="""Name of the model to use. Supported names are 
        `mel_24k_base`.""",
    )

    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=1,
        help="""Number of steps for generator forward.""",
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=-1,
        help="""Chunk size in frames for streaming inference. If -1, do non-streaming inference.""",
    )

    parser.add_argument(
        "--input-type",
        type=str,
        default="audio",
        help="Type of the input: audio or mel.",
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory of input files.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to save output wavs.",
    )

    return parser


def infer_audio(
    params: AttributeDict,
    model: nn.Module,
    cond_module: nn.Module,
    audio: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None, 
) -> torch.Tensor:
    """Run the inference process."""
    assert (audio is not None) or (cond is not None), "Either audio or cond should be provided."
    device = next(model.parameters()).device

    with torch.inference_mode():
        if cond is None:
            audio = audio.to(device)  # (batch, time)
            cond = cond_module(audio)
        else:
            cond = cond.to(device)

        pred_audio = model.infer(
            cond=cond,
            n_timesteps=params.n_timesteps,
            clamp_pred=True,
        )

    return pred_audio.cpu()


def streaming_infer_audio(
    params: AttributeDict,
    model: nn.Module,
    cond_module: nn.Module,
    audio: Optional[torch.Tensor] = None,
    cond: Optional[torch.Tensor] = None, 
) -> torch.Tensor:
    """Run the inference process."""
    assert (audio is not None) or (cond is not None), "Either audio or cond should be provided."
    device = next(model.parameters()).device    

    with torch.inference_mode():
        if cond is None:
            audio = audio.to(device)  # (batch, time)
            cond = cond_module(audio)
        else:
            cond = cond.to(device)

        num_frames = cond.size(2)
        side_context = 3 * 8  # in frames, assume conv_kernel_size=7, layers=8
        chunk = params.chunk_size  # in frames
        num_chunks = (num_frames + chunk - 1) // chunk
        chunked_preds = []

        for i in range(num_chunks):
            frame_start = max(0, i * chunk - side_context)
            frame_end = min(num_frames, (i + 1) * chunk + side_context)
            left_pad_samples = (i * chunk - frame_start) * model.mel_hop_length
            right_pad_samples = (frame_end - (i + 1) * chunk) * model.mel_hop_length

            logging.info(f"Processing chunk {i+1}/{num_chunks}, frames {frame_start}:{frame_end}")
            
            pred_audio_chunk = model.infer(
                cond=cond[:, :, frame_start:frame_end],
                n_timesteps=params.n_timesteps,
                clamp_pred=True,
            )
            pred_audio_chunk = pred_audio_chunk[:, left_pad_samples: pred_audio_chunk.size(1) - right_pad_samples]
            chunked_preds.append(pred_audio_chunk)

        chunked_preds = torch.cat(chunked_preds, dim=-1)
    
    return chunked_preds.cpu()


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict(vars(args))
    params.update(vars(args))

    logging.info("Inference start")

    logging.info("About to create model")
    cond_module, model = get_cond_module_and_generator(params) 
    logging.info(model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_param}")

    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    if params.checkpoint is not None:
        logging.info(f"Using local checkpoint: {params.checkpoint}")
        checkpoint = params.checkpoint
    else:
        logging.info("Using checkpoint from HF hub")
        assert params.hf_model_name in HF_MODEL_NAMES and params.n_timesteps == HF_MODEL_NAMES[params.hf_model_name] 
        checkpoint = hf_hub_download(HF_REPO, filename=params.hf_model_name + ".pt")
    load_checkpoint(checkpoint, model)

    model.to(device)
    model.eval()
    cond_module.to(device)
    cond_module.eval()

    os.makedirs(params.output_dir, exist_ok=True)
    for filename in os.listdir(params.input_dir):
        if not filename.endswith(".wav") and not filename.endswith(".pt"):
            continue
        
        input_path = os.path.join(params.input_dir, filename)
        output_path = os.path.join(params.output_dir, filename.replace(".pt", ".wav"))

        if params.input_type == "audio":
            audio, sr = torchaudio.load(input_path)  # (1, time)
            assert sr == params.sampling_rate
            mel_spec = None
        elif params.input_type == "mel":
            mel_spec = torch.load(input_path)  # (1, n_mels, frames)
            audio = None
        else:
            raise ValueError(f"Unsupported input type: {params.input_type}")
        
        infer_func = streaming_infer_audio if params.chunk_size > 0 else infer_audio
        pred_audio = infer_func(
            params=params, 
            model=model, 
            cond_module=cond_module,
            audio=audio, 
            cond=mel_spec,
        )  # (1, time)

        sf.write(output_path, pred_audio.squeeze(0).numpy(), params.sampling_rate)
        logging.info(f"Wrote output to {output_path}")

    logging.info("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
