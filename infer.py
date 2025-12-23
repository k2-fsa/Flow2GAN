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

from lhotse import RecordingSet
import soundfile as sf
import torch
import torch.nn as nn

from checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from config import get_gan_config
from dataset import build_data_loader
from gan import GAN
from pretrain import get_cond_module_and_generator
from utils import AttributeDict, setup_logger, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=20,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=1,
        help="""Number of checkpoints to average. Automatically select
        consecutive checkpoints before the checkpoint specified by --epoch""",
    )

    parser.add_argument(
        "--use-averaged-model",
        type=str2bool,
        default=True,
        help="Whether to load averaged model. Currently it only supports "
        "using --epoch. If True, it would decode with the averaged model "
        "over the epoch range from `epoch-avg` (excluded) to `epoch`."
        "Actually only the models with epoch number of `epoch-avg` and "
        "`epoch` are loaded for averaging. ",
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./exp-finetune",
        help="The experiment dir.",
    )

    parser.add_argument(
        "--infer-gan",
        type=str2bool,
        default=True,
        help="If true, infer with GAN model; If false, infer with the Flow Matching pretrained model.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mel_24k_base",
        help="""Name of the model to use. Supported names are 
        `mel_24k_base`.""",
    )

    parser.add_argument(
        "--gan-name",
        type=str,
        default="gan_multi_scale_mel_recon",
        help="""Name of the gan to use. Supported names are 
        `gan_multi_scale_mel_recon`, `gan_single_scale_mel_recon`.""",
    )

    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=1,
        help="""Number of steps for generator forward.""",
    )

    parser.add_argument(
        "--root-path",
        type=str,
        default=None,
        help="Root path of the audios. ",
    )

    parser.add_argument(
        "--test-recordings",
        type=str,
        default="",
        help="Manifest file to test set",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="How many subprocesses to use for data loading.",
    )

    return parser


def get_cond_module_and_model(params: AttributeDict) -> nn.Module:
    """Create condition module and generator or GAN."""
    # would add sampling_rate to params
    cond_module, generator = get_cond_module_and_generator(params) 

    if not params.infer_gan:
        return cond_module, generator

    gan_cfg = get_gan_config(params.gan_name)
    logging.info(gan_cfg)
    gan = GAN(generator=generator, **gan_cfg)

    return cond_module, gan


def infer_audio(
    params: AttributeDict,
    model: nn.Module,
    cond_module: nn.Module,
    dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run the inference process."""
    def makedir_if_necessary(filename: str):
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)

    device = next(model.parameters()).device
    total_samples = len(dataloader.dataset)
    cnt = 0
    log_interval = 10
    with torch.inference_mode():
        for batch_idx, (audio, audio_lens, file_names) in enumerate(dataloader):
            batch_size = audio.shape[0]
            audio = audio.to(device)  # (batch, time)
            audio_lens = audio_lens.to(device)  # (batch,)
            cond = cond_module(audio)  

            pred_audio = model.infer(
                cond=cond,
                audio_lens=audio_lens,
                n_timesteps=params.n_timesteps,
                clamp_pred=True,
            )

            for i in range(batch_size):
                pred = pred_audio[i, :audio_lens[i].item()].data.cpu().numpy()
                pred_out_file = f"{params.wav_dir_pred}/{file_names[i]}"
                makedir_if_necessary(pred_out_file)
                sf.write(pred_out_file, pred, params.sampling_rate)

            cnt += batch_size
            if batch_idx % log_interval == 0:
                logging.info(f"Processed {cnt} / {total_samples} samples")

        logging.info(f"Processed {cnt} samples in total.")


@torch.no_grad()
def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict(vars(args))
    params.update(vars(args))

    setup_logger(f"{params.exp_dir}/log/log-infer")
    logging.info("Inference start")

    logging.info("About to create model")
    cond_module, model = get_cond_module_and_model(params)
    logging.info(model)
    
    if params.infer_gan:
        num_param_gen = sum([p.numel() for p in model.generator.parameters()])
        logging.info(f"Number of parameters in generator: {num_param_gen}")
        num_param_disc = sum([p.numel() for p in model.discriminator.parameters()])
        logging.info(f"Number of parameters in discriminator: {num_param_disc}")
    else:
        num_param = sum([p.numel() for p in model.parameters()])
        logging.info(f"Number of parameters: {num_param}")

    params.suffix = f"wav-epoch-{params.epoch}-avg-{params.avg}"
    if params.use_averaged_model:
        params.suffix += "-use-avg-model"

    params.wav_dir_pred = f"{params.exp_dir}/{params.suffix}-pred-step-{params.n_timesteps}"
    os.makedirs(params.wav_dir_pred, exist_ok=True)

    logging.info(params)

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    logging.info(f"Device: {device}")

    if not params.use_averaged_model:
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            logging.info(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        logging.info(
            f"Calculating the averaged model over epoch range from "
            f"{start} (excluded) to {params.epoch}"
        )
        model.to(device)
        model.load_state_dict(
            average_checkpoints_with_averaged_model(
                filename_start=filename_start,
                filename_end=filename_end,
                device=device,
            )
        )

    if params.infer_gan:
        model = model.generator

    model.to(device)
    model.eval()
    cond_module.to(device)
    cond_module.eval()

    recs = RecordingSet.from_file(params.test_recordings).to_eager()
    logging.info(f"num_samples={len(recs)}")
    dataloader = build_data_loader(
        recordings=recs,
        root_path=params.root_path,
        sampling_rate=params.sampling_rate,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        train=False,
        apply_effects=False,
        drop_last=False,
    )

    infer_audio(
        params=params, 
        model=model, 
        cond_module=cond_module,
        dataloader=dataloader,
    )

    logging.info("Done!")


if __name__ == "__main__":
    main()
