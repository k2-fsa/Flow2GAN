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

import torch

from checkpoint import (
    average_checkpoints,
    average_checkpoints_with_averaged_model,
    load_checkpoint,
)
from pretrain import get_cond_module_and_generator
from utils import AttributeDict, str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--epoch",
        type=int,
        default=200,
        help="""It specifies the checkpoint to use for decoding.
        Note: Epoch counts from 1.
        You can specify --avg to use more checkpoints for model averaging.""",
    )

    parser.add_argument(
        "--avg",
        type=int,
        default=20,
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
        default="./exp-pretrain",
        help="The experiment dir.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mel_24k_base",
        help="""Name of the model to use. Supported names are 
        `mel_24k_base`.""",
    )

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    params = AttributeDict(vars(args))

    params.suffix = f"epoch-{params.epoch}-avg-{params.avg}"
    if params.use_averaged_model:
        params.suffix += "-use-avg-model"

    params.saved_model_path = f"{params.exp_dir}/{params.suffix}.pt"

    device = torch.device("cpu")
    print(f"Device: {device}")

    print(params)

    print("About to create model")
    _, model = get_cond_module_and_generator(params)

    num_param = sum([p.numel() for p in model.parameters()])
    print(f"Number of parameters: {num_param}")

    if not params.use_averaged_model:
        if params.avg == 1:
            load_checkpoint(f"{params.exp_dir}/epoch-{params.epoch}.pt", model)
        else:
            start = params.epoch - params.avg + 1
            filenames = []
            for i in range(start, params.epoch + 1):
                if i >= 1:
                    filenames.append(f"{params.exp_dir}/epoch-{i}.pt")
            print(f"averaging {filenames}")
            model.to(device)
            model.load_state_dict(average_checkpoints(filenames, device=device))
    else:
        assert params.avg > 0, params.avg
        start = params.epoch - params.avg
        assert start >= 1, start
        filename_start = f"{params.exp_dir}/epoch-{start}.pt"
        filename_end = f"{params.exp_dir}/epoch-{params.epoch}.pt"
        print(
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
    
    torch.save({"model": model.state_dict()}, params.saved_model_path)
    print(f"Saved to {params.saved_model_path}")


if __name__ == "__main__":
    main()
