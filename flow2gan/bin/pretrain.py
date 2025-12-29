# Adapted from https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/train.py
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
import copy
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
from lhotse import RecordingSet
from lhotse.utils import fix_random_seed
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.tensorboard import SummaryWriter

from flow2gan import diagnostics
from flow2gan.checkpoint import load_checkpoint, save_checkpoint, update_averaged_model
from flow2gan.dataset import LhotseRecordingDataset, build_data_loader, pad_seq_collate_fn
from flow2gan.dist import cleanup_dist, setup_dist
from flow2gan.env import get_env_info
from flow2gan.err import raise_grad_scale_is_too_small_error
from flow2gan.hooks import register_inf_check_hooks
from flow2gan.models.config import get_generator_config
from flow2gan.models.generator import MelAudioGenerator
from flow2gan.optim import Eden2, LRScheduler, ScaledAdam
from flow2gan.utils import (
    AttributeDict, 
    MetricsTracker, 
    get_parameter_groups_with_lrs, 
    plot_feature, 
    setup_logger, 
    str2bool, 
    to_float_tuple, 
    to_int_tuple,
)


LRSchedulerType = Union[torch.optim.lr_scheduler._LRScheduler, LRScheduler]


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="Number of GPUs for DDP training.",
    )

    parser.add_argument(
        "--master-port",
        type=int,
        default=12354,
        help="Master port to use for DDP training.",
    )

    parser.add_argument(
        "--tensorboard",
        type=str2bool,
        default=True,
        help="Should various information be logged in tensorboard.",
    )

    parser.add_argument(
        "--num-epochs",
        type=int,
        default=200,
        help="Number of epochs to train.",
    )

    parser.add_argument(
        "--start-epoch",
        type=int,
        default=1,
        help="""Resume training from this epoch. It should be positive.
        If larger than 1, it will load checkpoint from
        exp-dir/epoch-{start_epoch-1}.pt
        """,
    )

    parser.add_argument(
        "--exp-dir",
        type=str,
        default="./exp-pretrain",
        help="""The experiment dir.
        It specifies the directory where all training related
        files, e.g., checkpoints, log, etc, are saved
        """,
    )

    parser.add_argument(
        "--base-lr", type=float, default=0.035, help="The base learning rate."
    )

    parser.add_argument(
        "--lr-batches",
        type=float,
        default=7500,
        help="""Number of steps that affects how rapidly the learning rate
        decreases. We suggest not to change this.""",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The seed for random generators intended for reproducibility",
    )

    parser.add_argument(
        "--print-diagnostics",
        type=str2bool,
        default=False,
        help="Accumulate stats on activations, print them and exit.",
    )

    parser.add_argument(
        "--inf-check",
        type=str2bool,
        default=False,
        help="Add hooks to check for infinite module outputs and gradients.",
    )

    parser.add_argument(
        "--save-every-n",
        type=int,
        default=1,
        help="""Save checkpoint after processing this number of epochs"
        periodically. We save checkpoint to exp-dir/ whenever
        params.cur_epoch % save_every_n == 0. The checkpoint filename
        has the form: f'exp-dir/epoch-{params.cur_epoch}.pt'.
        Since it will take around 1000 epochs, we suggest using a large
        save_every_n to save disk space.
        """,
    )

    parser.add_argument(
        "--average-period",
        type=int,
        default=200,
        help="""Update the averaged model, namely `model_avg`, after processing
        this number of batches. `model_avg` is a separate version of model,
        in which each floating-point parameter is the average of all the
        parameters from the start of training. Each time we take the average,
        we do: `model_avg = model * (average_period / batch_idx_train) +
            model_avg * ((batch_idx_train - average_period) / batch_idx_train)`.
        """,
    )

    parser.add_argument(
        "--use-fp16",
        type=str2bool,
        default=False,
        help="Whether to use half precision training.",
    )

    parser.add_argument(
        "--train-recordings",
        type=str,
        help="""Manifest files for training set. You can provide multiple files
        separated by commas; each file will be used to create a separate dataloader.""",
    )

    parser.add_argument(
        "--train-dls-weights",
        type=str,
        default=None,
        help="""Weights to sample different dataloaders, separated by commas.
        Each weight corresponds to a dataloader specified in --train-recordings.""",
    )

    parser.add_argument(
        "--valid-recordings",
        type=str,
        help="""Manifest files for validation set. You can provide multiple files
        separated by commas; each file will be used to create a separate dataloader.""",
    )

    parser.add_argument(
        "--test-recordings",
        type=str,
        help="Manifest file for test set.",
    )

    parser.add_argument(
        "--save-infer-steps",
        type=str,
        default="2,4,8",
        help="Inference steps used when saving samples, separated by commas",
    )

    parser.add_argument(
        "--max-load-times",
        type=int,
        default=3,
        help="Maximum number of attempts to randomly load a non-silent segment from an audio file.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.5,
        help="Segment duration for training and validation",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training and validation",
    )

    parser.add_argument(
        "--train-num-workers",
        type=int,
        default=4,
        help="How many subprocesses to use for training data loading.",
    )

    parser.add_argument(
        "--valid-num-workers",
        type=int,
        default=1,
        help="How many subprocesses to use for validation data loading.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="mel_24k_base",
        help="""Name of the model to use. Supported names are 
        `mel_24k_base`, `mel_24k_small`.""",
    )

    return parser


def get_train_params() -> AttributeDict:
    params = AttributeDict(
        {
            "batch_idx_train": -1,  
            "log_interval": 50,
            "valid_interval": 2000,
            "env_info": get_env_info(),
        }
    )
    return params


def load_checkpoint_if_available(
    params: AttributeDict,
    model: nn.Module,
    model_avg: nn.Module,
) -> Optional[Dict[str, Any]]:
    """Load checkpoint from file.

    If params.start_epoch is larger than 1, it will load the checkpoint from
    `params.start_epoch - 1`.

    Apart from loading state dict for `model` and `optimizer`.

    Args:
      params:
        The return value of :func:`get_params`.
      model:
        The training model.
    Returns:
      Return a dict containing previously saved training info.
    """
    if params.start_epoch > 1:
        filename = params.exp_dir / f"epoch-{params.start_epoch - 1}.pt"
    else:
        return None

    assert filename.is_file(), f"{filename} does not exist!"

    saved_params = load_checkpoint(
        filename,
        model=model,
        model_avg=model_avg,
    )

    keys = ["batch_idx_train"]
    for k in keys:
        params[k] = saved_params[k]

    return saved_params


def get_cond_module_and_generator(params: AttributeDict) -> Tuple[nn.Module, nn.Module]:
    """Create condition module and generator model."""
    model_cfg = get_generator_config(params.model_name)
    logging.info(model_cfg)
    
    params.sampling_rate = model_cfg.sampling_rate

    if "mel" in params.model_name:
        from flow2gan.models.modules import LogMelSpectrogram
        cond_module = LogMelSpectrogram(
            sampling_rate=model_cfg.sampling_rate,
            n_fft=model_cfg.mel_n_fft,
            hop_length=model_cfg.mel_hop_length,
            n_mels=model_cfg.n_mels,
            center=True,
            power=1,
        )
        generator = MelAudioGenerator(**model_cfg)
    else:
        raise ValueError(f"Unsupported model name: {params.model_name}")
    
    return cond_module, generator


def compute_loss(
    audio: torch.Tensor,
    audio_lens: torch.Tensor,
    cond_module: nn.Module,
    model: Union[nn.Module, DDP],
    is_training: bool = True,
) -> Tuple[torch.Tensor, MetricsTracker]:
    """Compute loss given the model and its inputs."""
    with torch.set_grad_enabled(is_training):
        cond = cond_module(audio)
        loss = model(cond=cond, audio=audio, audio_lens=audio_lens)
    assert loss.requires_grad == is_training

    batch_size = audio.shape[0]
    loss_info = MetricsTracker()
    loss_info["samples"] = batch_size
    loss_info["loss"] = loss.detach().item() * batch_size
    
    return loss, loss_info


def train_one_epoch(
    params: AttributeDict,
    cond_module: nn.Module,
    model: Union[nn.Module, DDP],
    optimizer: Optimizer,
    scheduler: LRSchedulerType,
    train_dls: List[torch.utils.data.DataLoader],
    valid_dls: List[torch.utils.data.DataLoader],
    test_ds: LhotseRecordingDataset,
    scaler: GradScaler,
    model_avg: Optional[nn.Module] = None,
    tb_writer: Optional[SummaryWriter] = None,
    world_size: int = 1,
    rank: int = 0,
) -> None:
    """Train the model for one epoch.

    The training loss from the mean of all frames is saved in
    `params.train_loss`. It runs the validation process every
    `params.valid_interval` batches.

    Args:
      params:
        It is returned by :func:`get_params`.
      model:
        The model for training.
      cond_module:
        The module used to extract input condition.
      optimizer:
        The optimizer.
      train_dl:
        Dataloader for the training dataset.
      valid_dl:
        Dataloader for the validation dataset.
      scaler:
        The scaler used for mix precision training.
      tb_writer:
        Writer to write log messages to tensorboard.
    """
    model.train()
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device

    def save_bad_model(suffix: str = ""):
        save_checkpoint(
            filename=params.exp_dir / f"bad-model{suffix}-{rank}.pt",
            model=model,
            model_avg=model_avg,
            params=params,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            rank=0,
        )

    saved_bad_model = False

    train_dls = [iter(dl) for dl in train_dls]
    num_dls = len(train_dls)

    if params.train_dls_weights is not None:
        train_dls_weights = to_float_tuple(params.train_dls_weights)
        assert len(train_dls_weights) == num_dls
    else:
        train_dls_weights = (1.0,) * num_dls

    batch_idx = 0
    # used to track the stats over iterations in one epoch
    tot_losses = [MetricsTracker() for _ in range(num_dls)]

    while True:
        dl_idx = random.choices(list(range(num_dls)), weights=train_dls_weights, k=1)[0]
        dl = train_dls[dl_idx]

        try:
            batch = next(dl)
        except StopIteration:
            logging.info(f"Reach end of dataloader {dl_idx}")
            break

        batch_idx += 1
        params.batch_idx_train += 1

        audio = batch[0].to(device)
        audio_lens = batch[1].to(device)
        # audio: (N, T), float32
        batch_size = audio.shape[0]

        try:
            with autocast(enabled=params.use_fp16):
                # forward discriminator
                loss, loss_info = compute_loss(
                    audio=audio,
                    audio_lens=audio_lens,
                    cond_module=cond_module,
                    model=model,
                    is_training=True,
                )

            # summary stats
            tot_losses[dl_idx] += loss_info

            scaler.scale(loss).backward()
            scheduler.step_batch(params.batch_idx_train)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        except Exception as e:
            logging.info(f"Caught exception: {e}.")
            save_bad_model()
            raise

        if params.print_diagnostics and batch_idx == 5:
            return

        # Update model_avg at every average_period steps
        if (
            rank == 0
            and params.batch_idx_train > 0
            and params.batch_idx_train % params.average_period == 0
        ):
            update_averaged_model(
                params=params,
                model_cur=model,
                model_avg=model_avg,
            )

        if params.use_fp16:
            cur_grad_scale = scaler._scale.item()

            if cur_grad_scale < 0.01:
                if not saved_bad_model:
                    save_bad_model(suffix="-first-warning")
                    saved_bad_model = True
                    if not params.inf_check:
                        register_inf_check_hooks(model)
                logging.warning(f"Grad scale is small: {cur_grad_scale}")

            if cur_grad_scale < 1.0e-05:
                save_bad_model()
                raise_grad_scale_is_too_small_error(cur_grad_scale)

            # If the grad scale was less than 1, try increasing it. The _growth_interval
            # of the grad scaler is configurable, but we can't configure it to have different
            # behavior depending on the current grad scale.
            if (
                batch_idx % 25 == 0
                and cur_grad_scale < 2.0
                or batch_idx % 100 == 0
                and cur_grad_scale < 8.0
                or batch_idx % 400 == 0
                and cur_grad_scale < 32.0
            ):
                scaler.update(cur_grad_scale * 2.0)

        if params.batch_idx_train % params.log_interval == 0:
            cur_lr = max(scheduler.get_last_lr())
            cur_grad_scale = scaler._scale.item() if params.use_fp16 else 1.0

            logging.info(
                f"Epoch {params.cur_epoch}, batch {batch_idx}, "
                f"global_batch_idx: {params.batch_idx_train}, batch size: {batch_size}, "
                f"dataloader: {dl_idx}, loss_{dl_idx}[{loss_info}], "
                f"tot_loss_{dl_idx}[{tot_losses[dl_idx]}], "
                f"cur_lr: {cur_lr:.2e}, "
                + (f"grad_scale: {scaler._scale.item()}" if params.use_fp16 else "")
            )

            if tb_writer is not None:
                tb_writer.add_scalar("train/learning_rate", cur_lr, params.batch_idx_train)
                loss_info.write_summary(tb_writer, f"train/current_{dl_idx}_", params.batch_idx_train)
                tot_losses[dl_idx].write_summary(tb_writer, f"train/tot_{dl_idx}_", params.batch_idx_train)
                if params.use_fp16:
                    tb_writer.add_scalar("train/grad_scale", cur_grad_scale, params.batch_idx_train)

        if params.batch_idx_train % params.valid_interval == 0 and not params.print_diagnostics:
            model.eval()

            logging.info("Computing validation loss")
            valid_infos = compute_validation_loss(
                cond_module=cond_module,
                model=model,
                valid_dls=valid_dls,
                world_size=world_size,
            )

            logging.info(f"Epoch {params.cur_epoch}, validation: ")
            for i, info in enumerate(valid_infos):
                logging.info(f"dataloader {i}: {info}")

            if tb_writer is not None:
                for i, info in enumerate(valid_infos):
                    info.write_summary(tb_writer, f"train/valid_{i}_", params.batch_idx_train)

                save_test_samples(
                    params=params,
                    cond_module=cond_module,
                    model=model,
                    test_ds=test_ds,
                    tb_writer=tb_writer,
                )

            model.train()
            logging.info(f"Maximum memory allocated so far is {torch.cuda.max_memory_allocated() // 1000000}MB")


def compute_validation_loss(
    cond_module: nn.Module,
    model: Union[nn.Module, DDP],
    valid_dls: List[torch.utils.data.DataLoader],
    world_size: int = 1,
) -> List[MetricsTracker]:
    """Run the validation process."""
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    tot_losses = []

    with torch.no_grad():
        for i, dl in enumerate(valid_dls):
            logging.info(f"Processing dataloader {i}")
            # used to summary the stats over iterations
            tot_loss = MetricsTracker()

            for batch in dl:
                audio = batch[0].to(device)
                audio_lens = batch[1].to(device)
                # audio: (N, T), float32

                loss, loss_info = compute_loss(
                    audio=audio,
                    audio_lens=audio_lens,
                    cond_module=cond_module,
                    model=model,
                    is_training=False,
                )
                assert loss.requires_grad is False
                # summary stats
                tot_loss = tot_loss + loss_info

            if world_size > 1:
                tot_loss.reduce(device)

            tot_losses.append(tot_loss)

    return tot_losses


def save_test_samples(
    params: AttributeDict,
    cond_module: nn.Module,
    model: Union[nn.Module, DDP],
    test_ds: LhotseRecordingDataset,
    tb_writer: SummaryWriter,
):
    device = model.device if isinstance(model, DDP) else next(model.parameters()).device
    inner_model = model.module if isinstance(model, DDP) else model

    samples = [sample for sample in test_ds]
    audio, audio_lens, file_names = pad_seq_collate_fn(samples, filter_silence=False)
    num_samples = audio.shape[0]

    save_infer_steps = to_int_tuple(params.save_infer_steps)
        
    results = [
        {"gt": audio[i][:audio_lens[i].item()].numpy()}
        for i in range(num_samples)
    ]

    audio = audio.to(device)
    audio_lens = audio_lens.to(device)

    logging.info("Infering test samples")

    with torch.no_grad():
        cond = cond_module(audio)
        for step in save_infer_steps:
            pred_audio = inner_model.infer(
                cond=cond,
                audio_lens=audio_lens,
                n_timesteps=step,
                clamp_pred=True,
            )
            pred_audio = pred_audio.data.cpu().numpy()
            for i in range(num_samples):
                results[i][f"step_{step}"] = pred_audio[i][:audio_lens[i].item()]

    logging.info("Saving to tensorboard")

    def compute_spec(y):
        stft = librosa.stft(y, n_fft=1024, hop_length=256)
        return librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    for i in range(num_samples):
        gt_audio = results[i]["gt"]
        tb_writer.add_audio(
            f"valid/test_audio_{i}_gt",
            gt_audio,
            params.batch_idx_train,
            params.sampling_rate,
        )
        tb_writer.add_image(
            f"valid/test_audio_{i}_gt_spec",
            plot_feature(compute_spec(gt_audio)),
            params.batch_idx_train,
            dataformats="HWC",
        )
        for step in save_infer_steps:
            pred_audio = results[i][f"step_{step}"]
            tb_writer.add_audio(
                f"valid/test_audio_{i}_step_{step}",
                pred_audio,
                params.batch_idx_train,
                params.sampling_rate,
            )
            tb_writer.add_image(
                f"valid/test_audio_{i}_step_{step}_spec",
                plot_feature(compute_spec(pred_audio)),
                params.batch_idx_train,
                dataformats="HWC",
            )


def prepare_data_loaders(params: AttributeDict, world_size: int):
    logging.info("Building training dataloader")
    train_dls = []
    for i, f_rec in enumerate(params.train_recordings.split(",")):
        recs = RecordingSet.from_file(f_rec).to_eager()
        recs = recs.filter(lambda r: r.duration > 0.1)
        logging.info(f"Dataloader {i}: num_samples={len(recs)}")
        dl = build_data_loader(
            recordings=recs,
            sampling_rate=params.sampling_rate,
            batch_size=params.batch_size,
            num_workers=params.train_num_workers,
            train=True,
            duration=params.duration,
            apply_effects=True,
            max_load_times=params.max_load_times,
            world_size=world_size,
        )
        train_dls.append(dl)

    logging.info("Building validation dataloader")
    valid_dls = []
    for i, f_rec in enumerate(params.valid_recordings.split(",")):
        recs = RecordingSet.from_file(f_rec).to_eager()
        recs = recs.filter(lambda r: r.duration > 0.1)
        logging.info(f"Dataloader {i}: num_samples={len(recs)}")
        dl = build_data_loader(
            recordings=recs,
            sampling_rate=params.sampling_rate,
            batch_size=params.batch_size,
            num_workers=params.valid_num_workers,
            train=False,
            duration=params.duration,
            apply_effects=True,
            world_size=world_size,
        )
        valid_dls.append(dl)

    logging.info("Building test dataset")
    recs = RecordingSet.from_file(params.test_recordings)
    logging.info(f"Number of samples: {len(recs)}")
    test_ds = LhotseRecordingDataset(
        recordings=recs,
        sampling_rate=params.sampling_rate,
        train=False,
        duration=None,
        apply_effects=False,
    )

    return train_dls, valid_dls, test_ds


def run(rank, world_size, args):
    """
    Args:
      rank:
        It is a value between 0 and `world_size-1`, which is
        passed automatically by `mp.spawn()` in :func:`main`.
        The node with rank 0 is responsible for saving checkpoint.
      world_size:
        Number of GPUs for DDP training.
      args:
        The return value of get_parser().parse_args()
    """
    params = get_train_params()
    params.update(vars(args))

    fix_random_seed(params.seed)
    if world_size > 1:
        setup_dist(rank, world_size, params.master_port)

    setup_logger(f"{params.exp_dir}/log/log-train")
    logging.info("Training started")

    if args.tensorboard and rank == 0 and not params.print_diagnostics:
        tb_writer = SummaryWriter(log_dir=f"{params.exp_dir}/tensorboard")
    else:
        tb_writer = None

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", rank)
    logging.info(f"Device: {device}")

    logging.info(params)

    logging.info("About to create model")
    cond_module, model = get_cond_module_and_generator(params)
    logging.info(model)

    num_param = sum([p.numel() for p in model.parameters()])
    logging.info(f"Number of parameters: {num_param}")

    model_avg: Optional[nn.Module] = None
    if rank == 0:
        # model_avg is only used with rank 0
        model_avg = copy.deepcopy(model).to(torch.float64)

    assert params.start_epoch > 0, params.start_epoch
    checkpoints = load_checkpoint_if_available(
        params=params,
        model=model,
        model_avg=model_avg,
    )

    cond_module.to(device)
    model.to(device)
    if world_size > 1:
        logging.info("Using DDP")
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = ScaledAdam(
        get_parameter_groups_with_lrs(model, lr=params.base_lr, include_names=True),
        lr=params.base_lr,  # should have no effect
        clipping_scale=2.0,
    )
    scheduler = Eden2(optimizer, params.lr_batches, warmup_start=0.1)

    if checkpoints is not None:
        # load state_dict for optimizer and scheduler
        if "optimizer" in checkpoints:
            logging.info("Loading optimizer state dict")
            optimizer.load_state_dict(checkpoints["optimizer"])
        if "scheduler" in checkpoints:
            logging.info("Loading scheduler state dict")
            scheduler.load_state_dict(checkpoints["scheduler"])

    if params.print_diagnostics:
        opts = diagnostics.TensorDiagnosticOptions(
            512
        )  # allow 4 megabytes per sub-module
        diagnostic = diagnostics.attach_diagnostics(model, opts)

    if params.inf_check:
        register_inf_check_hooks(model)

    train_dls, valid_dls, test_ds = prepare_data_loaders(params, world_size)

    scaler = GradScaler(enabled=params.use_fp16, init_scale=1.0)
    if checkpoints and "grad_scaler" in checkpoints:
        logging.info("Loading grad scaler state dict")
        scaler.load_state_dict(checkpoints["grad_scaler"])

    for epoch in range(params.start_epoch, params.num_epochs + 1):
        logging.info(f"Start epoch {epoch}")

        fix_random_seed(params.seed + epoch - 1)
        if world_size > 1:
            # Calling the set_epoch() method on the DistributedSampler
            for dl in train_dls:
                dl.sampler.set_epoch(epoch - 1)

        if tb_writer is not None:
            tb_writer.add_scalar("train/epoch", epoch, params.batch_idx_train)

        params.cur_epoch = epoch

        train_one_epoch(
            params=params,
            cond_module=cond_module,
            model=model,
            model_avg=model_avg,
            optimizer=optimizer,
            scheduler=scheduler,
            train_dls=train_dls,
            valid_dls=valid_dls,
            test_ds=test_ds,
            scaler=scaler,
            tb_writer=tb_writer,
            world_size=world_size,
            rank=rank,
        )

        if params.print_diagnostics:
            diagnostic.print_diagnostics()
            break

        if epoch % params.save_every_n == 0 or epoch == params.num_epochs:
            filename = params.exp_dir / f"epoch-{params.cur_epoch}.pt"
            save_checkpoint(
                filename=filename,
                params=params,
                model=model,
                model_avg=model_avg,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                rank=rank,
            )

    logging.info("Done!")

    if world_size > 1:
        torch.distributed.barrier()
        cleanup_dist()


def main():
    parser = get_parser()
    args = parser.parse_args()
    args.exp_dir = Path(args.exp_dir)

    world_size = args.world_size
    assert world_size >= 1
    if world_size > 1:
        mp.spawn(run, args=(world_size, args), nprocs=world_size, join=True)
    else:
        run(rank=0, world_size=1, args=args)


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    main()
