#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Daniel Povey, Zengwei Yao)
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


import math
import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn

from modules import AudioConvNeXt, CondEncoder, LinearFilterSpectrogram
from utils import make_pad_mask


class BaseAudioGenerator(nn.Module):
    """Base class of audio generator"""
    def __init__(
        self,
        sampling_rate: int = 24000,
        n_ffts: Tuple[int, ...] = (512, 256, 128),
        hop_lengths: Tuple[int, ...] = (256, 128, 64),
        channels: Tuple[int, ...] = (768, 512, 384),
        time_embed_channels: int = 512,
        hidden_factor: int = 3,
        conv_kernel_sizes: Tuple[int, ...] = (7, 7, 7),
        num_layers: Tuple[int, ...] = (8, 8, 8),
        use_cond_encoder: bool = True,
        cond_dim: int = 100,
        cond_hop_length: int = 256,
        cond_enc_channels: int = 512,
        cond_enc_hidden_factor: int = 3,
        cond_enc_conv_kernel_size: int = 7,
        cond_enc_num_layers: int = 4,
        residual_scale: Optional[float] = 1.0,
        init_noise_scale: float = 0.1,
        pred_x1: bool = True,
        branch_reduction: str = "mean",
        spec_scaling_loss: bool = True,
        loss_n_filters: int = 256,
        loss_n_fft: int = 1024,
        loss_hop_length: int = 256,
        loss_power: float = 0.5,
        loss_eps: float = 1e-7,
        loss_scale_min: float = 1e-2, 
        loss_scale_max: float = 1e+2,
        branch_dropout: float = 0.05,
    ):
        super().__init__()
        self.num_branches = len(n_ffts)
        assert len(hop_lengths) == self.num_branches
        assert len(channels) == self.num_branches
        assert len(conv_kernel_sizes) == self.num_branches
        assert len(num_layers) == self.num_branches

        self.sampling_rate = sampling_rate
        self.init_noise_scale = init_noise_scale 

        self.pred_x1 = pred_x1  # If False, would predict vt

        assert branch_reduction in ("mean", "sum")
        self.branch_reduction = branch_reduction

        self.spec_scaling_loss = spec_scaling_loss
        self.loss_power = loss_power
        self.loss_eps = loss_eps
        self.loss_scale_min = loss_scale_min
        self.loss_scale_max = loss_scale_max
        self.branch_dropout = branch_dropout

        if spec_scaling_loss:
            self.loss_spec = LinearFilterSpectrogram(
                sample_rate=sampling_rate,
                n_fft=loss_n_fft,
                hop_length=loss_hop_length,
                n_filter=loss_n_filters,
                center=True,
                power=2,
            )
        
        if use_cond_encoder:
            self.cond_encoder = CondEncoder(
                cond_dim=cond_dim,
                channels=cond_enc_channels,
                hidden_factor=cond_enc_hidden_factor,
                conv_kernel_size=cond_enc_conv_kernel_size,
                num_layers=cond_enc_num_layers,
                residual_scale=residual_scale,
            )

        self.estimators = nn.ModuleList([
            AudioConvNeXt(
                n_fft=n_ffts[i],
                hop_length=hop_lengths[i],
                cond_hop_length=cond_hop_length,
                channels=channels[i],
                cond_channels=cond_enc_channels if use_cond_encoder else cond_dim,
                time_embed_channels=time_embed_channels,
                hidden_factor=hidden_factor,
                conv_kernel_size=conv_kernel_sizes[i],
                num_layers=num_layers[i],
                residual_scale=residual_scale,
            ) for i in range(self.num_branches)
        ])

        self.apply(self._init_weights)

    @torch.no_grad()
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.015)
            if hasattr(m, 'bias') and isinstance(m.bias, Tensor):
                nn.init.constant_(m.bias, 0)
    
    def process_model(
        self,
        x: Tensor,
        cond: Tensor,
        t: Optional[Tensor] = None,
        audio_lens: Optional[Tensor] = None,
    ) -> Tensor:
        branch_outputs = torch.stack([
            estimator(
                audio=x,
                cond=cond,
                t=t.flatten() if t is not None else None,
                audio_lens=audio_lens,
            ) for estimator in self.estimators
        ], dim=1)  # (batch, num_branches, time)

        if self.training and self.branch_dropout > 0.0 and self.num_branches > 1:
            batch_size = branch_outputs.shape[0]
            device = branch_outputs.device
            dtype = branch_outputs.dtype

            # randomly drop one branch during training
            branch_idx = torch.randint(0, self.num_branches, (batch_size,), device=device)
            mask = torch.ones((batch_size, self.num_branches), device=device, dtype=dtype)
            mask[torch.arange(batch_size, device=device), branch_idx] = 0.0
            mask = mask * (self.num_branches / (self.num_branches - 1))  # rescale

            # for each sample in the batch, with probability branch_dropout, we do the branch dropout
            weight = torch.where(
                torch.rand((batch_size, 1), device=device) < self.branch_dropout, 
                mask, 
                torch.ones_like(mask),
            )
            branch_outputs = branch_outputs * weight.unsqueeze(-1)

        # fuse branch outputs
        output = (
            branch_outputs.mean(dim=1) if self.branch_reduction == "mean" 
            else branch_outputs.sum(dim=1)
        )

        return output

    def compute_loss(
        self,
        pred: Tensor,
        ref: Tensor,
        audio_lens: Tensor,
        gt_audio: Optional[Tensor] = None,
    ) -> Tensor:
        err = pred - ref  # (batch, time) 

        if not self.spec_scaling_loss:
            mask = make_pad_mask(audio_lens).logical_not()
            loss = err ** 2
            loss = (loss * mask).sum() / mask.sum()
        else:
            gt_spec = self.loss_spec(gt_audio) 
            err_spec = self.loss_spec(err)
            # the err_spec.sum() is just an aggregation of squared errors for FFT bins, with
            # frequency-specific weightings.  Scaling by (gt_spec + eps) ** -loss_power is a heuristic
            # scale that puts more weight on quieter regions of the spectrum, where presumably
            # differences would be more audible; the choice of 0.5 for loss_power is arbitrary, it
            # could be anywhere between 0 (no correction for volume) and 1 (fully invariant to
            # local volume).
            spec_lens = torch.div(audio_lens, self.loss_spec.hop_length, rounding_mode="floor") + 1
            assert err_spec.shape[2] == spec_lens.max().item()
            mask = make_pad_mask(spec_lens).logical_not().unsqueeze(1)
            spec_scale = ((gt_spec + self.loss_eps) ** -self.loss_power).clamp(min=self.loss_scale_min, max=self.loss_scale_max)
            loss = err_spec * spec_scale
            loss = (loss * mask).sum() / (mask.sum() * err_spec.shape[1])
        return loss
    
    def forward(
        self,
        x0: Tensor,
        x1: Tensor,
        cond: Tensor,
        audio_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """ Compute Flow Matching loss.
        Args:
            x0: noise, (batch, time)
            x1: ground-truth audio, (batch, time)
            cond: condition features, e.g, mels, (batch, feat_dim, frames)
            audio_lens: lengths of audios, (batch,)
        """
        t = torch.rand((x0.shape[0], 1), device=x0.device, dtype=x0.dtype)
        x = (1.0 - t) * x0 + t * x1
        ref = x1 if self.pred_x1 else (x1 - x0)

        pred = self.process_model(
            x=x,
            cond=cond,
            t=t,
            audio_lens=audio_lens,
        )

        loss = self.compute_loss(
            pred=pred,
            ref=ref,
            audio_lens=audio_lens,
            gt_audio=x1,
        )

        return loss
    
    def infer(
        self,
        noise: Tensor,
        cond: Tensor,
        audio_lens: Optional[Tensor] = None,
        n_timesteps: int = 1,
        clamp_pred: bool = False,
    ) -> Tensor:
        """ Inference with Euler solver.
        Args:
            noise: initial noise, (batch, time)
            cond: condition features, e.g, mels, (batch, feat_dim, frames)
            audio_lens: lengths of audios, (batch,)
            n_timesteps: number of timesteps for ODE solver
            clamp_pred: whether to clamp the predicted audio to [-1.0, 1.0]
        """
        # Use fixed euler solver for ODEs.
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=noise.device)
        t, dt = t_span[0], t_span[1] - t_span[0]
        x = noise
        for step in range(1, len(t_span)):
            pred = self.process_model(
                x=x,
                cond=cond,
                t=t[None, None].expand(noise.shape[0], 1),
                audio_lens=audio_lens,
            )
            vt = (pred - x) / (1 - t) if self.pred_x1 else pred
            x = x + vt * dt
            t = t_span[step]
            
        pred_audio = x
        if clamp_pred:
            pred_audio = pred_audio.clamp(min=-1.0, max=1.0)

        return pred_audio


class MelAudioGenerator(BaseAudioGenerator):
    """Mel-conditioned audio generator."""
    def __init__(
        self,
        n_mels: int = 100,
        mel_hop_length: int = 256,
        max_add_noise_scale: float = 0.0,
        **kwargs,
    ):
        super().__init__(
            cond_dim=n_mels, 
            cond_hop_length=mel_hop_length, 
            **kwargs,
        )
        self.max_add_noise_scale = max_add_noise_scale

    def forward(
        self,
        cond: Tensor,
        audio: Tensor,
        audio_lens: Tensor,
    ) -> Tensor:
        """Compute Flow Matching loss.
        Args:
            cond: mels, (batch, n_mels, frames)
            audio: ground-truth audio, (batch, time)
            audio_lens: lengths of audios, (batch,)
        """
        if self.training and self.max_add_noise_scale > 0.0:
            # add small noise with random scale to mel_spec during training
            e = torch.randn_like(cond) * torch.rand(cond.shape[0], 1, 1, device=cond.device) * self.max_add_noise_scale
            cond = cond + e
    
        if hasattr(self, 'cond_encoder'):
            cond = self.cond_encoder(cond)
        else:
            cond = cond
        
        noise = torch.randn_like(audio) * self.init_noise_scale

        loss = super().forward(
            x0=noise,
            x1=audio,
            cond=cond,
            audio_lens=audio_lens,
        )

        return loss

    def infer(
        self,
        cond: Tensor,
        audio_lens: Optional[Tensor] = None,
        n_timesteps: int = 1,
        clamp_pred: bool = False,
    ) -> Tensor:
        """Inference with Euler solver.
        Args:
            cond: mels, (batch, n_mels, frames)
            audio_lens: lengths of audios, (batch,)
            n_timesteps: number of timesteps for ODE solver
            clamp_pred: whether to clamp the predicted audio to [-1.0, 1.0]
        """
        # In GAN finetuning, we may call infer() in training mode
        if self.training and self.max_add_noise_scale > 0.0:  
            # add small noise with random scale to mel_spec during training
            e = torch.randn_like(cond) * torch.rand(cond.shape[0], 1, 1, device=cond.device) * self.max_add_noise_scale
            cond = cond + e

        if hasattr(self, 'cond_encoder'):
            cond = self.cond_encoder(cond)
        else:
            cond = cond
        
        audio_shape = (cond.shape[0], cond.shape[2] * self.mel_hop_length) 
        noise = torch.randn(audio_shape, device=cond.device, dtype=cond.dtype) * self.init_noise_scale

        pred_audio = super().infer(
            noise=noise,
            cond=cond,
            audio_lens=audio_lens,
            n_timesteps=n_timesteps,
            clamp_pred=clamp_pred,
        )
        
        return pred_audio