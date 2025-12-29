#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (author: Zengwei Yao)
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


import random
from typing import List, Optional, Tuple

import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram

from flow2gan.models.discriminators import MultiPeriodDiscriminator, MultiResolutionDiscriminator
from flow2gan.utils import safe_log


class GAN(nn.Module):
    def __init__(
        self,
        generator: nn.Module,
        mel_recon_n_ffts: Tuple[int, ...] = (32, 64, 128, 256, 512, 1024, 2048),
        mel_recon_n_mels: Tuple[int, ...] = (5, 10, 20, 40, 80, 160, 320),
    ):
        super().__init__()
        self.generator = generator

        mp_disc = MultiPeriodDiscriminator()
        mr_disc = MultiResolutionDiscriminator()
        self.discriminator = nn.ModuleList([mp_disc, mr_disc])

        self.mel_recon_modules = nn.ModuleList()
        for n_fft, n_mels in zip(mel_recon_n_ffts, mel_recon_n_mels):
            self.mel_recon_modules.append(
                MelSpectrogram(
                    sample_rate=generator.sampling_rate,
                    n_fft=n_fft,
                    hop_length=n_fft // 4,
                    n_mels=n_mels,
                    center=True,
                    power=1,
                )
            )

    def discriminator_loss(
        self, 
        score_real: List[Tensor], 
        score_fake: List[Tensor],
    ):
        loss = 0
        for s_real, s_fake in zip(score_real, score_fake):
            loss += torch.mean(torch.clamp(1 - s_real, min=0))
            loss += torch.mean(torch.clamp(1 + s_fake, min=0))
        return loss

    def generator_loss(
        self, 
        score_fake: List[Tensor],
    ):
        loss = 0
        for s_fake in score_fake:
            loss += torch.mean(torch.clamp(1 - s_fake, min=0))
        return loss

    def feature_matching_loss(
        self, 
        fmap_real: List[Tensor], 
        fmap_fake: List[Tensor],
    ):
        loss = 0
        for f_real, f_fake in zip(fmap_real, fmap_fake):
            assert isinstance(f_real, List) and isinstance(f_fake, List)
            for r, f in zip(f_real, f_fake):
                loss += nn.functional.l1_loss(r.detach(), f)
        return loss

    def mel_recon_loss(
        self, 
        real: Tensor, 
        fake: Tensor,
    ):
        loss = 0
        for module in self.mel_recon_modules:
            real_mel = safe_log(module(real))
            fake_mel = safe_log(module(fake))
            loss += nn.functional.l1_loss(real_mel, fake_mel)
        return loss

    def forward(
        self,
        cond: Tensor,
        audio: Tensor,
        audio_lens: Optional[Tensor] = None,
        n_timesteps: int = 1,
        train_disc: bool = True,
    ):
        if train_disc:
            # train discriminator at this step
            self.discriminator.train()
            self.generator.eval()

            with torch.no_grad():
                pred_audio = self.generator.infer(
                    cond=cond,
                    audio_lens=audio_lens,
                    n_timesteps=n_timesteps,
                    clamp_pred=False,
                )

            score_real_mp, score_fake_mp, _, _ = self.discriminator[0](
                y=audio, y_hat=pred_audio
            )
            score_real_mr, score_fake_mr, _, _ = self.discriminator[1](
                y=audio, y_hat=pred_audio
            )

            disc_loss_mp = self.discriminator_loss(score_real=score_real_mp, score_fake=score_fake_mp)
            disc_loss_mr = self.discriminator_loss(score_real=score_real_mr, score_fake=score_fake_mr)

            return disc_loss_mp, disc_loss_mr
        else:
            # train generator at this step
            self.discriminator.eval()
            self.generator.train()

            pred_audio = self.generator.infer(
                cond=cond,
                audio_lens=audio_lens,
                n_timesteps=n_timesteps,
                clamp_pred=False,
            )

            _, score_fake_mp, fmap_real_mp, fmap_fake_mp = self.discriminator[0](
                y=audio, y_hat=pred_audio
            )
            _, score_fake_mr, fmap_real_mr, fmap_fake_mr = self.discriminator[1](
                y=audio, y_hat=pred_audio
            )

            gen_loss_mp = self.generator_loss(score_fake_mp)
            gen_loss_mr = self.generator_loss(score_fake_mr)

            feat_map_loss_mp = self.feature_matching_loss(fmap_real=fmap_real_mp, fmap_fake=fmap_fake_mp)
            feat_map_loss_mr = self.feature_matching_loss(fmap_real=fmap_real_mr, fmap_fake=fmap_fake_mr)

            mel_recon_loss = self.mel_recon_loss(real=audio, fake=pred_audio)

            return (
                gen_loss_mp,
                gen_loss_mr,
                feat_map_loss_mp,
                feat_map_loss_mr,
                mel_recon_loss,
            )
