#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao, Daniel Povey)
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
from typing import Callable, Optional

import torch
from torch import Tensor, nn
from torchaudio import functional as F
from torchaudio.transforms import MelSpectrogram, Spectrogram

from utils import convert_length, make_pad_mask, safe_log


def fft_to_real(fft: torch.Tensor):
    """
    fft: (batch_size, fft_channels, fft_frames), complex.
    Returns: real_fft: (batch_size, 2 * fft_channels, fft_frames), real
    """
    (batch_size, _, fft_frames) = fft.shape
    real_fft = torch.view_as_real(fft).permute(0, 3, 1, 2).reshape(batch_size, -1, fft_frames)
    return real_fft


def real_to_fft(real_fft: torch.Tensor):
    """
    real_fft: (batch_size, 2 * fft_channels, fft_frames), real
    Returns: fft: (batch_size, fft_channels, fft_frames), complex.
    """
    (batch_size, _, fft_frames) = real_fft.shape
    real_fft = real_fft.reshape(batch_size, 2, -1, fft_frames).permute(0, 2, 3, 1)
    fft = torch.view_as_complex(real_fft.contiguous())
    return fft


class STFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: str = "hann_window",
        onesided: bool = True,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.onesided = onesided
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, audio: torch.Tensor, audio_lens: Optional[torch.Tensor] = None):
        spec = torch.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            return_complex=True,
            onesided=self.onesided,
        )
        if audio_lens is not None:
            spec_lens = 1 + torch.div(audio_lens, self.hop_length, rounding_mode="floor")
            assert spec.shape[2] == spec_lens.max().item()
            return spec, spec_lens
        else:
            return spec, None


class ISTFT(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        window: str = "hann_window",
        onesided: bool = True,
        return_complex: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = n_fft
        self.onesided = onesided
        self.return_complex = return_complex
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor):
        audio = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )
        return audio
    
    
class LogMelSpectrogram(torch.nn.Module):
    def __init__(
        self, 
        sampling_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 100,
        center: bool = True,
        power: float = 1,
    ):
        super().__init__()
        
        self.mel = MelSpectrogram(
            sample_rate=sampling_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=center,
            power=power,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel_spec = self.mel(waveform)
        log_mel_spec = safe_log(mel_spec)
        return log_mel_spec


class LinearFilterSpectrogram(torch.nn.Module):
    def __init__(
        self,
        sample_rate: int,
        n_filter: int,
        n_fft: int,
        win_length: Optional[int] = None,
        hop_length: Optional[int] = None,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        pad: int = 0,
        window_fn: Callable[..., torch.Tensor] = torch.hann_window,
        power: float = 2.0,
        normalized: bool = False,
        wkwargs: Optional[dict] = None,
        center: bool = True,
        pad_mode: str = "reflect",
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length if win_length is not None else n_fft
        self.hop_length = hop_length if hop_length is not None else self.win_length // 2
        self.pad = pad
        self.power = power
        self.normalized = normalized
        self.n_filter = n_filter  # Number of (linear) triangular filter
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min

        if f_min > self.f_max:
            raise ValueError("Require f_min: {} <= f_max: {}".format(f_min, self.f_max))

        self.spectrogram = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            pad=self.pad,
            window_fn=window_fn,
            power=self.power,
            normalized=self.normalized,
            wkwargs=wkwargs,
            center=center,
            pad_mode=pad_mode,
            onesided=True,
        )
        
        fb = F.linear_fbanks(
            n_freqs=self.n_fft // 2 + 1,
            f_min=self.f_min,
            f_max=self.f_max,
            n_filter=self.n_filter,
            sample_rate=self.sample_rate,
        )
        self.register_buffer("fb", fb)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform (Tensor): Tensor of audio of dimension (..., time).

        Returns:
            Tensor: Linear filter spectrogram of size (..., n_filters, time).
        """
        # (..., time, freq) dot (freq, n_filter) -> (..., n_filter, time)
        specgram = self.spectrogram(waveform)
        specgram = torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)
        return specgram


class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# From https://github.com/k2-fsa/icefall/blob/master/egs/librispeech/ASR/zipformer/scaling.py
class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        assert max >= min
        ctx.min = min
        ctx.max = max
        return x

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors
        # where x < ctx.min, ensure all grads are negative (this will tend to make
        # x more positive).
        x_grad = x_grad * torch.where(
            torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0
        )
        # where x > ctx.max, ensure all grads are positive (this will tend to make
        # x more negative).
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


def limit_param_value(
    x: Tensor, min: float, max: float, prob: float = 0.6, training: bool = True
):
    # You apply this to (typically) an nn.Parameter during training to ensure that its
    # (elements mostly) stays within a supplied range.  This is done by modifying the
    # gradients in backprop.
    # It's not necessary to do this on every batch: do it only some of the time,
    # to save a little time.
    if training and random.random() < prob:
        return LimitParamValue.apply(x, min, max)
    else:
        return x


class ChannelScale(nn.Module):
    def __init__(self, channels: int, scale: float = 1.0):
        super().__init__()
        self.scale = nn.Parameter(torch.full((channels, 1), scale))

    def forward(self, x: Tensor) -> Tensor:
        # x: (B, C, T)
        scale = limit_param_value(
            self.scale, min=0.5, max=1.0, training=self.training
        )  # for training stabilization
        return x * scale


class BiasNormFunction(torch.autograd.Function):
    # This computes:
    #   scales = (torch.mean((x - bias) ** 2, keepdim=True)) ** -0.5 * log_scale.exp()
    #   return x * scales
    # (after unsqueezing the bias), but it does it in a memory-efficient way so that
    # it can just store the returned value (chances are, this will also be needed for
    # some other reason, related to the next operation, so we can save memory).
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        bias: Tensor,
        log_scale: Tensor,
        channel_dim: int,
        store_output_for_backprop: bool,
    ) -> Tensor:
        assert bias.ndim == 1
        if channel_dim < 0:
            channel_dim = channel_dim + x.ndim
        ctx.store_output_for_backprop = store_output_for_backprop
        ctx.channel_dim = channel_dim
        for _ in range(channel_dim + 1, x.ndim):
            bias = bias.unsqueeze(-1)
        scales = (
            torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
        ) * log_scale.exp()
        ans = x * scales
        ctx.save_for_backward(
            ans.detach() if store_output_for_backprop else x,
            scales.detach(),
            bias.detach(),
            log_scale.detach(),
        )
        return ans

    @staticmethod
    def backward(ctx, ans_grad: Tensor) -> Tensor:
        ans_or_x, scales, bias, log_scale = ctx.saved_tensors
        if ctx.store_output_for_backprop:
            x = ans_or_x / scales
        else:
            x = ans_or_x
        x = x.detach()
        x.requires_grad = True
        bias.requires_grad = True
        log_scale.requires_grad = True
        with torch.enable_grad():
            # recompute scales from x, bias and log_scale.
            scales = (
                torch.mean((x - bias) ** 2, dim=ctx.channel_dim, keepdim=True) ** -0.5
            ) * log_scale.exp()
            ans = x * scales
            ans.backward(gradient=ans_grad)
        return x.grad, bias.grad.flatten(), log_scale.grad, None, None


class BiasNorm(torch.nn.Module):
    """
    This is intended to be a simpler, and hopefully cheaper, replacement for
    LayerNorm.  The observation this is based on, is that Transformer-type
    networks, especially with pre-norm, sometimes seem to set one of the
    feature dimensions to a large constant value (e.g. 50), which "defeats"
    the LayerNorm because the output magnitude is then not strongly dependent
    on the other (useful) features.  Presumably the weight and bias of the
    LayerNorm are required to allow it to do this.

    Instead, we give the BiasNorm a trainable bias that it can use when
    computing the scale for normalization.  We also give it a (scalar)
    trainable scale on the output.

    Args:
       num_channels: the number of channels, e.g. 512.
       channel_dim: the axis/dimension corresponding to the channel,
         interpreted as an offset from the input's ndim if negative.
         This is NOT the num_channels; it should typically be one of
         {-2, -1, 0, 1, 2, 3}.
      log_scale: the initial log-scale that we multiply the output by; this
         is learnable.
      log_scale_min: FloatLike, minimum allowed value of log_scale
      log_scale_max: FloatLike, maximum allowed value of log_scale
      store_output_for_backprop: only possibly affects memory use; recommend
         to set to True if you think the output of this module is more likely
         than the input of this module to be required to be stored for the
         backprop.
    """

    def __init__(
        self,
        num_channels: int,
        channel_dim: int = -1,  # CAUTION: see documentation.
        log_scale: float = 1.0,
        log_scale_min: float = -1.5,
        log_scale_max: float = 1.5,
        store_output_for_backprop: bool = False,
    ) -> None:
        super(BiasNorm, self).__init__()
        self.num_channels = num_channels
        self.channel_dim = channel_dim
        self.log_scale = nn.Parameter(torch.tensor(log_scale))
        self.bias = nn.Parameter(torch.empty(num_channels).normal_(mean=0, std=1e-2))

        self.log_scale_min = log_scale_min
        self.log_scale_max = log_scale_max

        self.store_output_for_backprop = store_output_for_backprop

    def forward(self, x: Tensor) -> Tensor:
        assert x.shape[self.channel_dim] == self.num_channels

        if torch.jit.is_scripting() or torch.jit.is_tracing():
            channel_dim = self.channel_dim
            if channel_dim < 0:
                channel_dim += x.ndim
            bias = self.bias
            for _ in range(channel_dim + 1, x.ndim):
                bias = bias.unsqueeze(-1)
            scales = (
                torch.mean((x - bias) ** 2, dim=channel_dim, keepdim=True) ** -0.5
            ) * self.log_scale.exp()
            return x * scales

        log_scale = limit_param_value(
            self.log_scale,
            min=float(self.log_scale_min),
            max=float(self.log_scale_max),
            training=self.training,
        )

        return BiasNormFunction.apply(
            x, self.bias, log_scale, self.channel_dim, self.store_output_for_backprop
        )


class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.
    """
    def __init__(
        self,
        channels: int = 512,
        hidden_channels: int = 1536,
        conv_kernel_size: int = 7,
        cond_channels: Optional[int] = None,
        time_embed_channels: Optional[int] = None,
        residual_scale: Optional[float] = 1.0,
    ):
        super().__init__()

        assert conv_kernel_size % 2 == 1, conv_kernel_size
        self.dwconv = nn.Conv1d(
            channels,
            channels,
            kernel_size=conv_kernel_size,
            padding=conv_kernel_size // 2,
            groups=channels,
        )
        self.norm = BiasNorm(channels, channel_dim=1)

        self.pwconv1 = nn.Conv1d(channels, hidden_channels, kernel_size=1)
        self.act = nn.PReLU(hidden_channels)
        self.pwconv2 = nn.Conv1d(hidden_channels, channels, kernel_size=1)

        if cond_channels is not None:
            self.cond_proj = nn.Conv1d(cond_channels, channels, kernel_size=1)

        if time_embed_channels is not None:
            self.time_embed_proj = nn.Linear(time_embed_channels, channels)

        if residual_scale is not None:
            self.residual_scale = ChannelScale(channels)

    def forward(
        self,
        x: Tensor,
        cond: Optional[Tensor] = None,
        time_embed: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (batch_size, in_channels, time)
            cond: (batch_size, cond_channels, time)
            time_embed: (batch, channels)
            mask: (batch_size, 1, time)
        
        Returns:
            x: (batch_size, in_channels, time)
        """
        residual = x

        if mask is not None:
            x = x * mask
        x = self.dwconv(x)
        x = self.norm(x)

        # Add condition and time embeddings
        if cond is not None:
            x = x + self.cond_proj(cond)

        if time_embed is not None:
            x = x * (1. + self.time_embed_proj(time_embed).unsqueeze(-1))

        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)

        if hasattr(self, 'residual_scale'):
            residual = self.residual_scale(residual)
        x = x + residual
        
        return x
    

class CondEncoder(nn.Module):
    """ConvNeXt-based Encoder on input conditions, e.g., mels, codec embeddings."""
    def __init__(
        self,
        cond_dim: int = 100,
        channels: int = 512,
        hidden_factor: int = 3,
        conv_kernel_size: int = 7,
        num_layers: int = 4,
        residual_scale: Optional[float] = 1.0,
    ):
        super().__init__()

        self.in_proj = nn.Conv1d(cond_dim, channels, kernel_size=3, padding=1)
        self.in_norm = BiasNorm(channels, channel_dim=1)

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                channels=channels,
                hidden_channels=int(channels * hidden_factor),
                conv_kernel_size=conv_kernel_size,
                residual_scale=residual_scale,
            ) for _ in range(num_layers)
        ])

    def forward(
        self, 
        x: Tensor, 
        mask: Optional[Tensor] = None, 
    ) -> Tensor:
        """
        Args:
            x: (batch_size, n_mels, frames)
            mask: (batch, 1, frames)
        
        Returns:
            x: (batch_size, channels, frames)
        """
        x = self.in_proj(x)
        x = self.in_norm(x)

        for block in self.blocks:
            x = block(x, mask=mask)

        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        channels: int = 512,
        cond_channels: int = 512,
        time_embed_channels: int = 512,
        hidden_factor: int = 3,
        conv_kernel_size: int = 7,
        num_layers: int = 8,
        residual_scale: Optional[float] = 1.0,
        use_t: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_proj = nn.Conv1d(in_channels, channels, kernel_size=1)
        self.in_norm = BiasNorm(channels, channel_dim=1)

        if use_t:
            self.time_embed = SinusoidalPosEmb(time_embed_channels)
            time_embed_hidden = int(time_embed_channels * hidden_factor)
            self.time_mlp = nn.Sequential(
                nn.Linear(time_embed_channels, time_embed_hidden),
                nn.SiLU(),
                nn.Linear(time_embed_hidden, time_embed_channels),
            )

        cond_hidden = int(cond_channels * hidden_factor)
        self.cond_mlp = nn.Sequential(
            nn.Conv1d(cond_channels, cond_hidden, kernel_size=1),
            nn.PReLU(cond_hidden),
            nn.Conv1d(cond_hidden, cond_channels, kernel_size=1),
        )

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(
                channels=channels,
                hidden_channels=int(channels * hidden_factor),
                conv_kernel_size=conv_kernel_size,
                cond_channels=cond_channels,
                time_embed_channels=time_embed_channels if use_t else None,
                residual_scale=residual_scale,
            ) for _ in range(num_layers)
        ])

        self.out_proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(
        self,
        x: Tensor,
        cond: Tensor,
        t: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            x: (batch, in_channels, frame)
            cond: (batch, cond_channels, frame)
            t: (batch,)
            mask: (batch, 1, frame)

        Returns:
            x: (batch, out_channels, frame)
        """
        x = self.in_proj(x)
        x = self.in_norm(x)

        if t is not None:
            time_embed = self.time_mlp(self.time_embed(t))
        else:
            time_embed = None

        cond = self.cond_mlp(cond)

        for block in self.blocks:
            x = block(x, cond=cond, time_embed=time_embed, mask=mask)

        x = self.out_proj(x)

        return x


class AudioConvNeXt(nn.Module):
    """ConvNeXt-based model that processes audio wavforms"""
    def __init__(
        self,
        n_fft: int = 512,
        hop_length: int = 256,
        cond_hop_length: int = 256,
        channels: int = 768,
        cond_channels: int = 512,
        time_embed_channels: int = 512,
        hidden_factor: int = 3,
        conv_kernel_size: int = 7,
        num_layers: int = 8,
        residual_scale: Optional[float] = 1.0,
        use_t: bool = True,
    ):
        super().__init__()

        self.fft = STFT(n_fft=n_fft, hop_length=hop_length)
        self.ifft = ISTFT(n_fft=n_fft, hop_length=hop_length)

        assert cond_hop_length % hop_length == 0, "cond_hop_length should be integer multiple of hop_length."
        self.cond_upsample_factor = cond_hop_length // hop_length

        real_fft_channels = n_fft + 2
        self.decoder = ConvNeXtDecoder(
            in_channels=real_fft_channels,
            out_channels=real_fft_channels,
            channels=channels,
            cond_channels=cond_channels,
            time_embed_channels=time_embed_channels,
            hidden_factor=hidden_factor,
            conv_kernel_size=conv_kernel_size,
            num_layers=num_layers,
            residual_scale=residual_scale,
            use_t=use_t,
        )

    def upsample_cond(self, cond: Tensor, fft_frames: int) -> Tensor:
        """Resample condition, if necessary, to match the FFT coefficients.

        Args:
            cond: (batch_size, cond_channels, cond_frames)
        """
        batch_size, cond_channels, cond_frames = cond.shape
        factor = self.cond_upsample_factor
        if factor != 1:
            cond = cond.unsqueeze(-1).expand(-1, -1, -1, factor)
            cond = cond.reshape(batch_size, cond_channels, cond_frames * factor)
        cond = convert_length(cond, fft_frames)
        return cond

    def forward(
        self,
        audio: Tensor,
        cond: Tensor,
        t: Optional[Tensor] = None,
        audio_lens: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Args:
            audio: (batch, audio_len)
            cond: (batch, cond_channels, cond_frames)
            t: (batch,)
            audio_lens: (batch,)

        Returns: (batch_size, audio_len)
        """
        time = audio.shape[-1]

        fft, fft_lens = self.fft(audio, audio_lens)  
        # fft: (batch, fft_channels, fft_frames); complex.
        fft_real = fft_to_real(fft)  # (batch, 2 * fft_channels, fft_frames)

        cond = self.upsample_cond(cond, fft.shape[-1])  # (batch, cond_channels, fft_frames)

        if fft_lens is not None:
            mask = make_pad_mask(fft_lens).logical_not().unsqueeze(1)  # (batch, 1, fft_frames)
        else:
            mask = None

        fft_real = self.decoder(fft_real, cond=cond, t=t, mask=mask)
        # fft_real shape: (batch, out_channels, fft_frames)

        if mask is not None:
            fft_real = fft_real * mask

        fft = real_to_fft(fft_real)  # complex
        audio = self.ifft(fft)
        audio = convert_length(audio, time)

        return audio