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


from flow2gan.utils import AttributeDict


def get_generator_config(model_named: str = "mel_24k_base") -> AttributeDict:
    if model_named == "mel_24k_base":
        return AttributeDict(mel_24k_base)
    else:
        raise ValueError(f"Unsupported model name: {model_named}")


mel_24k_base = {
    "sampling_rate": 24000,
    "n_mels": 100,
    "mel_n_fft": 1024,
    "mel_hop_length": 256,
    "n_ffts": (512, 256, 128),
    "hop_lengths": (256, 128, 64),
    "channels": (768, 512, 384),
    "time_embed_channels": 512,
    "hidden_factor": 3,
    "conv_kernel_sizes": (7, 7, 7),
    "num_layers": (8, 8, 8),
    "use_cond_encoder": True,
    "cond_enc_channels": 512,
    "cond_enc_hidden_factor": 3,
    "cond_enc_conv_kernel_size": 7,
    "cond_enc_num_layers": 4,
    "residual_scale": 1.0,
    "init_noise_scale": 0.1,
    "pred_x1": True,
    "branch_reduction": "mean",
    "spec_scaling_loss": True,
    "loss_n_filters": 256,
    "loss_n_fft": 1024,
    "loss_hop_length": 256,
    "loss_power": 0.5,
    "loss_eps": 1e-7,
    "loss_scale_min": 1e-2,
    "loss_scale_max": 1e+2,
    "branch_dropout": 0.05,
    "max_add_noise_scale": 0.0,  
}


def get_gan_config(model_name: str) -> AttributeDict:
    if model_name == "gan_multi_scale_mel_recon":
        return AttributeDict(gan_multi_scale_mel_recon)
    elif model_name == "gan_single_scale_mel_recon":
        return AttributeDict(gan_single_scale_mel_recon)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


gan_multi_scale_mel_recon = {
    "mel_recon_n_ffts": (32, 64, 128, 256, 512, 1024, 2048),
    "mel_recon_n_mels": (5, 10, 20, 40, 80, 160, 320),
}

gan_single_scale_mel_recon = {
    "mel_recon_n_ffts": (1024,),
    "mel_recon_n_mels": (100),
}


HF_REPO = "k2-fsa/Flow2GAN"
HF_MODEL_NAMES = {
    "libritts-mel-1-step": 1,
    "libritts-mel-2-step": 2,
    "libritts-mel-4-step": 4,
    "universal-24k-mel-1-step": 1,
    "universal-24k-mel-2-step": 2,
    "universal-24k-mel-4-step": 4,
}