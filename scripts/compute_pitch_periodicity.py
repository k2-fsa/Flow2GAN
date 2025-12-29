# Copyright      2025  Xiaomi Corporation     (Author: Zengwei Yao)
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

# Adapted from 
# - https://github.com/descriptinc/cargan/blob/master/cargan/evaluate/objective/metrics.py
# - https://github.com/sh-lee-prml/PeriodWave/blob/main/Eval/pitch_periodicity.py


import argparse
import functools
import json
import logging
import os

import torch
import torchaudio
import torchcrepe
from tqdm import tqdm

from flow2gan.utils import setup_logger


# from https://github.com/descriptinc/cargan/blob/master/config/cargan.py
HOPSIZE = 256
NUM_FFT = 1024
FMIN = 50
FMAX = 550


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gt-wav-dir", type=str, help="Directory to the ground-truth wav files"
    )
    parser.add_argument(
        "--pred-wav-dir", type=str, help="Directory to the predicted wav files"
    )
    parser.add_argument(
        "--wav-list-file", type=str, help="wav list file of test set."
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=24000, help="Sampling rate."
    )
    return parser.parse_args()


# from https://github.com/descriptinc/cargan/blob/master/cargan/preprocess/pitch.py
def from_audio(audio, sample_rate=24000):
    """Preprocess pitch from audio"""
    # Target number of frames
    target_length = audio.shape[1] // HOPSIZE
    
    # Resample
    if sample_rate != torchcrepe.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate,
                                                   torchcrepe.SAMPLE_RATE)
        resampler = resampler.to(audio.device)
        audio = resampler(audio)
    
    # Resample hopsize
    hopsize = int(HOPSIZE * (torchcrepe.SAMPLE_RATE / sample_rate))

    # Pad
    padding = int((NUM_FFT - hopsize) // 2)
    audio = torch.nn.functional.pad(
        audio[None],
        (padding, padding),
        mode='reflect').squeeze(0)

    # Estimate pitch
    pitch, periodicity = torchcrepe.predict(
        audio,
        sample_rate=torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        fmin=FMIN,
        fmax=FMAX,
        model='full',
        return_periodicity=True,
        batch_size=1024,
        device=audio.device,
        pad=False)

    # Set low energy frames to unvoiced
    periodicity = torchcrepe.threshold.Silence()(
        periodicity,
        audio,
        torchcrepe.SAMPLE_RATE,
        hop_length=hopsize,
        pad=False)

    # Potentially resize due to resampled integer hopsize
    if pitch.shape[1] != target_length:
        interp_fn = functools.partial(
            torch.nn.functional.interpolate,
            size=target_length,
            mode='linear',
            align_corners=False)
        pitch = 2 ** interp_fn(torch.log2(pitch)[None]).squeeze(0)
        periodicity = interp_fn(periodicity[None]).squeeze(0)

    return pitch, periodicity


# from https://github.com/sh-lee-prml/PeriodWave/blob/main/Eval/pitch_periodicity.py
def p_p_F(threshold, true_pitch, true_periodicity, pred_pitch, pred_periodicity):
    true_threshold = threshold(true_pitch, true_periodicity)
    pred_threshold = threshold(pred_pitch, pred_periodicity)
    true_voiced = ~torch.isnan(true_threshold)
    pred_voiced = ~torch.isnan(pred_threshold)

    # Update periodicity rmse
    count = true_pitch.shape[1]
    periodicity_total = (true_periodicity - pred_periodicity).pow(2).sum()

    # Update pitch rmse
    voiced = true_voiced & pred_voiced
    voiced_sum = voiced.sum()

    difference_cents = 1200 * (torch.log2(true_pitch[voiced]) -
                               torch.log2(pred_pitch[voiced]))
    pitch_total = difference_cents.pow(2).sum()

    # Update voiced/unvoiced precision and recall
    true_positives = (true_voiced & pred_voiced).sum()
    false_positives = (~true_voiced & pred_voiced).sum()
    false_negatives = (true_voiced & ~pred_voiced).sum()

    pitch_rmse = torch.sqrt(pitch_total / voiced_sum)
    periodicity_rmse = torch.sqrt(periodicity_total / count)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * precision * recall / (precision + recall)

    # return pitch_rmse.nan_to_num().item(), periodicity_rmse.item(), f1.nan_to_num().item()
    return pitch_rmse, periodicity_rmse, f1


def evaluate(
    gt_wav_dir: str,
    pred_wav_dir: str,
    wav_list_file: str,
    out_file: str,
    sampling_rate: int = 24000,
):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    
    threshold = torchcrepe.threshold.Hysteresis()

    results = []

    with open(wav_list_file) as f:
        wav_list = f.read().splitlines()
    for fname in tqdm(wav_list):
        gt_wav_file = os.path.join(gt_wav_dir, fname)
        pred_wav_file = os.path.join(pred_wav_dir, fname)

        # load audio
        gt_wav, sr = torchaudio.load(gt_wav_file)
        assert sr == sampling_rate
        pred_wav, sr = torchaudio.load(pred_wav_file)
        assert sr == sampling_rate
        
        # trim to equal length
        min_len = min(gt_wav.shape[1], pred_wav.shape[1])
        gt_wav = gt_wav[:, :min_len].to(device)  # (T)
        pred_wav = pred_wav[:, :min_len].to(device)  # (T)

        true_pitch, true_periodicity = from_audio(gt_wav, sampling_rate)
        fake_pitch, fake_periodicity = from_audio(pred_wav, sampling_rate)

        pitch, periodicity, f1 = p_p_F(threshold, true_pitch, true_periodicity, fake_pitch, fake_periodicity)
        if torch.isnan(pitch) or torch.isnan(f1):
            logging.warning(f"All frames are unvoiced for {fname}, skipping...")
            continue
        pitch = pitch.item()
        periodicity = periodicity.item()
        f1 = f1.item()
            
        res = {
            "fname": fname,
            "pitch": pitch,
            "periodicity": periodicity,
            "f1": f1,
        }
        results.append(res)

    total = {k: 0.0 for k in results[0].keys() if k != "fname"}
    cnt = 0
    for data in results:
        cnt += 1
        for k, v in data.items():
            if k != "fname":
                total[k] += v
    avg = {k: v / cnt for k, v in total.items()}
    logging.info(f"Number of samples: {cnt}")
    logging.info(f"Averaged metrics: {avg}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Saved to: {out_file}")

    logging.info("Done!")


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    args = get_args()
    setup_logger(f"{args.pred_wav_dir}/log_metrics_pitch_periodicity")

    logging.info(
        f"Start computing metrics for pred_wav_dir={args.pred_wav_dir}, "
        f"wav_list_file={args.wav_list_file}"
    )

    out_file = os.path.join(args.pred_wav_dir, "metrics_pitch_periodicity.json")
    evaluate(
        gt_wav_dir=args.gt_wav_dir,
        pred_wav_dir=args.pred_wav_dir,
        wav_list_file=args.wav_list_file,
        out_file=out_file,
        sampling_rate=args.sampling_rate,
    )
