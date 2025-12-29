# Copyright      2024  Xiaomi Corporation     (Author: Zengwei Yao)
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

# Modified from
# - https://github.com/bfs18/rfwave/blob/main/calculate_voc_metrics.py
# - https://github.com/gemelo-ai/vocos/blob/main/vocos/experiment.py
# - https://github.com/descriptinc/audiotools/blob/master/audiotools/metrics/quality.py


import argparse
import logging
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch
import torchaudio
from auraloss.freq import MultiResolutionSTFTLoss
from pesq import pesq

from flow2gan.utils import setup_logger, str2bool


def visqol(
    estimate: np.ndarray,
    reference: np.ndarray,
    sample_rate: int = 16000,
    mode: str = "speech",
):  
    from visqol import visqol_lib_py
    from visqol.pb2 import visqol_config_pb2
    from visqol.pb2 import similarity_result_pb2

    config = visqol_config_pb2.VisqolConfig()
    if mode == "audio":
        target_sr = 48000
        assert sample_rate == target_sr
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == "speech":
        target_sr = 16000
        assert sample_rate == target_sr
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    config.audio.sample_rate = target_sr
    config.options.svr_model_path = os.path.join(
        os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path
    )

    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    _visqol = api.Measure(reference.astype(np.float64), estimate.astype(np.float64))
    return _visqol.moslqo


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
    parser.add_argument(
        "--use-visqol",
        type=str2bool,
        default=False,
        help="Whether to compute ViSQOL. Would be slower."
    )
    parser.add_argument(
        "--n-proc", type=int, default=1, help="Number of processes."
    )
    return parser.parse_args()


def compute_metrics(
    fname: str,
    gt_wav_file: str,
    pred_wav_file: str,
    sampling_rate: int = 24000,
    use_visqol: bool = False,
):
    device = torch.device("cpu")
    mrstft_fun = MultiResolutionSTFTLoss(sample_rate=sampling_rate, device=device)

    # load audio
    gt_wav, sr = torchaudio.load(gt_wav_file)
    assert sr == sampling_rate
    pred_wav, sr = torchaudio.load(pred_wav_file)
    assert sr == sampling_rate

    # trim to equal length
    min_len = min(gt_wav.shape[1], pred_wav.shape[1])
    gt_wav = gt_wav[0, :min_len].to(device)  # (T)
    pred_wav = pred_wav[0, :min_len].to(device)  # (T)

    # multi-resolution STFT loss
    mrstft_loss = mrstft_fun(pred_wav[None, None], gt_wav[None, None])

    # resample to 16k
    gt_wav_16k = torchaudio.functional.resample(gt_wav, orig_freq=sr, new_freq=16000)
    pred_wav_16k = torchaudio.functional.resample(pred_wav, orig_freq=sr, new_freq=16000)

    # pesq
    pesq_score = pesq(
        16000, gt_wav_16k.cpu().numpy(), pred_wav_16k.cpu().numpy(), 'wb', on_error=1
    )
    
    if use_visqol:
        try:
            if min_len < 16000: 
                pred_wav_16k_visqol = torch.nn.functional.pad(pred_wav_16k, (0, 16000 - min_len), value=0.0)
                gt_wav_16k_visqol = torch.nn.functional.pad(gt_wav_16k, (0, 16000 - min_len), value=0.0)
            else:
                pred_wav_16k_visqol = pred_wav_16k
                gt_wav_16k_visqol = gt_wav_16k

            visqol_score = visqol(
                estimate=pred_wav_16k_visqol.cpu().numpy(), 
                reference=gt_wav_16k_visqol.cpu().numpy(), 
                mode="speech", 
                sample_rate=16000,
            )
        except Exception as e:
            print(e)
            print(fname)
            print(pred_wav_16k.shape, gt_wav_16k.shape)

    def convert(x):
        x = x.item() if torch.is_tensor(x) else x
        return float(x)

    res = {
        "fname": fname,
        "mrstft_loss": convert(mrstft_loss),
        "pesq_score": convert(pesq_score),
    }
    if use_visqol:
        res["visqol_score"] = convert(visqol_score)
    return res


def evaluate(
    gt_wav_dir: str,
    pred_wav_dir: str,
    wav_list_file: str,
    out_file: str,
    sampling_rate: int = 24000,
    use_visqol: bool = False,
    n_proc: int = 1,
):
    with open(wav_list_file) as f:
        wav_list = f.read().splitlines()

    futures = []
    with ProcessPoolExecutor(n_proc, mp.get_context("fork")) as pool:
        for fname in wav_list:
            gt_wav_file = os.path.join(gt_wav_dir, fname)
            pred_wav_file = os.path.join(pred_wav_dir, fname)
            future = pool.submit(
                compute_metrics,
                fname, gt_wav_file, pred_wav_file, sampling_rate, use_visqol
            )
            futures.append(future)

        results = []
        for future in futures:
            results.append(future.result())

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
    args = get_args()
    setup_logger(f"{args.pred_wav_dir}/log_metrics")

    logging.info(
        f"Start computing metrics for pred_wav_dir={args.pred_wav_dir}, "
        f"wav_list_file={args.wav_list_file}"
    )
    logging.info(f"n_proc={args.n_proc}")
    out_file = os.path.join(args.pred_wav_dir, "metrics.json")
    evaluate(
        gt_wav_dir=args.gt_wav_dir,
        pred_wav_dir=args.pred_wav_dir,
        wav_list_file=args.wav_list_file,
        out_file=out_file,
        sampling_rate=args.sampling_rate,
        use_visqol=args.use_visqol,
        n_proc=args.n_proc,
    )
