#!/usr/bin/env python3
# Copyright    2025  Xiaomi Corp.             (authors: Zengwei Yao)
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


import logging
import os
import numpy as np
from typing import Optional, Tuple

import torch
import torchaudio
from lhotse import RecordingSet
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def pad_seq_collate_fn(data, filter_silence: bool = True):
    # each item in data: (audio: Tensor, silence: bool, file_name: str)
    if filter_silence:
        new_data = [x for x in data if not x[1]]  # remove silent items
        if len(new_data) == 0:
            new_data = data[0:1]
            logging.warning(
                "No non-silent audio in the batch, using the first item as fallback."
            )
    else:
        new_data = data
    audios = pad_sequence([torch.Tensor(x[0]) for x in new_data], batch_first=True)
    audio_lens = torch.tensor([len(x[0]) for x in new_data], dtype=torch.int32)
    file_names = [x[2] for x in new_data]
    return audios, audio_lens, file_names


def build_data_loader(
    recordings: RecordingSet,
    root_path: Optional[str] = None,
    sampling_rate: int = 24000,
    batch_size: int = 256,
    num_workers: int = 4,
    train: bool = False,
    duration: Optional[float] = None,
    apply_effects: bool = True,
    max_load_times: int = 1,
    world_size: int = 1,
    pin_memory: bool = True,
    persistent_workers: bool = True,
    drop_last: bool = False,
):
    """Build a DataLoader, each batch is (audios, audio_lens, file_names)
    """
    dataset = LhotseRecordingDataset(
        recordings=recordings,
        root_path=root_path,
        sampling_rate=sampling_rate,
        train=train,
        duration=duration,
        apply_effects=apply_effects,
        max_load_times=max_load_times,
    )

    shuffle = train

    if world_size > 1:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if sampler is None else None,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=pad_seq_collate_fn,
        drop_last=drop_last,
    )
    return dataloader


class LhotseRecordingDataset(Dataset):
    def __init__(
        self,
        recordings: RecordingSet,
        sampling_rate: int = 24000,
        root_path: Optional[str] = None,
        train: bool = False,
        duration: Optional[float] = None,
        apply_effects: bool = True,
        max_load_times: int = 1,
        min_rms: float = 0.005,
    ):
        if recordings.is_lazy:
            recordings = recordings.to_eager()
        self.recordings = recordings
        self.sampling_rate = sampling_rate
        self.root_path = root_path
        self.train = train
        self.duration = duration
        self.apply_effects = apply_effects
        self.max_load_times = max_load_times
        self.min_rms = min_rms

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, bool, str]:
        recording = self.recordings[index]
        sr = recording.sampling_rate

        fname = recording.sources[0].source
        if self.root_path is not None:
            fname = os.path.relpath(fname, self.root_path)

        def is_silence(x):
            return np.sqrt(np.mean(x ** 2)) < self.min_rms

        silence = False
        if self.duration is None:
            # Load the whole audio
            y = recording.load_audio()
            silence = is_silence(y)
        else:
            duration = min(self.duration, recording.duration)
            if not self.train:
                # In validation, always take the first segment
                y = recording.load_audio(offset=0.0, duration=duration)
                silence = is_silence(y)
            else:
                # In training, try to take a random segment with rms over a target value
                times = 0
                while times < self.max_load_times:
                    times += 1
                    offset = np.random.uniform(0, recording.duration - duration)
                    y = recording.load_audio(offset=offset, duration=duration)
                    silence = is_silence(y)
                    if not silence:
                        break

        y = torch.from_numpy(y)

        if y.ndim == 1:
            y = y.unsqueeze(0)  # (N,) -> (1, N)

        if y.shape[0] > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)

        if self.apply_effects:
            gain = np.random.uniform(-1, -6) if self.train else -3
            y, _ = torchaudio.sox_effects.apply_effects_tensor(
                y, sr, [["norm", f"{gain:.2f}"]]
            )

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(
                y, orig_freq=sr, new_freq=self.sampling_rate
            )

        return y[0], silence, fname
