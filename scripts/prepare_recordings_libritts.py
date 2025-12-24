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
import os

from lhotse import RecordingSet


def get_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--root-dir",
        type=str,
        default="download/libritts/LibriTTS/",
        help="Path to the LibriTTS root directory",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/manifests/libritts",
        help="Output directory to store recording manifests",
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        default=10,
        help="Number of parallel jobs to scan audio files",
    )
    return parser.parse_args()


def main():
    args = get_args()

    parts = [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]

    os.makedirs(args.out_dir, exist_ok=True)
    for p in parts:
        print(f"Processing {p}")
        recs = RecordingSet.from_dir(
            f"{args.root_dir}/{p}", pattern="*.wav", num_jobs=args.num_jobs
        )
        print(f"Number of samples: {len(recs)}")
        recs.to_file(f"{args.out_dir}/recordings_{p}.jsonl.gz")
    print("Done")


if __name__ == "__main__":
    main()
