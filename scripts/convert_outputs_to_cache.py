"""Convert output files to cache file."""

import argparse
import os
import pickle
import sys
from glob import glob
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

from utils.io_utils import read_json


def create_cache(file):
    outputs = read_json(file)
    model_name = outputs["model_name"]
    system_msg = None
    max_tokens = None
    num_beams = 1

    prompt2key = lambda p, h, n, temperature: (
        p,
        model_name,
        system_msg,
        tuple([tuple(e.items()) for e in h]) if h is not None else None,
        max_tokens,
        temperature,
        num_beams,
        n,
    )

    cache = {}
    for interactions in outputs["interactions"]:
        for interaction in interactions:
            key = prompt2key(
                interaction["query"],
                interaction["history"],
                interaction["n"],
                interaction["temperature"],
            )
            cache[key] = interaction["response"]
    print(f"Created cache of size {len(cache)} for {file}")
    return cache


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_file", type=str, default="cache.pkl")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert not os.path.exists(args.cache_file)
    files = glob(f"{args.output_dir}/*.json")
    cache = {}
    for file in files:
        new_cache = create_cache(file)
        cache.update(new_cache)
    print(f"\nTotal number of files: {len(files)}")
    print(f"Total cache size: {len(cache)}")

    with open(args.cache_file, "wb") as f:
        pickle.dump(cache, f)
    print(f"Saved cache to {args.cache_file}")


if __name__ == "__main__":
    main()
