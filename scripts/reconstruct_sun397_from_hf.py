"""Rebuild the original SUN397 directory tree from the HF mirror.

The official download hosts were unreachable at the time of writing. The HF
dataset
1aurent/SUN397 was generated (see its gen_script.py) by feeding
datasets.Image the original file PATHS from the official tar, so every
parquet row carries {bytes, path=original filename} and a label indexing the
original "/a/abbey"-style class paths. That makes the original tree
    sun397/SUN397/<class path>/<filename>
exactly reconstructible, which is what split_zhou_SUN397.json needs.

Shards are deleted from the HF cache after processing, so peak extra disk is
one shard (~450 MB) beyond the images themselves (~39 GB).
"""

import argparse
import io
import json
import os
import urllib.request

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "1aurent/SUN397"
N_SHARDS = 93
OUT = os.path.join(os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter"),
                   "sun397", "SUN397")
INFO = "https://datasets-server.huggingface.co/info?dataset=1aurent/SUN397"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    args = ap.parse_args()

    with urllib.request.urlopen(INFO) as r:
        names = json.load(r)["dataset_info"]["default"]["features"]["label"]["names"]
    assert len(names) == 397, len(names)

    total = 0
    for i in range(args.start, N_SHARDS):
        fn = f"data/train-{i:05d}-of-{N_SHARDS:05d}.parquet"
        p = hf_hub_download(REPO, fn, repo_type="dataset")
        t = pq.read_table(p, columns=["image", "label"])
        for row in t.to_pylist():
            cls = names[row["label"]].lstrip("/")          # a/abbey
            d = os.path.join(OUT, cls)
            os.makedirs(d, exist_ok=True)
            fp = os.path.join(d, row["image"]["path"])
            if not os.path.exists(fp):
                with open(fp, "wb") as fh:
                    fh.write(row["image"]["bytes"])
            total += 1
        os.remove(p)
        print(f"shard {i + 1}/{N_SHARDS} done, {total} images", flush=True)
    print(f"DONE {total} images")


if __name__ == "__main__":
    main()
