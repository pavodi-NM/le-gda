"""Fetch the ImageNet subset the pipeline needs (~2-3 GB, not 155 GB).

Source: public mirror evanarlian/imagenet_1k_resized_256.
- val: all 50k images (test 40/class + val 10/class are split later)
- train: 32/class for the draw pool. Train rows are class-sorted, so we
  read parquet row-group footers over HTTP and fetch only row groups whose
  label range still owes classes.

Writes data/tip-adapter/imagenet_subset/{val,train}/<label>/<n>.jpg
"""

import io
import json
import os
import urllib.request

import fsspec
import pyarrow.parquet as pq

BASE = "https://huggingface.co/api/datasets/evanarlian/imagenet_1k_resized_256/parquet/default"
OUT = os.path.join(os.environ.get("LEGDA_DATA_ROOT", "data/tip-adapter"),
                   "imagenet_subset")
POOL_CAP = 32
N_CLS = 1000


def shard_urls(split):
    with urllib.request.urlopen(f"{BASE}/{split}") as r:
        return json.load(r)


def save(split, label, counts, img_bytes):
    d = os.path.join(OUT, split, str(label))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{counts[label]:05d}.jpg"), "wb") as fh:
        fh.write(img_bytes)
    counts[label] += 1


def main():
    fs = fsspec.filesystem("https")

    # ---- val: everything ----
    vd = os.path.join(OUT, "val")
    n_have = sum(len(os.listdir(os.path.join(vd, d)))
                 for d in os.listdir(vd)) if os.path.isdir(vd) else 0
    if n_have == 50000:
        print("val already complete, skipping", flush=True)
    else:
        counts = [0] * N_CLS
        for url in shard_urls("val"):
            with fs.open(url) as f:
                pf = pq.ParquetFile(f)
                for rg in range(pf.num_row_groups):
                    t = pf.read_row_group(rg, columns=["image", "label"])
                    for row in t.to_pylist():
                        save("val", row["label"], counts,
                             row["image"]["bytes"])
            print(f"val shard done, {sum(counts)} images", flush=True)
        assert sum(counts) == 50000, sum(counts)

    # ---- train: 32/class via row-group label ranges ----
    counts = [0] * N_CLS
    need = set(range(N_CLS))
    for url in shard_urls("train"):
        if not need:
            break
        with fs.open(url) as f:
            pf = pq.ParquetFile(f)
            # statistics index LEAF columns (image.bytes, image.path, label)
            li = next(i for i in range(pf.metadata.row_group(0).num_columns)
                      if pf.metadata.row_group(0).column(i)
                      .path_in_schema == "label")
            for rg in range(pf.num_row_groups):
                st = pf.metadata.row_group(rg).column(li).statistics
                lo, hi = st.min, st.max
                if not any(lo <= c <= hi for c in need):
                    continue
                t = pf.read_row_group(rg, columns=["image", "label"])
                for row in t.to_pylist():
                    lab = row["label"]
                    if counts[lab] < POOL_CAP:
                        save("train", lab, counts, row["image"]["bytes"])
                        if counts[lab] == POOL_CAP:
                            need.discard(lab)
        print(f"train shard done, {sum(counts)} pool images, "
              f"{len(need)} classes still open", flush=True)
    assert not need, f"{len(need)} classes under-filled"
    print(f"DONE: 50000 val + {sum(counts)} train images -> {OUT}")


if __name__ == "__main__":
    main()
