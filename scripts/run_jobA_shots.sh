#!/usr/bin/env bash
# Job A -- Tip-Adapter + Tip-Adapter-F, UPSTREAM main.py, shot sweep,
# ViT-B/16 only. Requires patches/tip_adapter_our_protocol.patch (now carrying
# the --shots hunk) and patches/tip_adapter_searchhp_hoist.patch to be applied.
#
# Output goes to evidence/runB_shots/, NOT evidence/runB/: scripts/
# update_runB_status.py globs `evidence/runB/runB_*.json` and keys on
# (backbone, dataset) only, so K<16 files dropped in there would corrupt the
# status block. Separate directory, glob untouched.
#
# --test-split testfull for all 11, INCLUDING imagenet: that is what Run B and
# the proto-adapter K=16 grid both used (imagenet testfull = 40000, `test` =
# 30000 capped), and the K=16 endpoints of these curves come from those files.
set -euo pipefail
cd "$(dirname "$0")/.."
DS="caltech101 dtd eurosat fgvc food101 oxford_flowers oxford_pets stanford_cars sun397 ucf101 imagenet"
for K in 1 2 4 8; do
  for d in $DS; do
    out="evidence/runB_shots/runB_${d}_ViT-B-16_K${K}.json"
    if [ -s "$out" ]; then echo "skip $out"; continue; fi
    echo "=== $d K=$K  $(date -Is) ==="
    PYTHONPATH=. python main.py --config "configs/${d}.yaml" \
      --backbone ViT-B/16 --shots "$K" --seeds 0 1 2 \
      --test-split testfull --out "$out"
  done
done
echo "=== JOB A DONE $(date -Is) ==="
