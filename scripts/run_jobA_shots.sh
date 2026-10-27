#!/usr/bin/env bash
# Tip-Adapter and Tip-Adapter-F (original code under our protocol, main.py) at
# K in {1, 2, 4, 8}, ViT-B/16, 11 datasets, seeds 0 1 2.
#
# Results go to evidence/runB_shots/, not evidence/runB/: the table scripts read
# every file in evidence/runB/ as K=16.
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
