#!/usr/bin/env bash
# Job B -- Proto-Adapter + Proto-Adapter-F shot sweep, ViT-B/16 only.
# s=64 is FROZEN (U1, S1.12b): never re-selected per K.
set -euo pipefail
cd "$(dirname "$0")/.."
for K in 1 2 4 8; do
  echo "=== K=$K  $(date -Is) ==="
  PYTHONPATH=. python scripts/run_proto_adapter.py \
    --protocol ours --arms free ft --scale 64 --seeds 0 1 2 \
    --backbones ViT-B/16 --k "$K" \
    --out "evidence/proto_adapter/ours_ViT-B-16_K${K}.json"
done
echo "=== JOB B DONE $(date -Is) ==="
