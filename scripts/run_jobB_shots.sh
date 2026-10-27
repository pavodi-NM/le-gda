#!/usr/bin/env bash
# Proto-Adapter and Proto-Adapter-F at K in {1, 2, 4, 8}, ViT-B/16.
# The ArcFace scale s=64 is the value selected once on validation at K=16 and
# kept fixed for every K (see scripts/proto_adapter.py).
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
