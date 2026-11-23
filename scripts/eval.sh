#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_env.sh"
cd "$ROOT/signvip"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
exec "${PYTHON_BIN:-python3}" eval_C5_1_full_test.py \
  --config configs/paper/proposal.yaml \
  --checkpoint-dir "${CHECKPOINT_DIR:-$ROOT/weights/proposed/best}" \
  --test-root "${DATA_ROOT:-$ROOT/data/rwth}/test" \
  --hamer-root "${HAMER_ROOT:-$ROOT/data/hamer}/test" \
  --detail-mask-cache "${MASK_ROOT:-$ROOT/data/detail_masks}" \
  --pose-keypoint-cache "${POSE_ROOT:-$ROOT/data/dwpose_keypoints}" \
  --output-dir "${OUTPUT_DIR:-${OUTPUT_ROOT:-$ROOT/outputs}/eval_proposed}" \
  --signvip-protocol --inference-chunk-size 5 \
  --num-inference-steps 50 --guidance-scale 3.5 --seed 42 \
  --max-samples 0 --save-previews 0 "$@"
