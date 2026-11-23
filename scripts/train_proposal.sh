#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_env.sh"
cd "$ROOT/signvip"
PYTHON_BIN="${PYTHON_BIN:-python3}"
export SIGNVIP_WORKSPACE_ROOT="${SIGNVIP_WORKSPACE_ROOT:-$ROOT/outputs/stage1}"
export SIGNVIP_TELEGRAM_BEST_NOTIFY=0 SIGNVIP_TELEGRAM_PROGRESS_STEPS=0
exec "$PYTHON_BIN" -m accelerate.commands.launch \
  --num_processes 1 --num_machines 1 --mixed_precision fp16 \
  --gpu_ids "${GPU_ID:-0}" \
  train_stage1_multi_cond.py --config configs/paper/proposal.yaml "$@"
