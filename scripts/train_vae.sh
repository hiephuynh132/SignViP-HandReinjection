#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_env.sh"
cd "$ROOT/signvip"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
export SIGNVIP_WORKSPACE_ROOT="${SIGNVIP_WORKSPACE_ROOT:-$ROOT/outputs/vae}"
exec "${PYTHON_BIN:-python3}" train_vae_decoder_reconstruction.py \
  --config configs/paper/vae.yaml "$@"
