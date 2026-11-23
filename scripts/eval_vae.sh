#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/scripts/common_env.sh"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
# This config is generated from your local paths, never from bundled weights.
CONFIG_FILE="$(mktemp /tmp/signvip-paper-vae-eval.XXXXXX.yaml)"
trap 'unlink "$CONFIG_FILE"' EXIT
"$PYTHON_BIN" - "$CONFIG_FILE" <<'PY'
import os, sys, yaml
from pathlib import Path
specs = []
for name, location in [
    ('pretrained', Path(os.environ.get('SD15_ROOT', 'weights/stable-diffusion-v1-5'))),
    ('optimized', Path(os.environ.get('PAPER_VAE_ROOT', 'weights/optimized_vae'))),
]:
    if not (location / 'vae/config.json').is_file():
        raise SystemExit(f'Missing VAE: {location}/vae/config.json')
    specs.append(dict(name=name,backend='diffusers_autoencoder_kl',
                      checkpoint=str(location.resolve()),subfolder='vae',image_size=128))
with open(sys.argv[1], 'w') as handle:
    yaml.safe_dump({'autoencoders': specs}, handle)
PY
"$PYTHON_BIN" autoencoder_used/eval_autoencoders_used.py \
  --config "$CONFIG_FILE" \
  --hdf5 "${DATA_ROOT:-$ROOT/data/rwth}/test" \
  --detail-mask-cache "${MASK_ROOT:-$ROOT/data/detail_masks}" \
  --detail-mask-split test --detail-mask-strict \
  --max-samples 0 --frames-per-video 999999 --batch-size 16 \
  --device cuda --lpips \
  --output-dir "${OUTPUT_DIR:-${OUTPUT_ROOT:-$ROOT/outputs}/eval_vae}" "$@"
