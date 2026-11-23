#!/usr/bin/env bash
# Resolve local paths from the checkout; user/container overrides win.
export SD15_ROOT="${SD15_ROOT:-$ROOT/weights/stable-diffusion-v1-5}"
export PAPER_VAE_ROOT="${PAPER_VAE_ROOT:-$ROOT/weights/optimized_vae}"
export PAPER_INIT_ROOT="${PAPER_INIT_ROOT:-$ROOT/weights/stage1_init}"
export DATA_ROOT="${DATA_ROOT:-$ROOT/data/rwth}"
export HAMER_ROOT="${HAMER_ROOT:-$ROOT/data/hamer}"
export MASK_ROOT="${MASK_ROOT:-$ROOT/data/detail_masks}"
export POSE_ROOT="${POSE_ROOT:-$ROOT/data/dwpose_keypoints}"
