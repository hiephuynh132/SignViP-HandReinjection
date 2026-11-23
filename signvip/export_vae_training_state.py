#!/usr/bin/env python3
"""Export a decoder-only VAE training checkpoint as a Diffusers VAE."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
# The repository may contain an older xFormers build that is incompatible with
# the CUDA/Triton stack used for thesis VAE training. AutoencoderKL does not
# need xFormers for export, so prevent Diffusers from importing it.
import diffusers.utils.import_utils as diffusers_import_utils

diffusers_import_utils._xformers_available = False
diffusers_import_utils._xformers_version = None
from diffusers import AutoencoderKL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-vae", required=True, type=Path)
    parser.add_argument("--training-state", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state_path = args.training_state.resolve()
    output_dir = args.output_dir.resolve()
    vae_dir = output_dir / "vae"

    if not state_path.is_file():
        raise FileNotFoundError(f"Training state does not exist: {state_path}")
    if not (args.base_vae / "config.json").is_file():
        raise FileNotFoundError(f"Base VAE is invalid: {args.base_vae}")

    state = torch.load(state_path, map_location="cpu")
    required = {"step", "post_quant_conv", "decoder"}
    missing = required.difference(state)
    if missing:
        raise KeyError(f"Training state is missing keys: {sorted(missing)}")

    vae = AutoencoderKL.from_pretrained(
        str(args.base_vae.resolve()), local_files_only=True
    )
    vae.post_quant_conv.load_state_dict(state["post_quant_conv"], strict=True)
    vae.decoder.load_state_dict(state["decoder"], strict=True)
    vae.save_pretrained(vae_dir, safe_serialization=True)

    metadata = {
        "step": int(state["step"]),
        "source_training_state": str(state_path),
        "base_vae": str(args.base_vae.resolve()),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "export.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Exported step {metadata['step']} to {vae_dir}")


if __name__ == "__main__":
    main()
