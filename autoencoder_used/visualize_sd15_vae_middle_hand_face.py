#!/usr/bin/env python3
"""Visualize SD1.5 VAE reconstruction on hand-rich middle video frames."""

import argparse
import json
import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from autoencoder_used.eval_autoencoders_used import (
    DiffusersAutoencoderKLFirstStage,
    DETAIL_MASK_SUFFIX,
    is_dense_dataset,
    read_hdf5_text_key,
    resize_rgb_tensor,
    to_model_space,
    to_rgb_space,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--detail-mask-cache", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-hand-fraction", type=float, default=0.005)
    parser.add_argument("--min-face-fraction", type=float, default=0.01)
    return parser.parse_args()


def hdf5_paths(root):
    root = Path(root)
    if root.is_file():
        return [root]
    return sorted(root.glob("*.hdf5")) + sorted(root.glob("*.h5"))


def read_mask_at(path, frame_index, image_size):
    with np.load(path, allow_pickle=False) as data:
        positions = {
            int(frame): index for index, frame in enumerate(data["frame_indices"])
        }
        if frame_index not in positions:
            return None
        position = positions[frame_index]
        hand = torch.from_numpy(np.asarray(data["hands"][position])).float()
        face = torch.from_numpy(np.asarray(data["face"][position])).float()
    masks = torch.stack((hand, face)).unsqueeze(1)
    if tuple(masks.shape[-2:]) != (image_size, image_size):
        masks = F.interpolate(masks, (image_size, image_size), mode="nearest")
    return masks[:, 0] > 0.5


def collect_candidates(hdf5_root, mask_root, image_size):
    mask_dir = Path(mask_root) / "test"
    candidates = []
    for shard in hdf5_paths(hdf5_root):
        with h5py.File(shard, "r") as handle:
            for video_key in sorted(handle["video"], key=int):
                video = handle["video"][video_key]
                num_frames = len(video)
                middle = num_frames // 2
                text_key = read_hdf5_text_key(handle, video_key)
                mask_path = mask_dir / f"{text_key}{DETAIL_MASK_SUFFIX}"
                if not mask_path.is_file():
                    continue
                masks = read_mask_at(mask_path, middle, image_size)
                if masks is None:
                    continue
                hand, face = masks
                candidates.append(
                    {
                        "shard": str(shard),
                        "video_key": str(video_key),
                        "text_key": text_key,
                        "frame_index": middle,
                        "num_frames": num_frames,
                        "hand_fraction": float(hand.float().mean()),
                        "face_fraction": float(face.float().mean()),
                        "hand_mask": hand,
                        "face_mask": face,
                    }
                )
    return candidates


def load_frame(candidate, image_size):
    with h5py.File(candidate["shard"], "r") as handle:
        video = handle["video"][candidate["video_key"]]
        index = candidate["frame_index"]
        raw = video[index] if is_dense_dataset(video) else video[str(index)][()]
    image = Image.fromarray(np.asarray(raw).astype(np.uint8)).convert("RGB")
    tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255.0
    return resize_rgb_tensor(tensor, image_size)


def tensor_image(tensor):
    array = (
        tensor.detach().float().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255
    ).round().astype(np.uint8)
    return Image.fromarray(array, "RGB")


def mask_image(mask):
    array = (mask.detach().cpu().numpy().astype(np.uint8) * 255)
    return Image.fromarray(array, "L").convert("RGB")


def crop_image(image, mask, output_size, padding=8):
    positions = torch.where(mask.detach().cpu().bool())
    if positions[0].numel() == 0:
        return Image.new("RGB", (output_size, output_size), "black")
    y0 = max(0, int(positions[0].min()) - padding)
    y1 = min(image.shape[-2], int(positions[0].max()) + 1 + padding)
    x0 = max(0, int(positions[1].min()) - padding)
    x1 = min(image.shape[-1], int(positions[1].max()) + 1 + padding)
    crop = image[:, y0:y1, x0:x1].unsqueeze(0)
    crop = F.interpolate(crop, (output_size, output_size), mode="bilinear", align_corners=False)[0]
    return tensor_image(crop)


def write_contact_sheet(selected, originals, reconstructions, output_path, image_size):
    headers = [
        "GT", "VAE recon", "|error| x5", "hand mask", "face mask",
        "GT hand", "recon hand", "GT face", "recon face",
    ]
    label_width, header_height = 245, 28
    canvas = Image.new(
        "RGB",
        (label_width + len(headers) * image_size, header_height + len(selected) * image_size),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((label_width + column * image_size + 4, 7), header, fill="black")

    for row, (item, gt, recon) in enumerate(zip(selected, originals, reconstructions)):
        y = header_height + row * image_size
        mse = float((gt - recon).pow(2).mean())
        psnr = -10.0 * math.log10(max(mse, 1e-12))
        label = (
            f"{item['text_key']}\nframe {item['frame_index']}/{item['num_frames'] - 1}\n"
            f"hand={item['hand_fraction']:.3f} face={item['face_fraction']:.3f}\n"
            f"PSNR={psnr:.2f} dB"
        )
        draw.multiline_text((4, y + 8), label, fill="black", spacing=3)
        hand, face = item["hand_mask"], item["face_mask"]
        images = [
            tensor_image(gt),
            tensor_image(recon),
            tensor_image((gt - recon).abs() * 5.0),
            mask_image(hand),
            mask_image(face),
            crop_image(gt, hand, image_size),
            crop_image(recon, hand, image_size),
            crop_image(gt, face, image_size),
            crop_image(recon, face, image_size),
        ]
        for column, image in enumerate(images):
            canvas.paste(image, (label_width + column * image_size, y))
    canvas.save(output_path)


def main():
    args = parse_args()
    if args.num_samples < 1:
        raise ValueError("--num-samples must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    candidates = collect_candidates(args.hdf5, args.detail_mask_cache, args.image_size)
    eligible = [
        item for item in candidates
        if item["hand_fraction"] >= args.min_hand_fraction
        and item["face_fraction"] >= args.min_face_fraction
    ]
    eligible.sort(key=lambda item: item["hand_fraction"], reverse=True)
    selected = eligible[: args.num_samples]
    if len(selected) < args.num_samples:
        raise RuntimeError(
            f"Only {len(selected)} middle frames satisfy hand/face thresholds"
        )

    originals = torch.stack([load_frame(item, args.image_size) for item in selected])
    spec = {
        "name": "sd15_vae_pretrained_128",
        "backend": "diffusers_autoencoder_kl",
        "checkpoint": args.vae,
        "subfolder": "vae",
        "image_size": args.image_size,
    }
    model = DiffusersAutoencoderKLFirstStage(spec).to(device).eval().requires_grad_(False)
    with torch.inference_mode():
        recon_model, _ = model(to_model_space(originals.to(device)))
    reconstructions = to_rgb_space(recon_model).cpu()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vae_reconstruction_middle_hand_face_5.png"
    write_contact_sheet(selected, originals, reconstructions, output_path, args.image_size)

    metadata = []
    for item, gt, recon in zip(selected, originals, reconstructions):
        record = {key: value for key, value in item.items() if not key.endswith("_mask")}
        mse = float((gt - recon).pow(2).mean())
        record["mse"] = mse
        record["psnr"] = -10.0 * math.log10(max(mse, 1e-12))
        metadata.append(record)
    (output_dir / "visualized_samples.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
