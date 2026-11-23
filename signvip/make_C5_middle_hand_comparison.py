#!/usr/bin/env python3
"""Render one hand-visible middle frame with all C5 Stage-I conditions/results."""

import argparse
import json
import random
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity


MASK_SUFFIX = "_grounded_sam2_face_hands_384_to128_v4.npz"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=628)
    parser.add_argument("--c51-report", type=Path, required=True)
    parser.add_argument("--c52-report", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--hamer-root", type=Path, required=True)
    parser.add_argument("--mask-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref-margin", type=int, default=5)
    parser.add_argument("--scale", type=int, default=2)
    return parser.parse_args()


def read_record(report, index):
    path = report / "per_video_metrics.jsonl"
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if int(record["index"]) == index:
                return record
    raise KeyError(f"Index {index} is absent from {path}")


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode(value.item())
    return str(value)


def find_hdf5_item(root, text_key, required_group):
    for path in sorted(list(root.glob("*.hdf5")) + list(root.glob("*.h5"))):
        with h5py.File(path, "r") as handle:
            for key in handle["text_key"].keys():
                if decode(handle["text_key"][key][()]) == text_key:
                    return path, key, int(handle["len"][key][()])
    raise KeyError(f"Cannot find {text_key} with {required_group} under {root}")


def load_preview_frame(report, record, frame_position):
    preview = next((report / "previews").glob(f"{int(record['index']):04d}_*.png"))
    image = np.asarray(Image.open(preview).convert("RGB"))
    frame_height = image.shape[0] // int(record["num_frames"])
    row = image[frame_position * frame_height : (frame_position + 1) * frame_height]
    midpoint = row.shape[1] // 2
    return row[:, :midpoint].copy(), row[:, midpoint:].copy()


def load_font(size, bold=False):
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def masked_ssim(target, prediction, mask):
    target = target.astype(np.float64) / 255.0
    prediction = prediction.astype(np.float64) / 255.0
    weights = mask.astype(np.float64)
    maps = []
    for channel in range(3):
        _, ssim_map = structural_similarity(
            target[..., channel],
            prediction[..., channel],
            data_range=1.0,
            gaussian_weights=True,
            use_sample_covariance=False,
            full=True,
        )
        maps.append(ssim_map)
    return float((np.mean(maps, axis=0) * weights).sum() / weights.sum())


def overlay_masks(image, hand_mask, face_mask):
    result = image.copy()
    face = face_mask.astype(bool)
    result[face] = np.rint(
        0.45 * result[face] + 0.55 * np.array([30, 180, 255])
    ).astype(np.uint8)
    hand = hand_mask.astype(bool)
    result[hand] = np.rint(
        0.45 * result[hand] + 0.55 * np.array([255, 30, 30])
    ).astype(np.uint8)
    return result


def centered_text(draw, x0, x1, y0, y1, text, font):
    box = draw.textbbox((0, 0), text, font=font)
    width, height = box[2] - box[0], box[3] - box[1]
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2), text, font=font, fill=(20, 20, 20))


def main():
    args = parse_args()
    c51 = read_record(args.c51_report, args.index)
    c52 = read_record(args.c52_report, args.index)
    if (c51["text_key"], c51["frame_ids"]) != (c52["text_key"], c52["frame_ids"]):
        raise RuntimeError("C5.1 and C5.2 records are not aligned")

    text_key = c52["text_key"]
    frame_ids = [int(frame) for frame in c52["frame_ids"]]
    mask_path = args.mask_cache / "test" / f"{text_key}{MASK_SUFFIX}"
    with np.load(mask_path) as data:
        positions = {int(frame): i for i, frame in enumerate(data["frame_indices"])}
        masks = [data["hands"][positions[frame]].astype(bool) for frame in frame_ids]
        face_masks = [data["face"][positions[frame]].astype(bool) for frame in frame_ids]

    middle = len(frame_ids) // 2
    valid_positions = [i for i, mask in enumerate(masks) if mask.any()]
    if not valid_positions:
        raise RuntimeError("Selected video has no GT hand mask")
    # Prefer the exact middle; otherwise choose the hand-visible frame nearest it.
    frame_position = min(valid_positions, key=lambda i: (abs(i - middle), i))
    frame_id = frame_ids[frame_position]
    hand_mask = masks[frame_position]
    face_mask = face_masks[frame_position]

    gt, prediction51 = load_preview_frame(args.c51_report, c51, frame_position)
    gt52, prediction52 = load_preview_frame(args.c52_report, c52, frame_position)
    if not np.array_equal(gt, gt52):
        raise RuntimeError("Ground Truth pixels differ between the two evaluations")

    main_path, main_key, length = find_hdf5_item(args.test_root, text_key, "video")
    hamer_path, hamer_key, hamer_length = find_hdf5_item(args.hamer_root, text_key, "hamer_video")
    if length != hamer_length:
        raise RuntimeError("Main and HAMER frame counts differ")
    with h5py.File(main_path, "r") as main_handle:
        pose = np.asarray(main_handle["pose_video"][main_key][frame_id], dtype=np.uint8)
        forbidden_start = max(0, min(frame_ids) - args.ref_margin)
        forbidden_end = min(length, max(frame_ids) + args.ref_margin + 1)
        reference_ids = list(range(forbidden_start)) + list(range(forbidden_end, length))
        if not reference_ids:
            reference_ids = [i for i in range(length) if i not in set(frame_ids)]
        reference_id = random.Random(args.seed + args.index).choice(reference_ids or [frame_ids[0]])
        reference = np.asarray(main_handle["video"][main_key][reference_id], dtype=np.uint8)
    with h5py.File(hamer_path, "r") as hamer_handle:
        hamer = np.asarray(hamer_handle["hamer_video"][hamer_key][frame_id], dtype=np.uint8)

    panels = (
        (f"Reference (f{reference_id})", reference),
        ("DWPose condition", pose),
        ("HAMER condition", hamer),
        ("Ground Truth", gt),
        ("GT + Hand/Face", overlay_masks(gt, hand_mask, face_mask)),
        ("C5.1 (197k)", prediction51),
        ("C5.2 (27.5k)", prediction52),
    )
    scale = max(1, args.scale)
    panel_width, panel_height = gt.shape[1] * scale, gt.shape[0] * scale
    header, footer, gap = 50, 86, 4
    canvas_width = len(panels) * panel_width + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (canvas_width, header + panel_height + footer), "white")
    draw = ImageDraw.Draw(canvas)
    title_font, body_font = load_font(17, True), load_font(13)
    for i, (label, array) in enumerate(panels):
        x = i * (panel_width + gap)
        centered_text(draw, x, x + panel_width, 0, header, label, title_font)
        panel = Image.fromarray(array).resize((panel_width, panel_height), Image.Resampling.NEAREST)
        canvas.paste(panel, (x, header))

    hand51 = masked_ssim(gt, prediction51, hand_mask)
    hand52 = masked_ssim(gt, prediction52, hand_mask)
    footer_y = header + panel_height
    draw.line((0, footer_y, canvas_width, footer_y), fill=(190, 190, 190))
    draw.text(
        (8, footer_y + 10),
        f"Sample {args.index}: {text_key} | middle position={frame_position}, frame_id={frame_id} | Hand area={hand_mask.mean():.2%} | Face area={face_mask.mean():.2%}",
        font=body_font,
        fill=(20, 20, 20),
    )
    draw.text(
        (8, footer_y + 38),
        f"Masks: HAND=red, FACE=cyan | Masked Hand SSIM: C5.1={hand51:.4f} | C5.2={hand52:.4f} | Paper 0.614 is dataset-level",
        font=body_font,
        fill=(20, 90, 45),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, quality=95)
    print(f"Saved: {args.output.resolve()}")
    print(f"index={args.index}, frame_position={frame_position}, frame_id={frame_id}, hand_area={hand_mask.mean():.6f}, face_area={face_mask.mean():.6f}")
    print(f"C5.1 masked_hand_ssim={hand51:.6f}; C5.2 masked_hand_ssim={hand52:.6f}")


if __name__ == "__main__":
    main()
