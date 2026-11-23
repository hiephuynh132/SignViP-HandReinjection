#!/usr/bin/env python3
"""Visualize matched baseline/final frames with GT-aligned hand and face crops."""

import argparse
import json
import os
import h5py
import numpy as np
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from PIL import Image, ImageDraw
from make_C5_middle_hand_comparison import find_hdf5_item


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SOURCE = ROOT / "make_paper_figure4_c5_comparison.py"
OUTPUT_DIR = REPO / "reports/figures/C5.1_vs_C5.16_hand_face"
OUTPUT = REPO / "reports/figures/C5.1_vs_C5.16_hand_face_comparison.png"
# These are the selected-index preview outputs, generated from the full-test
# cherry-pick list.  Keeping them separate avoids mixing them with the older
# 20-sample smoke previews.
BASELINE = ROOT / "reports/C5.1_top5_cherrypick"
C516 = ROOT / "reports/C5.16_top5_cherrypick"
# Reuse the samples selected during C5.16 full-test evaluation (ranked by
# Hand-SSIM), instead of choosing a new qualitative subset by hand.
C516_SELECTION = ROOT / "reports/C5.16_best_full_test_50steps_rerun/top10_hand/manifest.json"
TEST_ROOT = Path(os.environ.get("DATA_ROOT", str(REPO / "data/rwth"))) / "test"
HAMER_ROOT = Path(os.environ.get("HAMER_ROOT", str(REPO / "data/hamer"))) / "test"


def load_helpers():
    spec = spec_from_file_location("paper_figure_helpers", SOURCE)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric(record, key, position):
    values = record.get(f"{key}_values", [])
    return float(values[position]) if position < len(values) else float("nan")


def outlined_full(helper, array, boxes, size):
    image = helper.resize(array, size)
    draw = ImageDraw.Draw(image)
    sx, sy = size[0] / array.shape[1], size[1] / array.shape[0]
    colors = ((224, 45, 55), (255, 145, 20), (0, 145, 185))
    for box, color in zip(boxes, colors):
        x0, y0, x1, y1 = box
        draw.rectangle((x0 * sx, y0 * sy, x1 * sx, y1 * sy), outline=color, width=3)
    return image


def sample_panel(helper, index, baseline_records, c516_records):
    baseline_record = baseline_records[index]
    c516_record = c516_records[index]
    gt, baseline, middle = helper.preview_pair(BASELINE, baseline_record)
    gt2, c516, middle2 = helper.preview_pair(C516, c516_record)
    if middle != middle2 or baseline_record["frame_ids"][middle] != c516_record["frame_ids"][middle2]:
        raise RuntimeError(f"Frame mismatch for index {index}")
    if abs(gt.astype("float32").mean() - gt2.astype("float32").mean()) > 1e-3:
        raise RuntimeError(f"GT mismatch for index {index}")
    frame_id, hand, face = helper.masks(baseline_record, middle)
    main_path, main_key, _ = find_hdf5_item(TEST_ROOT, baseline_record["text_key"], "video")
    hamer_path, hamer_key, _ = find_hdf5_item(HAMER_ROOT, baseline_record["text_key"], "hamer_video")
    with h5py.File(main_path, "r") as handle:
        pose = np.asarray(handle["pose_video"][main_key][frame_id], dtype="uint8")
    with h5py.File(hamer_path, "r") as handle:
        hamer = np.asarray(handle["hamer_video"][hamer_key][frame_id], dtype="uint8")
    left, right = helper.split_hands(hand)
    boxes = (
        helper.box_from_points(left, hand.shape, padding=7),
        helper.box_from_points(right, hand.shape, padding=7),
        helper.mask_box(face, padding=6),
    )
    arrays = (gt, baseline, c516)
    names = ("Ground truth", "Baseline", "VAE+reinjection")
    panel, full_h, crop_h, gap = 260, 260, 112, 8
    header, sample_header = 92, 48
    width = 3 * panel + 2 * gap
    condition_label_h = 36
    condition_h = condition_label_h + full_h
    height = header + sample_header + condition_h + gap + full_h + 3 * (crop_h + gap) + 12
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    hand_base = metric(baseline_record, "hand_ssim", middle)
    hand_new = metric(c516_record, "hand_ssim", middle)
    face_base = metric(baseline_record, "face_ssim", middle)
    face_new = metric(c516_record, "face_ssim", middle)
    subtitles = (
        "ROI lấy từ GT",
        f"Hand {hand_base:.4f} | Face {face_base:.4f}",
        f"Hand {hand_new:.4f} | Face {face_new:.4f}",
    )
    for column, (name, subtitle) in enumerate(zip(names, subtitles)):
        x = column * (panel + gap)
        helper.centered(draw, (x, 0, x + panel, 46), name, helper.font(15, True))
        helper.centered(draw, (x, 42, x + panel, header), subtitle, helper.font(8))
    # Keep the numeric frame id fully visible; the long video key gets its own
    # smaller line instead of being allowed to clip the header at the right.
    draw.text(
        (5, header + 5),
        f"index={index} | frame_id={frame_id}",
        font=helper.font(9, True), fill=(25, 40, 60),
    )
    draw.text(
        (5, header + 25),
        baseline_record["text_key"],
        font=helper.font(6), fill=(45, 55, 70),
    )
    condition_y = header + sample_header
    draw.text((5, condition_y + 7), "DWPose", font=helper.font(9, True), fill=(20, 90, 150))
    draw.text((panel + gap + 5, condition_y + 7), "HaMeR 3D", font=helper.font(9, True), fill=(20, 90, 150))
    canvas.paste(helper.resize(pose, (panel, full_h)), (0, condition_y + condition_label_h))
    canvas.paste(helper.resize(hamer, (panel, full_h)), (panel + gap, condition_y + condition_label_h))
    y = condition_y + condition_h + gap
    for column, array in enumerate(arrays):
        x = column * (panel + gap)
        canvas.paste(outlined_full(helper, array, boxes, (panel, full_h)), (x, y))
        for row, box in enumerate(boxes):
            x0, y0, x1, y1 = box
            crop = array[y0:y1, x0:x1]
            canvas.paste(helper.resize(crop, (panel, crop_h)), (x, y + full_h + gap + row * (crop_h + gap)))
    labels = (("HAND 1", (224, 45, 55)), ("HAND 2", (225, 120, 10)), ("FACE", (0, 130, 170)))
    for row, (label, color) in enumerate(labels):
        yy = y + full_h + gap + row * (crop_h + gap)
        draw.rectangle((2, yy + 2, 85, yy + 24), fill=(255, 255, 255))
        draw.text((6, yy + 3), label, font=helper.font(9, True), fill=color)
    metadata = {
        "index": index,
        "text_key": baseline_record["text_key"],
        "frame_id": frame_id,
        "position": middle,
        "baseline_hand_ssim": hand_base,
        "c5_16_hand_ssim": hand_new,
        "baseline_face_ssim": face_base,
        "c5_16_face_ssim": face_new,
        "boxes_xyxy": {"hand_1": boxes[0], "hand_2": boxes[1], "face": boxes[2]},
    }
    return canvas, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--index", type=int, default=None,
        help="Render one cherry-picked dataset index; default renders the top five.",
    )
    args = parser.parse_args()
    helper = load_helpers()
    baseline_records = helper.records(BASELINE)
    c516_records = helper.records(C516)
    selection = json.loads(C516_SELECTION.read_text(encoding="utf-8"))
    selected_indices = tuple(int(item["index"]) for item in selection["samples"][:5])
    if args.index is not None:
        if args.index not in selected_indices:
            raise ValueError(
                f"index {args.index} is not among the C5.16 cherry-picked samples: "
                f"{selected_indices}"
            )
        indices = (args.index,)
    else:
        indices = selected_indices
    missing = []
    for index in indices:
        for report, records in ((BASELINE, baseline_records), (C516, c516_records)):
            record = records.get(index)
            previews = list((report / "previews").glob(f"{index:04d}_*.png"))
            if not previews:
                previews = list((report / "top10_hand").glob(f"sample_{index:06d}_*.png"))
            if record is None or len(previews) != 1:
                missing.append(f"{report.name}: index {index}")
    if missing:
        raise RuntimeError(
            "C5.16 cherry-picked samples do not have aligned preview files yet: "
            + ", ".join(missing)
            + ". Run the selected-index eval command first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panels, manifest = [], []
    for index in indices:
        panel, metadata = sample_panel(helper, index, baseline_records, c516_records)
        path = OUTPUT_DIR / f"sample_{index:03d}.png"
        panel.save(path, optimize=True)
        metadata["image"] = str(path.resolve())
        panels.append(panel)
        manifest.append(metadata)
    gap = 16
    master = Image.new("RGB", (panels[0].width, sum(x.height for x in panels) + gap * (len(panels) - 1)), "white")
    y = 0
    for panel in panels:
        master.paste(panel, (0, y)); y += panel.height + gap
    master.save(OUTPUT, optimize=True)
    OUTPUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT.resolve())
    for item in manifest:
        print(item["image"])


if __name__ == "__main__":
    main()
