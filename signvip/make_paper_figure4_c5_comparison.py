#!/usr/bin/env python3
"""Build the aligned qualitative comparison used as Figure 4 in paper_take_note.md."""

import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
MASK_ROOT = Path(os.environ.get("MASK_ROOT", str(REPO / "data/detail_masks"))) / "test"
MASK_SUFFIX = "_grounded_sam2_face_hands_384_to128_v4.npz"
REPORTS = (
    ("Baseline", ROOT / "reports/C5.1_best197000_full_test_50steps_hand_face"),
    ("ROI loss", ROOT / "reports/C5.5_best4000_full_test_50steps"),
    ("VAE+diffusion", ROOT / "reports/C5.16_unfreeze_unet_condition_full_test"),
)
# Fixed qualitative subset selected by the C5.16 full-test evaluator.  The
# first five entries are retained for the paper figure; no re-ranking is done
# while generating the visualization.
C516_SELECTION = ROOT / "reports/C5.16_best_full_test_50steps_rerun/top10_hand/manifest.json"
LOCALITY_IMAGE = ROOT / "reports/vae_hand_latent_locality_C5_13/vae_hand_latent_noise_outside_full.png"
OUTPUT = REPO / "reports/figures/figure4_c5_qualitative_full_hand_face.png"


FIG_FONT_SCALE = 2


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", int(size * FIG_FONT_SCALE))


def records(report):
    output = {}
    with (report / "per_video_metrics.jsonl").open() as handle:
        for line in handle:
            item = json.loads(line)
            output[int(item["index"])] = item
    return output


def preview_pair(report, record):
    matches = sorted((report / "previews").glob(f"{int(record['index']):04d}_*.png"))
    # Selected-index evals may retain the same real|generated contact sheet in
    # top10_hand without duplicating it under previews.
    if len(matches) == 0:
        matches = sorted((report / "top10_hand").glob(f"sample_{int(record['index']):06d}_*.png"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one preview for index {record['index']} in {report}")
    image = np.asarray(Image.open(matches[0]).convert("RGB"))
    frame_count = int(record["num_frames"])
    # top10_hand previews include a 24-pixel title strip; regular previews do
    # not. Detect and remove that strip before indexing the frame grid.
    title_height = 0
    if image.shape[0] >= 24 and (image.shape[0] - 24) % frame_count == 0:
        title_height = 24
    frame_height = (image.shape[0] - title_height) // frame_count
    middle = frame_count // 2
    start = title_height + middle * frame_height
    row = image[start : start + frame_height]
    split = row.shape[1] // 2
    return row[:, :split].copy(), row[:, split:].copy(), middle


def vae_reconstruction(row):
    # Layout is fixed by probe_vae_hand_latent_locality.py: label=180, panel=180,
    # header=74, footer=56; VAE reconstruction is column 1.
    image = Image.open(LOCALITY_IMAGE).convert("RGB")
    x0, y0 = 180 + 180, 74 + row * (180 + 56)
    return np.asarray(image.crop((x0, y0, x0 + 180, y0 + 180)).resize((128, 128), Image.Resampling.LANCZOS))


def masks(record, middle):
    frame_id = int(record["frame_ids"][middle])
    path = MASK_ROOT / f"{record['text_key']}{MASK_SUFFIX}"
    with np.load(path) as data:
        positions = {int(frame): i for i, frame in enumerate(data["frame_indices"])}
        position = positions[frame_id]
        return frame_id, data["hands"][position].astype(bool), data["face"][position].astype(bool)


def components(mask):
    seen = np.zeros_like(mask, dtype=bool)
    found = []
    height, width = mask.shape
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        stack, points = [(int(y), int(x))], []
        seen[y, x] = True
        while stack:
            cy, cx = stack.pop()
            points.append((cy, cx))
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))
        if len(points) >= 4:
            found.append(points)
    return sorted(found, key=len, reverse=True)


def split_hands(mask):
    parts = components(mask)[:2]
    if len(parts) >= 2:
        parts.sort(key=lambda pts: np.mean([x for _, x in pts]))
        return parts
    points = [(int(y), int(x)) for y, x in np.argwhere(mask)]
    if not points:
        return ([], [])
    median = float(np.median([x for _, x in points]))
    return ([p for p in points if p[1] <= median], [p for p in points if p[1] > median])


def box_from_points(points, shape, padding=7):
    height, width = shape
    if not points:
        return (0, 0, width, height)
    ys, xs = zip(*points)
    x0, x1 = max(0, min(xs) - padding), min(width, max(xs) + padding + 1)
    y0, y1 = max(0, min(ys) - padding), min(height, max(ys) + padding + 1)
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x0, y0 = min(max(x0, 0), width - side), min(max(y0, 0), height - side)
    return (x0, y0, min(width, x0 + side), min(height, y0 + side))


def mask_box(mask, padding=7):
    return box_from_points([(int(y), int(x)) for y, x in np.argwhere(mask)], mask.shape, padding)


def resize(array, size):
    return Image.fromarray(array).resize(size, Image.Resampling.LANCZOS)


def centered(draw, box, text, text_font):
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
    x0, y0, x1, y1 = box
    draw.text((x0 + (x1 - x0 - width) / 2, y0 + (y1 - y0 - height) / 2), text, font=text_font, fill=(20, 20, 20))


def main():
    selection = json.loads(C516_SELECTION.read_text(encoding="utf-8"))
    indices = tuple(int(item["index"]) for item in selection["samples"][:5])
    report_records = [records(path) for _, path in REPORTS]
    panel, full_h, crop_h, gap = 220, 220, 104, 6
    labels = ("GT", REPORTS[0][0], REPORTS[1][0], "VAE-only", REPORTS[2][0])
    header, sample_header = 94, 38
    sample_height = sample_header + full_h + 3 * crop_h + 4 * gap
    width = len(labels) * panel + (len(labels) - 1) * gap
    height = header + len(indices) * sample_height + (len(indices) - 1) * 14
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    centered(draw, (0, 0, width, 52), "Aligned qualitative comparison", font(16, True))
    for column, label in enumerate(labels):
        x = column * (panel + gap)
        centered(draw, (x, 42, x + panel, header), label, font(14, True))

    manifest = []
    y = header
    for row_index, index in enumerate(indices):
        anchor = report_records[0][index]
        gt, baseline, middle = preview_pair(REPORTS[0][1], anchor)
        frame_id, hand, face = masks(anchor, middle)
        _, roi_loss, other_middle = preview_pair(REPORTS[1][1], report_records[1][index])
        _, c516, c516_middle = preview_pair(REPORTS[2][1], report_records[2][index])
        if other_middle != middle or c516_middle != middle:
            raise RuntimeError(f"Middle-frame mismatch at index {index}")
        vae = vae_reconstruction(row_index)
        images = (gt, baseline, roi_loss, vae, c516)
        draw.text(
            (6, y + 8),
            f"index={index} | {anchor['text_key']} | frame_id={frame_id}",
            font=font(11, True), fill=(30, 41, 59),
        )
        y0 = y + sample_header
        left_points, right_points = split_hands(hand)
        boxes = (
            box_from_points(left_points, hand.shape),
            box_from_points(right_points, hand.shape),
            mask_box(face, padding=6),
        )
        for column, array in enumerate(images):
            x = column * (panel + gap)
            canvas.paste(resize(array, (panel, full_h)), (x, y0))
            for crop_index, box in enumerate(boxes):
                x0, cy0, x1, cy1 = box
                crop = array[cy0:cy1, x0:x1]
                canvas.paste(resize(crop, (panel, crop_h)), (x, y0 + full_h + gap + crop_index * (crop_h + gap)))
        for crop_index, label in enumerate(("LEFT/IMAGE-RIGHT HAND", "RIGHT/IMAGE-LEFT HAND", "FACE")):
            yy = y0 + full_h + gap + crop_index * (crop_h + gap)
            draw.rectangle((0, yy, 185, yy + 21), fill=(255, 255, 255))
            draw.text((5, yy + 2), label, font=font(9, True), fill=(155, 20, 45) if "HAND" in label else (0, 105, 145))
        manifest.append({"index": index, "text_key": anchor["text_key"], "middle": middle, "frame_id": frame_id})
        y += sample_height + 14

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT, quality=95)
    OUTPUT.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
