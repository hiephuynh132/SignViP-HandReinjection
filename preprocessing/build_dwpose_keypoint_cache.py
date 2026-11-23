#!/usr/bin/env python3
"""Build center-cropped DWPose keypoint caches for PHOENIX-2014-T."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


REPO_DIR = Path(__file__).resolve().parent
DWPOSE_DIR = REPO_DIR / "DWPose" / "ControlNet-v1-1-nightly"
if str(DWPOSE_DIR) not in sys.path:
    sys.path.insert(0, str(DWPOSE_DIR))

from annotator.dwpose import DWposeDetector  # noqa: E402
from dwpose_keypoints import (  # noqa: E402
    infer_keypoints_video,
    load_keypoint_cache,
    render_part_masks_video,
    save_keypoint_cache,
)


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache normalized DWPose keypoints instead of rendered RGB pose videos."
    )
    parser.add_argument(
        "--video-root",
        default="data/raw_rwth/videos",
        help="Directory containing the train/dev/test video folders.",
    )
    parser.add_argument(
        "--corpus-root",
        default="data/raw_rwth/manual",
        help="Directory containing PHOENIX-2014-T.<split>.corpus.csv.",
    )
    parser.add_argument(
        "--cache-root",
        default=str(REPO_DIR / "cache_pose_keypoints_center_crop_384"),
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument("--infer-size", type=int, default=384)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--mask-cache-root",
        help="Optionally export binary masks as compressed NPZ files.",
    )
    parser.add_argument(
        "--mask-parts",
        nargs="+",
        choices=("body", "hands", "face", "all"),
        default=("hands", "face"),
        help="Mask arrays to save. Defaults to hands and face only.",
    )
    parser.add_argument("--mask-size", type=int, default=128)
    parser.add_argument("--mask-dilation", type=int, default=0)
    return parser.parse_args()


def center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(height, width)
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def read_video_rgb(path: Path, image_size: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_rgb = center_crop(frame_rgb)
            frame_rgb = cv2.resize(
                frame_rgb,
                (image_size, image_size),
                interpolation=cv2.INTER_AREA,
            )
            frames.append(frame_rgb.astype(np.uint8, copy=False))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"No frames decoded from video: {path}")
    return np.stack(frames, axis=0)


def sample_names(corpus_path: Path) -> list[str]:
    with corpus_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        return [row["name"].strip() for row in reader]


def find_video(split_dir: Path, name: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = split_dir / f"{name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No video found for sample {name!r} under {split_dir}")


def save_masks(
    path: Path,
    cache: dict[str, np.ndarray],
    size: int,
    score_threshold: float,
    dilation_radius: int,
    parts: tuple[str, ...] | list[str],
) -> None:
    masks = render_part_masks_video(
        cache,
        height=size,
        width=size,
        score_threshold=score_threshold,
        dilation_radius=dilation_radius,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(
            temporary,
            **{
                name: masks[name].astype(np.uint8)
                for name in dict.fromkeys(parts)
            },
        )
        temporary.replace(path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def main() -> None:
    args = parse_args()
    video_root = Path(args.video_root)
    corpus_root = Path(args.corpus_root)
    cache_root = Path(args.cache_root)
    mask_root = Path(args.mask_cache_root) if args.mask_cache_root else None
    detector = None

    for split in args.splits:
        corpus = corpus_root / f"PHOENIX-2014-T.{split}.corpus.csv"
        names = sample_names(corpus)
        if args.limit is not None:
            names = names[: max(0, args.limit)]
        split_dir = video_root / split

        for name in tqdm(names, desc=f"DWPose keypoints [{split}]"):
            output = cache_root / split / f"{name}_dwpose{args.infer_size}_keypoints_v1.npz"
            mask_output = (
                mask_root / split / f"{name}_dwpose_mask{args.mask_size}_v1.npz"
                if mask_root is not None
                else None
            )
            cache = None
            if output.exists() and not args.overwrite:
                try:
                    cache = load_keypoint_cache(output)
                except Exception as exc:
                    print(f"WARNING: rebuilding invalid cache {output}: {exc}")

            if cache is None:
                if detector is None:
                    detector = DWposeDetector()
                video_path = find_video(split_dir, name)
                frames = read_video_rgb(video_path, args.infer_size)
                cache = infer_keypoints_video(detector, frames)
                save_keypoint_cache(output, cache)

            if mask_output is not None and (args.overwrite or not mask_output.exists()):
                save_masks(
                    mask_output,
                    cache,
                    size=args.mask_size,
                    score_threshold=args.score_threshold,
                    dilation_radius=args.mask_dilation,
                    parts=args.mask_parts,
                )


if __name__ == "__main__":
    main()
