#!/usr/bin/env python3
"""Build face/hand masks aligned with pose or fused-keypoint caches.

Both pipelines see the same square, center-cropped 384x384 frame. Grounding
DINO and SAM2 operate in that coordinate system, then binary masks are resized
to the requested output resolution. The ``thesis`` layout reads the word-video
split from HDF5 and uses fused COCO-WholeBody-133 keypoints.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import h5py
import numpy as np
import pycocotools.mask as mask_util
import supervision as sv
import torch
from tqdm import tqdm
from torchvision.ops import box_convert, nms

from grounding_dino.groundingdino.util.inference import Model, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

REPO_DIR = Path(__file__).resolve().parent
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg")
MASK_SCHEMA_VERSION = 4
DWPOSE_PARTS = {
    "face": slice(24, 92),
    "left_hand": slice(92, 113),
    "right_hand": slice(113, 134),
}
DWPOSE_HAND_BODY_JOINTS = {
    "left_hand": {"elbow": 6, "wrist": 7},
    "right_hand": {"elbow": 3, "wrist": 4},
}
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache Grounded-SAM2 face/hand masks aligned with DWPose."
    )
    parser.add_argument(
        "--video-root",
        default="data/raw_rwth/videos",
        help="Directory containing train/dev/test video folders.",
    )
    parser.add_argument(
        "--dataset-layout",
        choices=("phoenix", "thesis"),
        default="phoenix",
        help=(
            "phoenix uses split video folders, corpus CSVs, and DWPose caches; "
            "thesis uses a flat video directory, HDF5 split manifests, and fused keypoints."
        ),
    )
    parser.add_argument(
        "--thesis-hdf5-root",
        default=str(
            REPO_DIR.parent
            / "signlanguage_data/processed/signvip_hdf5_256/main"
        ),
        help="Root containing <split>/shard_*.hdf5 for --dataset-layout=thesis.",
    )
    parser.add_argument(
        "--fused-keypoint-root",
        default=str(
            REPO_DIR.parent / "signlanguage_data/processed/fused_keypoints"
        ),
        help="Flat COCO-WholeBody-133 NPZ directory for --dataset-layout=thesis.",
    )
    parser.add_argument(
        "--corpus-root",
        default="data/raw_rwth/manual",
        help="Directory containing PHOENIX-2014-T.<split>.corpus.csv.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "dev", "test"])
    parser.add_argument(
        "--output-root",
        default=str(REPO_DIR.parent / "cache_grounded_sam2_face_hand_center_crop_128"),
    )
    parser.add_argument(
        "--sample",
        action="append",
        help="Process only this sample name. Repeat for multiple samples.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--infer-size", type=int, default=384)
    parser.add_argument("--output-size", type=int, default=128)
    parser.add_argument(
        "--prompt-mode",
        choices=("grounding", "dwpose", "hybrid"),
        default="hybrid",
        help="Use text boxes, DWPose prompts, or DWPose with text fallback.",
    )
    parser.add_argument(
        "--dwpose-keypoint-root",
        default=str(REPO_DIR.parent / "cache_pose_keypoints_center_crop_384"),
    )
    parser.add_argument("--dwpose-score-threshold", type=float, default=0.3)
    parser.add_argument("--dwpose-min-hand-points", type=int, default=4)
    parser.add_argument("--dwpose-min-face-points", type=int, default=8)
    parser.add_argument("--dwpose-hand-padding", type=float, default=0.35)
    parser.add_argument("--dwpose-face-padding", type=float, default=0.18)
    parser.add_argument("--dwpose-max-points", type=int, default=24)
    parser.add_argument("--keypoint-coverage-weight", type=float, default=0.5)
    parser.add_argument("--min-keypoint-coverage", type=float, default=0.5)
    parser.add_argument(
        "--min-hand-sam2-score",
        type=float,
        default=0.4,
        help="Reject DWPose-guided hand masks below this SAM2 score.",
    )
    parser.add_argument(
        "--min-hand-median-keypoint-score",
        type=float,
        default=0.35,
        help="Minimum median confidence among visible hand keypoints.",
    )
    parser.add_argument(
        "--max-hand-wrist-distance",
        type=float,
        default=0.15,
        help="Maximum normalized distance from hand root to matching body wrist.",
    )
    parser.add_argument(
        "--max-hand-center-jump",
        type=float,
        default=0.2,
        help="Maximum normalized hand-center displacement to an adjacent frame.",
    )
    parser.add_argument(
        "--hybrid-hand-fallback",
        choices=("none", "grounding"),
        default="none",
        help="Fallback used when a guided hand is rejected. 'none' is safer.",
    )
    parser.add_argument(
        "--text-prompt",
        default="face. hands.",
    )
    parser.add_argument("--box-threshold", type=float, default=0.35)
    parser.add_argument("--text-threshold", type=float, default=0.25)
    parser.add_argument("--grounding-nms-threshold", type=float, default=0.5)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sam2-checkpoint",
        default=str(REPO_DIR / "checkpoints" / "sam2.1_hiera_large.pt"),
    )
    parser.add_argument(
        "--sam2-model-config",
        default="configs/sam2.1/sam2.1_hiera_l.yaml",
    )
    parser.add_argument(
        "--grounding-dino-config",
        default=str(
            REPO_DIR
            / "grounding_dino"
            / "groundingdino"
            / "config"
            / "GroundingDINO_SwinT_OGC.py"
        ),
    )
    parser.add_argument(
        "--grounding-dino-checkpoint",
        default=str(
            REPO_DIR
            / "gdino_checkpoints"
            / "groundingdino_swint_ogc.pth"
        ),
    )
    parser.add_argument(
        "--save-annotated-video",
        action="store_true",
        help="Save a 384x384 diagnostic video next to each mask cache.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save every annotated 384x384 inference frame as a PNG sequence.",
    )
    parser.add_argument(
        "--save-separate-masks",
        action="store_true",
        help="Also save .hands.npz and .face.npz files beside the combined cache.",
    )
    parser.add_argument("--dump-json", action="store_true")
    parser.add_argument("--use-mediapipe-keypoints", action="store_true")
    parser.add_argument(
        "--hand-landmarker-task",
        default=str(REPO_DIR / "hand_landmarker.task"),
    )
    parser.add_argument("--multimask-output", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.infer_size <= 0 or args.output_size <= 0:
        raise ValueError("--infer-size and --output-size must be positive")
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    if args.dwpose_max_points <= 0:
        raise ValueError("--dwpose-max-points must be positive")
    if not 0.0 <= args.min_keypoint_coverage <= 1.0:
        raise ValueError("--min-keypoint-coverage must be in [0, 1]")
    if not 0.0 <= args.min_hand_sam2_score <= 1.0:
        raise ValueError("--min-hand-sam2-score must be in [0, 1]")
    if not 0.0 <= args.min_hand_median_keypoint_score <= 1.0:
        raise ValueError(
            "--min-hand-median-keypoint-score must be in [0, 1]"
        )
    if args.max_hand_wrist_distance <= 0.0:
        raise ValueError("--max-hand-wrist-distance must be positive")
    if args.max_hand_center_jump <= 0.0:
        raise ValueError("--max-hand-center-jump must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
    for path in (
        args.sam2_checkpoint,
        args.grounding_dino_config,
        args.grounding_dino_checkpoint,
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(path)


def center_crop(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(height, width)
    y0 = (height - size) // 2
    x0 = (width - size) // 2
    return image[y0 : y0 + size, x0 : x0 + size]


def preprocess_frame(frame_bgr: np.ndarray, infer_size: int) -> np.ndarray:
    cropped = center_crop(frame_bgr)
    return cv2.resize(
        cropped,
        (infer_size, infer_size),
        interpolation=cv2.INTER_AREA,
    ).astype(np.uint8, copy=False)


def resize_binary_mask(mask: np.ndarray, output_size: int) -> np.ndarray:
    mask = np.asarray(mask, dtype=np.float32)
    interpolation = (
        cv2.INTER_AREA
        if output_size < min(mask.shape[-2:])
        else cv2.INTER_NEAREST
    )
    resized = cv2.resize(
        mask,
        (output_size, output_size),
        interpolation=interpolation,
    )
    return (resized > 0).astype(np.uint8)


def class_to_part(class_name: str) -> str | None:
    normalized = str(class_name).strip().lower()
    if "face" in normalized or "head" in normalized:
        return "face"
    if "hand" in normalized or "finger" in normalized:
        return "hands"
    return None


def load_dwpose_keypoint_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "keypoints",
            "scores",
            "frame_offsets",
            "inference_size",
            "schema_version",
        }
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing DWPose arrays in {path}: {sorted(missing)}")
        cache = {name: archive[name] for name in required}
    keypoints = cache["keypoints"]
    scores = cache["scores"]
    offsets = cache["frame_offsets"]
    if keypoints.ndim != 3 or keypoints.shape[1:] != (134, 2):
        raise ValueError(f"Invalid DWPose keypoint shape in {path}: {keypoints.shape}")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(f"Invalid DWPose score shape in {path}: {scores.shape}")
    if offsets.ndim != 1 or offsets[0] != 0 or offsets[-1] != len(keypoints):
        raise ValueError(f"Invalid DWPose frame offsets in {path}")
    return cache


def load_fused_keypoint_cache(path: Path) -> dict[str, np.ndarray]:
    """Adapt one-person COCO-WholeBody-133 fusion data to the DWPose prompt API."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"keypoints", "frame_indices", "metadata"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"Missing fused-keypoint arrays in {path}: {sorted(missing)}")
        fused = np.asarray(archive["keypoints"], dtype=np.float32)
        frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata"].item()))
    if fused.ndim != 3 or fused.shape[1:] != (133, 3):
        raise ValueError(f"Invalid fused keypoint shape in {path}: {fused.shape}")
    expected = np.arange(len(fused), dtype=np.int64)
    if not np.array_equal(frame_indices, expected):
        raise ValueError(
            f"Fused frames in {path} must be contiguous 0..N-1; "
            f"got first={frame_indices[:5].tolist()} last={frame_indices[-5:].tolist()}"
        )

    # Fusion coordinates are normalized against the uncropped source frame.
    # Convert them to the centered-square coordinate system seen by SAM2.
    width = int(metadata["frame_width"])
    height = int(metadata["frame_height"])
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid fused frame size in {path}: {width}x{height}")
    side = min(width, height)
    x0 = (width - side) * 0.5
    y0 = (height - side) * 0.5
    fused = fused.copy()
    fused[:, :, 0] = (fused[:, :, 0] * width - x0) / side
    fused[:, :, 1] = (fused[:, :, 1] * height - y0) / side

    # The paper pipeline consumes a 134-point DWPose layout. Only its face,
    # hand, elbow, and wrist slots are used for prompts/filtering, so map those
    # explicitly from COCO-WholeBody-133.
    keypoints = np.zeros((len(fused), 134, 2), dtype=np.float32)
    scores = np.zeros((len(fused), 134), dtype=np.float32)

    def copy_part(target: slice, source: slice) -> None:
        keypoints[:, target] = fused[:, source, :2]
        scores[:, target] = fused[:, source, 2]

    copy_part(slice(24, 92), slice(23, 91))
    copy_part(slice(92, 113), slice(91, 112))
    copy_part(slice(113, 134), slice(112, 133))
    # Expected DWPose filter slots: left elbow/wrist=6/7, right=3/4.
    # COCO-WholeBody slots: left elbow/wrist=7/9, right=8/10.
    for target, source in ((6, 7), (7, 9), (3, 8), (4, 10)):
        keypoints[:, target] = fused[:, source, :2]
        scores[:, target] = fused[:, source, 2]

    return {
        "keypoints": keypoints,
        "scores": scores,
        "frame_offsets": np.arange(len(fused) + 1, dtype=np.int64),
        "inference_size": np.asarray([384, 384], dtype=np.int32),
        "schema_version": np.asarray(1, dtype=np.int32),
    }


def dwpose_frame(
    cache: dict[str, np.ndarray], frame_index: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = cache["frame_offsets"]
    if frame_index < 0 or frame_index >= len(offsets) - 1:
        raise IndexError(
            f"DWPose frame {frame_index} outside cache with {len(offsets) - 1} frames"
        )
    start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
    return cache["keypoints"][start:end], cache["scores"][start:end]


def padded_points_box(
    points: np.ndarray,
    image_size: int,
    padding_ratio: float,
    minimum_size: float,
) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    center = (minimum + maximum) * 0.5
    extent = np.maximum(
        (maximum - minimum) * (1.0 + 2.0 * padding_ratio),
        minimum_size,
    )
    xyxy = np.concatenate((center - extent * 0.5, center + extent * 0.5))
    xyxy[[0, 2]] = np.clip(xyxy[[0, 2]], 0, image_size - 1)
    xyxy[[1, 3]] = np.clip(xyxy[[1, 3]], 0, image_size - 1)
    return xyxy.astype(np.float32)


def keypoint_support_mask(
    points: np.ndarray,
    image_size: int,
    source_part: str,
) -> np.ndarray:
    """Create a local region that prevents SAM2 masks leaking into limbs/neck."""
    rounded = np.rint(points).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, image_size - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, image_size - 1)
    support = np.zeros((image_size, image_size), dtype=np.uint8)
    if len(rounded) >= 3:
        hull = cv2.convexHull(rounded.reshape(-1, 1, 2))
        cv2.fillConvexPoly(support, hull, 1)
    else:
        for point in rounded:
            cv2.circle(support, tuple(point), 2, 1, -1)
    radius = 7 if source_part == "face" else 11
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
    )
    return cv2.dilate(support, kernel, iterations=1).astype(bool)


def visible_part_points(
    keypoints: np.ndarray,
    scores: np.ndarray,
    part_slice: slice,
    score_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    part_scores = scores[part_slice]
    normalized_points = keypoints[part_slice]
    valid = (
        (part_scores > score_threshold)
        & np.isfinite(normalized_points).all(axis=1)
        & (normalized_points[:, 0] >= 0.0)
        & (normalized_points[:, 0] <= 1.0)
        & (normalized_points[:, 1] >= 0.0)
        & (normalized_points[:, 1] <= 1.0)
    )
    return normalized_points[valid], part_scores[valid]


def hand_wrist_distance(
    keypoints: np.ndarray,
    scores: np.ndarray,
    source_part: str,
    score_threshold: float,
) -> float | None:
    joints = DWPOSE_HAND_BODY_JOINTS[source_part]
    part_slice = DWPOSE_PARTS[source_part]
    hand_root_index = int(part_slice.start)
    wrist_index = joints["wrist"]
    if (
        scores[hand_root_index] <= score_threshold
        or scores[wrist_index] <= score_threshold
    ):
        return None
    hand_root = keypoints[hand_root_index]
    wrist = keypoints[wrist_index]
    if not np.isfinite(hand_root).all() or not np.isfinite(wrist).all():
        return None
    return float(np.linalg.norm(hand_root - wrist))


def adjacent_hand_center_distances(
    current_center: np.ndarray,
    person_index: int,
    source_part: str,
    neighbors: tuple[tuple[np.ndarray, np.ndarray] | None, ...],
    args: argparse.Namespace,
) -> list[float]:
    distances: list[float] = []
    part_slice = DWPOSE_PARTS[source_part]
    for neighbor in neighbors:
        if neighbor is None:
            continue
        neighbor_keypoints, neighbor_scores = neighbor
        if person_index >= len(neighbor_keypoints):
            continue
        points, _ = visible_part_points(
            neighbor_keypoints[person_index],
            neighbor_scores[person_index],
            part_slice,
            args.dwpose_score_threshold,
        )
        if len(points) < args.dwpose_min_hand_points:
            continue
        distances.append(float(np.linalg.norm(current_center - points.mean(axis=0))))
    return distances


def dwpose_part_prompts(
    keypoints: np.ndarray,
    scores: np.ndarray,
    args: argparse.Namespace,
    temporal_neighbors: tuple[
        tuple[np.ndarray, np.ndarray] | None, ...
    ] = (),
) -> list[dict[str, object]]:
    prompts: list[dict[str, object]] = []
    for person_index in range(len(keypoints)):
        for source_part, part_slice in DWPOSE_PARTS.items():
            normalized_points, visible_scores = visible_part_points(
                keypoints[person_index],
                scores[person_index],
                part_slice,
                args.dwpose_score_threshold,
            )
            minimum_points = (
                args.dwpose_min_face_points
                if source_part == "face"
                else args.dwpose_min_hand_points
            )
            if len(normalized_points) < minimum_points:
                continue
            is_face = source_part == "face"
            wrist_distance = None
            temporal_jump = None
            if not is_face:
                median_score = float(np.median(visible_scores))
                if median_score < args.min_hand_median_keypoint_score:
                    continue
                wrist_distance = hand_wrist_distance(
                    keypoints[person_index],
                    scores[person_index],
                    source_part,
                    args.dwpose_score_threshold,
                )
                if (
                    wrist_distance is not None
                    and wrist_distance > args.max_hand_wrist_distance
                ):
                    continue
                temporal_distances = adjacent_hand_center_distances(
                    normalized_points.mean(axis=0),
                    person_index,
                    source_part,
                    temporal_neighbors,
                    args,
                )
                if temporal_distances:
                    temporal_jump = min(temporal_distances)
                    if temporal_jump > args.max_hand_center_jump:
                        continue
            points = normalized_points * float(args.infer_size)
            if len(points) > args.dwpose_max_points:
                selected = np.linspace(
                    0, len(points) - 1, args.dwpose_max_points
                ).round().astype(np.int64)
                points = points[selected]
            box = padded_points_box(
                points,
                args.infer_size,
                (
                    args.dwpose_face_padding
                    if is_face
                    else args.dwpose_hand_padding
                ),
                minimum_size=64.0 if is_face else 36.0,
            )
            prompts.append(
                {
                    "source_part": source_part,
                    "part": "face" if is_face else "hands",
                    "person_index": person_index,
                    "points": points.astype(np.float32),
                    "box": box,
                    "median_keypoint_score": float(np.median(visible_scores)),
                    "wrist_distance": wrist_distance,
                    "temporal_jump": temporal_jump,
                }
            )
    return prompts


def merge_part_masks(
    masks: np.ndarray,
    class_names: list[str],
    output_size: int,
) -> dict[str, np.ndarray]:
    if len(masks) != len(class_names):
        raise ValueError("Number of masks does not match number of class names")
    if len(masks):
        infer_height, infer_width = np.asarray(masks[0]).shape[-2:]
    else:
        infer_height = infer_width = output_size
    merged = {
        "hands": np.zeros((infer_height, infer_width), dtype=bool),
        "face": np.zeros((infer_height, infer_width), dtype=bool),
    }
    for mask, class_name in zip(masks, class_names):
        part = class_to_part(class_name)
        if part is not None:
            merged[part] |= np.asarray(mask, dtype=bool)
    return {
        part: resize_binary_mask(mask, output_size)
        for part, mask in merged.items()
    }


def single_mask_to_rle(mask: np.ndarray) -> dict[str, object]:
    rle = mask_util.encode(
        np.array(mask[:, :, None], order="F", dtype="uint8")
    )[0]
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


def sample_names(corpus_path: Path) -> list[str]:
    with corpus_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="|")
        return [row["name"].strip() for row in reader]


def thesis_sample_names(hdf5_root: Path, split: str) -> list[str]:
    names: list[str] = []
    paths = sorted((hdf5_root / split).glob("*.hdf5")) + sorted(
        (hdf5_root / split).glob("*.h5")
    )
    if not paths:
        raise FileNotFoundError(f"No thesis HDF5 shards under {hdf5_root / split}")
    for path in paths:
        with h5py.File(path, "r") as handle:
            for key in sorted(handle["text_key"].keys(), key=lambda value: int(value)):
                value = handle["text_key"][key][()]
                if isinstance(value, bytes):
                    value = value.decode("utf-8")
                names.append(str(value))
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate thesis sample IDs in split {split}")
    return names


def find_video(split_dir: Path, sample_name: str) -> Path:
    for extension in VIDEO_EXTENSIONS:
        candidate = split_dir / f"{sample_name}{extension}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No source video for {sample_name!r} under {split_dir}"
    )


def output_path(
    output_root: Path,
    split: str,
    sample_name: str,
    infer_size: int,
    output_size: int,
) -> Path:
    return (
        output_root
        / split
        / (
            f"{sample_name}_grounded_sam2_face_hands_"
            f"{infer_size}_to{output_size}_v{MASK_SCHEMA_VERSION}.npz"
        )
    )


def valid_existing_cache(
    path: Path,
    infer_size: int,
    output_size: int,
) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(("hands", "face")).difference(archive.files):
                return False
            hands = archive["hands"]
            face = archive["face"]
            return (
                hands.shape == face.shape
                and hands.ndim == 3
                and hands.shape[1:] == (output_size, output_size)
                and tuple(archive["inference_size"].tolist())
                == (infer_size, infer_size)
                and tuple(archive["output_size"].tolist())
                == (output_size, output_size)
                and int(archive["schema_version"].item())
                == MASK_SCHEMA_VERSION
            )
    except Exception:
        return False


def requested_sidecars_exist(
    cache_path: Path,
    args: argparse.Namespace,
) -> bool:
    if args.dump_json and not cache_path.with_suffix(".json").is_file():
        return False
    if (
        args.save_annotated_video
        and not cache_path.with_suffix(".annotated.mp4").is_file()
    ):
        return False
    if args.save_frames:
        frames_dir = cache_path.with_suffix(".frames")
        if not frames_dir.is_dir():
            return False
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                expected_frames = int(archive["hands"].shape[0])
        except Exception:
            return False
        if len(list(frames_dir.glob("*.png"))) != expected_frames:
            return False
    if args.save_separate_masks:
        if not cache_path.with_suffix(".hands.npz").is_file():
            return False
        if not cache_path.with_suffix(".face.npz").is_file():
            return False
    return True


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def init_mediapipe_detector(args: argparse.Namespace):
    if not args.use_mediapipe_keypoints:
        return None
    global mp, mediapipe_python, mediapipe_vision
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mediapipe_python
        from mediapipe.tasks.python import vision as mediapipe_vision
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "mediapipe is not installed but --use-mediapipe-keypoints was set"
        ) from exc
    if not Path(args.hand_landmarker_task).is_file():
        raise FileNotFoundError(args.hand_landmarker_task)
    options = mediapipe_vision.HandLandmarkerOptions(
        base_options=mediapipe_python.BaseOptions(
            model_asset_path=args.hand_landmarker_task
        ),
        running_mode=mediapipe_vision.RunningMode.IMAGE,
        num_hands=2,
    )
    return mediapipe_vision.HandLandmarker.create_from_options(options)


def detect_mediapipe_on_segment(
    detector,
    segmented_roi_bgr: np.ndarray,
    offset_x: int,
    offset_y: int,
) -> dict[str, object] | None:
    if detector is None or segmented_roi_bgr.size == 0:
        return None
    roi_height, roi_width = segmented_roi_bgr.shape[:2]
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(segmented_roi_bgr, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect(mp_image)
    if not result.hand_landmarks:
        return None
    landmarks = [
        {
            "x": float(offset_x + point.x * roi_width),
            "y": float(offset_y + point.y * roi_height),
            "z": float(point.z),
        }
        for point in result.hand_landmarks[0]
    ]
    handedness = result.handedness[0][0] if result.handedness else None
    return {
        "handedness": handedness.category_name if handedness else None,
        "handedness_score": float(handedness.score) if handedness else None,
        "landmarks": landmarks,
    }


def draw_mediapipe_landmarks(
    image: np.ndarray, mediapipe_hand: dict[str, object] | None
) -> None:
    if not mediapipe_hand:
        return
    points = mediapipe_hand.get("landmarks", [])
    for start_index, end_index in HAND_CONNECTIONS:
        if start_index < len(points) and end_index < len(points):
            start, end = points[start_index], points[end_index]
            cv2.line(
                image,
                (int(start["x"]), int(start["y"])),
                (int(end["x"]), int(end["y"])),
                (0, 255, 255),
                2,
            )
    for point in points:
        cv2.circle(
            image,
            (int(point["x"]), int(point["y"])),
            3,
            (0, 255, 0),
            -1,
        )


class FaceHandMaskProcessor:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.sam2_model = build_sam2(
            args.sam2_model_config,
            args.sam2_checkpoint,
            device=args.device,
        )
        self.sam2_predictor = SAM2ImagePredictor(self.sam2_model)
        self.grounding_model = None
        if args.prompt_mode in {"grounding", "hybrid"}:
            self.grounding_model = load_model(
                model_config_path=args.grounding_dino_config,
                model_checkpoint_path=args.grounding_dino_checkpoint,
                device=args.device,
            )
        self.mediapipe_detector = init_mediapipe_detector(args)
        self.box_annotator = sv.BoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.mask_annotator = sv.MaskAnnotator()

        if args.device.startswith("cuda"):
            device_index = torch.device(args.device).index or 0
            if torch.cuda.get_device_properties(device_index).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    def close(self) -> None:
        if self.mediapipe_detector is not None:
            self.mediapipe_detector.close()

    def _dwpose_guided_records(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        temporal_neighbors: tuple[
            tuple[np.ndarray, np.ndarray] | None, ...
        ] = (),
    ) -> tuple[list[dict[str, object]], set[str]]:
        args = self.args
        prompts = dwpose_part_prompts(
            keypoints, scores, args, temporal_neighbors
        )
        records: list[dict[str, object]] = []
        successful_sources: set[str] = set()
        for prompt in prompts:
            points = prompt["points"]
            candidate_masks, candidate_scores, _ = self.sam2_predictor.predict(
                point_coords=points,
                point_labels=np.ones(len(points), dtype=np.int32),
                box=prompt["box"],
                multimask_output=True,
            )
            candidate_masks = np.asarray(candidate_masks, dtype=bool)
            candidate_scores = np.asarray(candidate_scores).reshape(-1)
            source_part = str(prompt["source_part"])
            support = keypoint_support_mask(
                points, args.infer_size, source_part
            )
            candidate_masks &= support[None, ...]
            rounded = np.rint(points).astype(np.int32)
            rounded[:, 0] = np.clip(rounded[:, 0], 0, args.infer_size - 1)
            rounded[:, 1] = np.clip(rounded[:, 1], 0, args.infer_size - 1)
            coverages = np.asarray(
                [
                    mask[rounded[:, 1], rounded[:, 0]].mean()
                    for mask in candidate_masks
                ],
                dtype=np.float32,
            )
            combined_scores = (
                candidate_scores
                + args.keypoint_coverage_weight * coverages
            )
            best = int(np.argmax(combined_scores))
            mask = candidate_masks[best]
            coverage = float(coverages[best])
            sam2_score = float(candidate_scores[best])
            if (
                not mask.any()
                or coverage < args.min_keypoint_coverage
                or (
                    source_part in {"left_hand", "right_hand"}
                    and sam2_score < args.min_hand_sam2_score
                )
            ):
                continue
            successful_sources.add(source_part)
            records.append(
                {
                    "class_name": f"dwpose {source_part}",
                    "part": prompt["part"],
                    "source_part": source_part,
                    "source": "dwpose",
                    "box": prompt["box"],
                    "mask": mask,
                    "sam2_score": sam2_score,
                    "keypoint_coverage": coverage,
                    "points": points,
                    "median_keypoint_score": prompt[
                        "median_keypoint_score"
                    ],
                    "wrist_distance": prompt["wrist_distance"],
                    "temporal_jump": prompt["temporal_jump"],
                }
            )
        return records, successful_sources

    def _grounding_records(
        self,
        infer_frame_bgr: np.ndarray,
    ) -> list[dict[str, object]]:
        args = self.args
        boxes, confidences, class_names = predict(
            model=self.grounding_model,
            image=Model.preprocess_image(infer_frame_bgr),
            caption=args.text_prompt,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            device=args.device,
        )
        if boxes is None or boxes.shape[0] == 0:
            return []
        class_names = [str(name) for name in class_names]
        scale = torch.tensor(
            [args.infer_size] * 4,
            dtype=boxes.dtype,
            device=boxes.device,
        )
        absolute_boxes = boxes * scale
        input_boxes = box_convert(
            boxes=absolute_boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        keep_indices: list[int] = []
        for part, maximum in (("hands", 2), ("face", 1)):
            indices = [
                index
                for index, name in enumerate(class_names)
                if class_to_part(name) == part
            ]
            if not indices:
                continue
            index_tensor = torch.as_tensor(
                indices, dtype=torch.long, device=input_boxes.device
            )
            kept_local = nms(
                input_boxes[index_tensor],
                confidences[index_tensor],
                args.grounding_nms_threshold,
            )[:maximum]
            keep_indices.extend(index_tensor[kept_local].tolist())
        if not keep_indices:
            return []
        keep_indices.sort()
        input_boxes = input_boxes[keep_indices].detach().cpu().numpy()
        kept_names = [class_names[index] for index in keep_indices]
        kept_confidences = confidences[keep_indices].detach().cpu().numpy()

        masks, scores, _ = self.sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_boxes,
            multimask_output=args.multimask_output,
        )
        if args.multimask_output:
            best = np.argmax(scores, axis=1)
            masks = masks[np.arange(masks.shape[0]), best]
            scores = scores[np.arange(scores.shape[0]), best]
        if masks.ndim == 4:
            masks = masks.squeeze(1)
        masks = masks.astype(bool)
        score_list = np.asarray(scores).reshape(-1)
        records = []
        for class_name, confidence, box, mask, score in zip(
            kept_names, kept_confidences, input_boxes, masks, score_list
        ):
            if not mask.any():
                continue
            records.append(
                {
                    "class_name": class_name,
                    "part": class_to_part(class_name),
                    "source_part": class_to_part(class_name),
                    "source": "grounding",
                    "box": box,
                    "mask": mask,
                    "sam2_score": float(score),
                    "grounding_score": float(confidence),
                    "keypoint_coverage": None,
                    "points": None,
                }
            )
        return records

    def _annotate_records(
        self,
        infer_frame_bgr: np.ndarray,
        records: list[dict[str, object]],
    ) -> tuple[np.ndarray, list[dict[str, object]]]:
        args = self.args
        annotated = infer_frame_bgr.copy()
        annotations: list[dict[str, object]] = []
        if not records:
            return annotated, annotations
        input_boxes = np.stack([record["box"] for record in records])
        masks = np.stack([record["mask"] for record in records]).astype(bool)
        detections = sv.Detections(
            xyxy=input_boxes,
            mask=masks,
            class_id=np.arange(len(records)),
        )
        labels = [
            (
                f"{record['source']} {record['source_part']} "
                f"{record['sam2_score']:.2f}"
            )
            for record in records
        ]
        annotated = self.box_annotator.annotate(annotated, detections)
        annotated = self.label_annotator.annotate(
            annotated, detections, labels=labels
        )
        annotated = self.mask_annotator.annotate(annotated, detections)

        for record in records:
            class_name = str(record["class_name"])
            part = str(record["part"])
            box = np.asarray(record["box"])
            mask = np.asarray(record["mask"], dtype=bool)
            x1, y1, x2, y2 = np.rint(box).astype(np.int32).tolist()
            x1 = max(0, min(x1, args.infer_size - 1))
            y1 = max(0, min(y1, args.infer_size - 1))
            x2 = max(0, min(x2, args.infer_size))
            y2 = max(0, min(y2, args.infer_size))
            hand_landmarks = None
            if (
                part == "hands"
                and self.mediapipe_detector is not None
                and x2 > x1
                and y2 > y1
            ):
                roi = infer_frame_bgr[y1:y2, x1:x2].copy()
                roi_mask = mask[y1:y2, x1:x2]
                roi[~roi_mask] = 0
                hand_landmarks = detect_mediapipe_on_segment(
                    self.mediapipe_detector, roi, x1, y1
                )
                draw_mediapipe_landmarks(annotated, hand_landmarks)
            points = record.get("points")
            if points is not None:
                for point in np.asarray(points):
                    cv2.circle(
                        annotated,
                        tuple(np.rint(point).astype(np.int32)),
                        2,
                        (0, 255, 0),
                        -1,
                    )
            annotations.append(
                {
                    "class_name": class_name,
                    "part": part,
                    "source": record["source"],
                    "source_part": record["source_part"],
                    "bbox_inference_xyxy": box.tolist(),
                    "sam2_score": float(record["sam2_score"]),
                    "grounding_score": record.get("grounding_score"),
                    "keypoint_coverage": record.get("keypoint_coverage"),
                    "median_keypoint_score": record.get(
                        "median_keypoint_score"
                    ),
                    "wrist_distance": record.get("wrist_distance"),
                    "temporal_jump": record.get("temporal_jump"),
                    "segmentation_output": single_mask_to_rle(
                        resize_binary_mask(mask, args.output_size)
                    ),
                    "mediapipe_hand": hand_landmarks,
                }
            )
        return annotated, annotations

    def infer_frame(
        self,
        frame_bgr: np.ndarray,
        dwpose_data: tuple[np.ndarray, np.ndarray] | None = None,
        dwpose_neighbors: tuple[
            tuple[np.ndarray, np.ndarray] | None, ...
        ] = (),
    ) -> tuple[dict[str, np.ndarray], np.ndarray, list[dict[str, object]]]:
        args = self.args
        infer_frame_bgr = preprocess_frame(frame_bgr, args.infer_size)
        infer_frame_rgb = cv2.cvtColor(infer_frame_bgr, cv2.COLOR_BGR2RGB)
        self.sam2_predictor.set_image(infer_frame_rgb)

        records: list[dict[str, object]] = []
        successful_sources: set[str] = set()
        if args.prompt_mode in {"dwpose", "hybrid"}:
            if dwpose_data is None:
                raise ValueError(
                    f"--prompt-mode={args.prompt_mode} requires a DWPose cache"
                )
            guided, successful_sources = self._dwpose_guided_records(
                *dwpose_data, temporal_neighbors=dwpose_neighbors
            )
            records.extend(guided)

        if args.prompt_mode == "grounding":
            records.extend(self._grounding_records(infer_frame_bgr))
        elif args.prompt_mode == "hybrid":
            missing_face = "face" not in successful_sources
            missing_hand = not {
                "left_hand",
                "right_hand",
            }.issubset(successful_sources)
            use_hand_fallback = (
                missing_hand
                and args.hybrid_hand_fallback == "grounding"
            )
            if missing_face or use_hand_fallback:
                fallback = self._grounding_records(infer_frame_bgr)
                records.extend(
                    record
                    for record in fallback
                    if (
                        (record["part"] == "face" and missing_face)
                        or (
                            record["part"] == "hands"
                            and use_hand_fallback
                        )
                    )
                )

        merged = {
            "hands": np.zeros(
                (args.infer_size, args.infer_size), dtype=bool
            ),
            "face": np.zeros(
                (args.infer_size, args.infer_size), dtype=bool
            ),
        }
        for record in records:
            part = record["part"]
            if part in merged:
                merged[part] |= np.asarray(record["mask"], dtype=bool)
        part_masks = {
            part: resize_binary_mask(mask, args.output_size)
            for part, mask in merged.items()
        }
        annotated, annotations = self._annotate_records(
            infer_frame_bgr, records
        )
        return part_masks, annotated, annotations

    def process_video(
        self,
        video_path: Path,
        cache_path: Path,
        dwpose_cache: dict[str, np.ndarray] | None = None,
    ) -> None:
        args = self.args
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        writer = None
        frames_dir = None
        if args.save_annotated_video:
            annotated_path = cache_path.with_suffix(".annotated.mp4")
            annotated_path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(annotated_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (args.infer_size, args.infer_size),
            )
        if args.save_frames:
            frames_dir = cache_path.with_suffix(".frames")
            frames_dir.mkdir(parents=True, exist_ok=True)

        hands_frames: list[np.ndarray] = []
        face_frames: list[np.ndarray] = []
        json_frames: list[dict[str, object]] = []
        frame_index = 0
        try:
            while True:
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                if frame_index % args.frame_stride == 0:
                    keypoint_data = (
                        dwpose_frame(dwpose_cache, frame_index)
                        if dwpose_cache is not None
                        else None
                    )
                    keypoint_neighbors = ()
                    if dwpose_cache is not None:
                        num_keypoint_frames = (
                            len(dwpose_cache["frame_offsets"]) - 1
                        )
                        previous = (
                            dwpose_frame(dwpose_cache, frame_index - 1)
                            if frame_index > 0
                            else None
                        )
                        following = (
                            dwpose_frame(dwpose_cache, frame_index + 1)
                            if frame_index + 1 < num_keypoint_frames
                            else None
                        )
                        keypoint_neighbors = (previous, following)
                    masks, annotated, annotations = self.infer_frame(
                        frame_bgr, keypoint_data, keypoint_neighbors
                    )
                else:
                    zero = np.zeros(
                        (args.output_size, args.output_size), dtype=np.uint8
                    )
                    masks = {"hands": zero.copy(), "face": zero.copy()}
                    annotated = preprocess_frame(frame_bgr, args.infer_size)
                    annotations = []
                hands_frames.append(masks["hands"])
                face_frames.append(masks["face"])
                if writer is not None:
                    writer.write(annotated)
                if frames_dir is not None:
                    frame_path = frames_dir / f"{frame_index:05d}.png"
                    if not cv2.imwrite(str(frame_path), annotated):
                        raise OSError(
                            f"Failed to save annotated frame: {frame_path}"
                        )
                if args.dump_json:
                    json_frames.append(
                        {
                            "frame_index": frame_index,
                            "timestamp_sec": frame_index / fps,
                            "annotations": annotations,
                        }
                    )
                frame_index += 1
        finally:
            capture.release()
            if writer is not None:
                writer.release()

        if frame_index == 0:
            raise ValueError(f"No frames decoded from {video_path}")
        if expected_frames > 0 and frame_index != expected_frames:
            print(
                f"WARNING: decoded {frame_index}/{expected_frames} frames "
                f"from {video_path.name}"
            )

        hands_array = np.stack(hands_frames).astype(np.uint8, copy=False)
        face_array = np.stack(face_frames).astype(np.uint8, copy=False)
        frame_indices = np.arange(frame_index, dtype=np.int32)
        inference_size = np.asarray(
            [args.infer_size, args.infer_size], dtype=np.int32
        )
        output_size = np.asarray(
            [args.output_size, args.output_size], dtype=np.int32
        )
        common_metadata = {
            "frame_indices": frame_indices,
            "inference_size": inference_size,
            "output_size": output_size,
            "frame_stride": np.asarray(args.frame_stride, dtype=np.int32),
            "prompt_mode": np.asarray(args.prompt_mode),
            "schema_version": np.asarray(
                MASK_SCHEMA_VERSION, dtype=np.int32
            ),
        }
        atomic_save_npz(
            cache_path,
            hands=hands_array,
            face=face_array,
            **common_metadata,
        )
        if args.save_separate_masks:
            atomic_save_npz(
                cache_path.with_suffix(".hands.npz"),
                hands=hands_array,
                **common_metadata,
            )
            atomic_save_npz(
                cache_path.with_suffix(".face.npz"),
                face=face_array,
                **common_metadata,
            )

        if args.dump_json:
            json_path = cache_path.with_suffix(".json")
            payload = {
                "video_path": str(video_path),
                "text_prompt": args.text_prompt,
                "preprocessing": "center_crop_then_resize",
                "prompt_mode": args.prompt_mode,
                "schema_version": MASK_SCHEMA_VERSION,
                "hand_filter": {
                    "min_sam2_score": args.min_hand_sam2_score,
                    "min_median_keypoint_score":
                        args.min_hand_median_keypoint_score,
                    "max_wrist_distance": args.max_hand_wrist_distance,
                    "max_center_jump": args.max_hand_center_jump,
                    "fallback": args.hybrid_hand_fallback,
                },
                "inference_size": [args.infer_size, args.infer_size],
                "output_size": [args.output_size, args.output_size],
                "fps": fps,
                "num_frames": frame_index,
                "frames": json_frames,
            }
            with json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)


def selected_samples(args: argparse.Namespace, split: str) -> list[str]:
    if args.sample:
        names = list(dict.fromkeys(args.sample))
    elif args.dataset_layout == "thesis":
        names = thesis_sample_names(Path(args.thesis_hdf5_root), split)
    else:
        corpus_path = (
            Path(args.corpus_root)
            / f"PHOENIX-2014-T.{split}.corpus.csv"
        )
        names = sample_names(corpus_path)
    if args.limit is not None:
        names = names[: max(0, args.limit)]
    return names


def main() -> None:
    args = parse_args()
    validate_args(args)
    os.chdir(REPO_DIR)
    output_root = Path(args.output_root)
    video_root = Path(args.video_root)
    dwpose_root = Path(args.dwpose_keypoint_root)
    fused_root = Path(args.fused_keypoint_root)

    processor = FaceHandMaskProcessor(args)
    try:
        for split in args.splits:
            names = selected_samples(args, split)
            for name in tqdm(names, desc=f"Grounded-SAM2 [{split}]"):
                cache_path = output_path(
                    output_root,
                    split,
                    name,
                    args.infer_size,
                    args.output_size,
                )
                if (
                    not args.overwrite
                    and valid_existing_cache(
                        cache_path,
                        args.infer_size,
                        args.output_size,
                    )
                    and requested_sidecars_exist(cache_path, args)
                ):
                    continue
                video_dir = video_root if args.dataset_layout == "thesis" else video_root / split
                video_path = find_video(video_dir, name)
                keypoint_cache = None
                if args.prompt_mode in {"dwpose", "hybrid"}:
                    if args.dataset_layout == "thesis":
                        keypoint_path = fused_root / f"{name}.npz"
                    else:
                        keypoint_path = (
                            dwpose_root
                            / split
                            / f"{name}_dwpose{args.infer_size}_keypoints_v1.npz"
                        )
                    if not keypoint_path.is_file():
                        raise FileNotFoundError(
                            f"DWPose cache required by {args.prompt_mode}: "
                            f"{keypoint_path}"
                        )
                    keypoint_cache = (
                        load_fused_keypoint_cache(keypoint_path)
                        if args.dataset_layout == "thesis"
                        else load_dwpose_keypoint_cache(keypoint_path)
                    )
                    if args.dataset_layout == "thesis":
                        keypoint_cache["inference_size"] = np.asarray(
                            [args.infer_size, args.infer_size], dtype=np.int32
                        )
                    inference_size = tuple(
                        keypoint_cache["inference_size"].tolist()
                    )
                    if inference_size != (
                        args.infer_size,
                        args.infer_size,
                    ):
                        raise ValueError(
                            f"DWPose inference size {inference_size} does not "
                            f"match --infer-size={args.infer_size}"
                        )
                processor.process_video(
                    video_path, cache_path, keypoint_cache
                )
    finally:
        processor.close()


if __name__ == "__main__":
    inference_context = torch.inference_mode()
    with inference_context:
        main()
