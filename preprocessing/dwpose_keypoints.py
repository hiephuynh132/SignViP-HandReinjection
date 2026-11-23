"""Numeric, pickle-free DWPose keypoint caches and deterministic rendering.

The cache stores normalized ``(x, y)`` coordinates and confidence scores for
every detected person.  ``frame_offsets`` makes the format support a variable
number of people per frame without object arrays or pickle.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from annotator.dwpose import draw_pose
from annotator.dwpose.util import draw_bodypose, draw_facepose, draw_handpose


SCHEMA_VERSION = 1
BODY_SLICE = slice(0, 18)
FOOT_SLICE = slice(18, 24)
FACE_SLICE = slice(24, 92)
LEFT_HAND_SLICE = slice(92, 113)
RIGHT_HAND_SLICE = slice(113, 134)
EXPECTED_KEYPOINTS = 134
PART_SLICES = {
    "body": BODY_SLICE,
    "feet": FOOT_SLICE,
    "face": FACE_SLICE,
    "left_hand": LEFT_HAND_SLICE,
    "right_hand": RIGHT_HAND_SLICE,
}


def infer_keypoints_frame(detector, image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run DWPose and return normalized xy coordinates plus confidence."""
    height, width = image_bgr.shape[:2]
    keypoints, scores = detector.pose_estimation(image_bgr)
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    if keypoints.ndim != 3 or keypoints.shape[-1] != 2:
        raise ValueError(f"Unexpected DWPose keypoint shape: {keypoints.shape}")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(
            f"DWPose score shape {scores.shape} does not match {keypoints.shape[:2]}"
        )
    if keypoints.shape[1] != EXPECTED_KEYPOINTS:
        raise ValueError(
            f"Expected {EXPECTED_KEYPOINTS} DWPose keypoints, got {keypoints.shape[1]}"
        )

    keypoints = keypoints.copy()
    keypoints[..., 0] /= float(width)
    keypoints[..., 1] /= float(height)
    return keypoints, scores


def infer_keypoints_video(detector, frames_rgb: Iterable[np.ndarray]) -> dict[str, np.ndarray]:
    """Infer and pack a video, preserving a variable person count per frame."""
    keypoint_frames: list[np.ndarray] = []
    score_frames: list[np.ndarray] = []
    offsets = [0]
    inference_size = None

    for frame in frames_rgb:
        frame = np.asarray(frame, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected an RGB HWC frame, got {frame.shape}")
        if inference_size is None:
            inference_size = frame.shape[:2]
        elif tuple(frame.shape[:2]) != tuple(inference_size):
            raise ValueError("All frames in a keypoint cache must have the same size")

        image_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        keypoints, scores = infer_keypoints_frame(detector, image_bgr)
        keypoint_frames.append(keypoints)
        score_frames.append(scores)
        offsets.append(offsets[-1] + len(keypoints))

    if inference_size is None:
        raise ValueError("Cannot create a DWPose cache for an empty video")

    if offsets[-1]:
        keypoints = np.concatenate(keypoint_frames, axis=0).astype(np.float32, copy=False)
        scores = np.concatenate(score_frames, axis=0).astype(np.float32, copy=False)
    else:
        keypoints = np.empty((0, EXPECTED_KEYPOINTS, 2), dtype=np.float32)
        scores = np.empty((0, EXPECTED_KEYPOINTS), dtype=np.float32)

    return {
        "keypoints": keypoints,
        "scores": scores,
        "frame_offsets": np.asarray(offsets, dtype=np.int64),
        "inference_size": np.asarray(inference_size, dtype=np.int32),
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
    }


def validate_keypoint_cache(cache: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {
        "keypoints",
        "scores",
        "frame_offsets",
        "inference_size",
        "schema_version",
    }
    missing = required.difference(cache)
    if missing:
        raise ValueError(f"Missing DWPose cache arrays: {sorted(missing)}")

    keypoints = np.asarray(cache["keypoints"])
    scores = np.asarray(cache["scores"])
    offsets = np.asarray(cache["frame_offsets"])
    inference_size = np.asarray(cache["inference_size"])
    version = int(np.asarray(cache["schema_version"]).item())

    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported DWPose cache schema version: {version}")
    if keypoints.ndim != 3 or keypoints.shape[1:] != (EXPECTED_KEYPOINTS, 2):
        raise ValueError(f"Invalid keypoints shape: {keypoints.shape}")
    if scores.shape != keypoints.shape[:2]:
        raise ValueError(f"Invalid scores shape: {scores.shape}")
    if offsets.ndim != 1 or len(offsets) < 1 or offsets[0] != 0:
        raise ValueError(f"Invalid frame_offsets: shape={offsets.shape}")
    if np.any(offsets[1:] < offsets[:-1]) or offsets[-1] != len(keypoints):
        raise ValueError("frame_offsets are not monotonic or do not cover all people")
    if inference_size.shape != (2,) or np.any(inference_size <= 0):
        raise ValueError(f"Invalid inference_size: {inference_size}")

    return {
        "keypoints": keypoints.astype(np.float32, copy=False),
        "scores": scores.astype(np.float32, copy=False),
        "frame_offsets": offsets.astype(np.int64, copy=False),
        "inference_size": inference_size.astype(np.int32, copy=False),
        "schema_version": np.asarray(version, dtype=np.int32),
    }


def save_keypoint_cache(path: str | os.PathLike, cache: dict[str, np.ndarray]) -> None:
    """Validate and atomically save a compressed NPZ without pickle."""
    cache = validate_keypoint_cache(cache)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    try:
        np.savez_compressed(temporary, **cache)
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def load_keypoint_cache(path: str | os.PathLike) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        cache = {name: archive[name] for name in archive.files}
    return validate_keypoint_cache(cache)


def frame_keypoints(
    cache: dict[str, np.ndarray], frame_index: int
) -> tuple[np.ndarray, np.ndarray]:
    offsets = cache["frame_offsets"]
    num_frames = len(offsets) - 1
    if frame_index < 0:
        frame_index += num_frames
    if frame_index < 0 or frame_index >= num_frames:
        raise IndexError(f"Frame {frame_index} is outside a {num_frames}-frame cache")
    start, end = int(offsets[frame_index]), int(offsets[frame_index + 1])
    return cache["keypoints"][start:end], cache["scores"][start:end]


def _pose_dict(
    keypoints: np.ndarray, scores: np.ndarray, score_threshold: float
) -> dict[str, object]:
    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.shape[:2] != scores.shape or keypoints.shape[-1] != 2:
        raise ValueError("Keypoint and score shapes do not match")

    visible = scores > float(score_threshold)
    visible_keypoints = keypoints.copy()
    visible_keypoints[~visible] = -1.0

    people = len(keypoints)
    body = keypoints[:, BODY_SLICE, :].reshape(people * 18, 2).copy()
    subset = np.full((people, 18), -1, dtype=np.int32)
    for person_index in range(people):
        person_visible = visible[person_index, BODY_SLICE]
        indices = np.arange(18, dtype=np.int32) + person_index * 18
        subset[person_index, person_visible] = indices[person_visible]

    hands = np.concatenate(
        (
            visible_keypoints[:, LEFT_HAND_SLICE, :],
            visible_keypoints[:, RIGHT_HAND_SLICE, :],
        ),
        axis=0,
    )
    return {
        "bodies": {"candidate": body, "subset": subset},
        "hands": hands,
        "faces": visible_keypoints[:, FACE_SLICE, :],
    }


def render_pose_frame(
    keypoints: np.ndarray,
    scores: np.ndarray,
    height: int = 128,
    width: int = 128,
    score_threshold: float = 0.3,
) -> np.ndarray:
    """Render the same RGB pose convention as the bundled DWPose detector."""
    pose = _pose_dict(keypoints, scores, score_threshold)
    return draw_pose(pose, int(height), int(width)).astype(np.uint8, copy=False)


def render_pose_video(
    cache: dict[str, np.ndarray],
    height: int = 128,
    width: int = 128,
    score_threshold: float = 0.3,
) -> np.ndarray:
    cache = validate_keypoint_cache(cache)
    frames = [
        render_pose_frame(
            *frame_keypoints(cache, index),
            height=height,
            width=width,
            score_threshold=score_threshold,
        )
        for index in range(len(cache["frame_offsets"]) - 1)
    ]
    return np.stack(frames, axis=0).astype(np.uint8, copy=False)


def render_part_masks_frame(
    keypoints: np.ndarray,
    scores: np.ndarray,
    height: int = 128,
    width: int = 128,
    score_threshold: float = 0.3,
    dilation_radius: int = 0,
) -> dict[str, np.ndarray]:
    """Return binary stroke masks matching the bundled DWPose renderer.

    The bundled renderer does not draw its six foot keypoints, so ``all`` is
    the union of body, hands, and face.  Foot coordinates remain available in
    the numeric cache through ``PART_SLICES["feet"]``.
    """
    pose = _pose_dict(keypoints, scores, score_threshold)
    shape = (int(height), int(width), 3)

    body_rgb = draw_bodypose(
        np.zeros(shape, dtype=np.uint8),
        pose["bodies"]["candidate"],
        pose["bodies"]["subset"],
    )
    hands_rgb = draw_handpose(np.zeros(shape, dtype=np.uint8), pose["hands"])
    face_rgb = draw_facepose(np.zeros(shape, dtype=np.uint8), pose["faces"])
    masks = {
        "body": np.any(body_rgb > 0, axis=-1),
        "hands": np.any(hands_rgb > 0, axis=-1),
        "face": np.any(face_rgb > 0, axis=-1),
    }
    masks["all"] = masks["body"] | masks["hands"] | masks["face"]

    radius = max(0, int(dilation_radius))
    if radius:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)
        )
        masks = {
            name: cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)
            for name, mask in masks.items()
        }
    return masks


def render_part_masks_video(
    cache: dict[str, np.ndarray],
    height: int = 128,
    width: int = 128,
    score_threshold: float = 0.3,
    dilation_radius: int = 0,
) -> dict[str, np.ndarray]:
    cache = validate_keypoint_cache(cache)
    per_frame = [
        render_part_masks_frame(
            *frame_keypoints(cache, index),
            height=height,
            width=width,
            score_threshold=score_threshold,
            dilation_radius=dilation_radius,
        )
        for index in range(len(cache["frame_offsets"]) - 1)
    ]
    return {
        name: np.stack([frame[name] for frame in per_frame], axis=0)
        for name in ("body", "hands", "face", "all")
    }
