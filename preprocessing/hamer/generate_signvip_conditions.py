#!/usr/bin/env python3
"""Generate SignVIP HaMeR condition frames for SignGen HDF5 shards.

Each output shard is a compact sidecar HDF5 file containing ``hamer_video``
with camera-aligned hand meshes on a black background. Sample IDs, frame IDs,
spatial resolution, ``len``, and ``text_key`` remain aligned with the source.
The source shards are opened read-only and are never modified.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

from hamer.configs import CACHE_DIR_HAMER
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, download_models, load_hamer
from hamer.utils import recursive_to
from hamer.utils.download import cache_url
from hamer.utils.renderer import Renderer, cam_crop_to_full
from vitpose_model import ViTPoseModel


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/rwth"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/hamer"),
    )
    parser.add_argument("--splits", nargs="+", default=("train", "dev", "test"))
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--rescale-factor", type=float, default=2.0)
    parser.add_argument("--person-score", type=float, default=0.5)
    parser.add_argument("--hand-score", type=float, default=0.5)
    parser.add_argument("--min-hand-keypoints", type=int, default=4)
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--limit-samples", type=int, help="Global sample limit for a smoke run.")
    parser.add_argument("--max-frames", type=int, help="Frame limit per sample for a smoke run.")
    parser.add_argument(
        "--save-preview",
        action="store_true",
        help="Save side-by-side source/HaMeR PNGs under the output root.",
    )
    parser.add_argument("--preview-samples", type=int, default=1)
    parser.add_argument("--preview-frames", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def build_detector():
    from detectron2.config import LazyConfig

    import hamer
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg_path = (
        Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    )
    cfg = LazyConfig.load(str(cfg_path))
    detector_url = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    detector_checkpoint = os.path.join(
        CACHE_DIR_HAMER, "detectron2", "model_final_f05665.pkl"
    )
    cfg.train.init_checkpoint = cache_url(detector_url, detector_checkpoint)
    for predictor in cfg.model.roi_heads.box_predictors:
        predictor.test_score_thresh = 0.25
    return DefaultPredictor_Lazy(cfg)


def numeric_keys(group: h5py.Group) -> list[str]:
    return sorted((key for key in group.keys() if key.isdigit()), key=int)


def frame_count(node: h5py.Group | h5py.Dataset) -> int:
    return int(node.shape[0]) if isinstance(node, h5py.Dataset) else len(numeric_keys(node))


def read_frame(node: h5py.Group | h5py.Dataset, index: int) -> np.ndarray:
    frame = (
        node[index]
        if isinstance(node, h5py.Dataset)
        else node[numeric_keys(node)[index]][()]
    )
    frame = np.asarray(frame)
    if frame.ndim != 3:
        raise ValueError(f"Expected an HWC/CHW RGB frame, got {frame.shape}")
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if frame.shape[-1] < 3:
        raise ValueError(f"Expected at least three image channels, got {frame.shape}")
    frame = frame[..., :3]
    if np.issubdtype(frame.dtype, np.floating):
        if frame.size and float(np.nanmax(frame)) <= 1.0:
            frame = frame * 255.0
        frame = np.nan_to_num(frame, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(frame, 0, 255).astype(np.uint8)


def hand_boxes_from_pose(
    pose_results: Iterable[dict], hand_score: float, min_keypoints: int
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    boxes: list[list[float]] = []
    handedness: list[int] = []
    for pose in pose_results:
        keypoints = pose["keypoints"]
        for hand_keypoints, is_right in (
            (keypoints[-42:-21], 0),
            (keypoints[-21:], 1),
        ):
            valid = hand_keypoints[:, 2] > hand_score
            if int(valid.sum()) < min_keypoints:
                continue
            xy = hand_keypoints[valid, :2]
            boxes.append(
                [xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()]
            )
            handedness.append(is_right)
    if not boxes:
        return None, None
    return np.asarray(boxes, np.float32), np.asarray(handedness, np.float32)


@torch.inference_mode()
def render_condition_frame(
    frame_rgb: np.ndarray,
    detector,
    pose_model: ViTPoseModel,
    hamer_model,
    model_cfg,
    renderer: Renderer,
    device: torch.device,
    args: argparse.Namespace,
) -> np.ndarray:
    height, width = frame_rgb.shape[:2]
    black = np.zeros((height, width, 3), dtype=np.uint8)
    frame_bgr = frame_rgb[:, :, ::-1].copy()

    detections = detector(frame_bgr)["instances"]
    valid_people = (detections.pred_classes == 0) & (
        detections.scores > args.person_score
    )
    person_boxes = detections.pred_boxes.tensor[valid_people].cpu().numpy()
    person_scores = detections.scores[valid_people].cpu().numpy()
    if len(person_boxes) == 0:
        return black

    detection_results = [
        np.concatenate((person_boxes, person_scores[:, None]), axis=1)
    ]
    poses = pose_model.predict_pose(frame_rgb, detection_results)
    hand_boxes, is_right = hand_boxes_from_pose(
        poses, args.hand_score, args.min_hand_keypoints
    )
    if hand_boxes is None:
        return black

    dataset = ViTDetDataset(
        model_cfg,
        frame_bgr,
        hand_boxes,
        is_right,
        rescale_factor=args.rescale_factor,
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    vertices: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    handedness_values: list[np.ndarray] = []
    focal_length = None
    for batch in loader:
        batch = recursive_to(batch, device)
        prediction = hamer_model(batch)
        multiplier = 2 * batch["right"] - 1
        pred_cam = prediction["pred_cam"].clone()
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        image_size = batch["img_size"].float()
        focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * image_size.max()
        )
        batch_translations = cam_crop_to_full(
            pred_cam,
            batch["box_center"].float(),
            batch["box_size"].float(),
            image_size,
            focal_length,
        ).detach().cpu().numpy()

        for index in range(batch["img"].shape[0]):
            handedness = batch["right"][index].detach().cpu().numpy()
            verts = prediction["pred_vertices"][index].detach().cpu().numpy()
            verts[:, 0] = (2 * handedness - 1) * verts[:, 0]
            vertices.append(verts)
            translations.append(batch_translations[index])
            handedness_values.append(handedness)

    if not vertices or focal_length is None:
        return black
    rgba = renderer.render_rgba_multiple(
        vertices,
        cam_t=translations,
        render_res=[width, height],
        is_right=handedness_values,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(0, 0, 0),
        focal_length=float(focal_length.detach().cpu().item()),
    )
    return np.clip(rgba[:, :, :3] * 255.0, 0, 255).astype(np.uint8)


def discover_shards(input_root: Path, splits: list[str]) -> list[tuple[Path, Path]]:
    if input_root.is_file() and input_root.suffix.lower() in (".h5", ".hdf5"):
        return [(input_root, Path(input_root.name))]
    shards: list[tuple[Path, Path]] = []
    for split in splits:
        split_root = input_root / split
        for source in sorted((*split_root.glob("*.hdf5"), *split_root.glob("*.h5"))):
            shards.append((source, source.relative_to(input_root)))
    return shards


def copy_scalar(source: h5py.Group | None, destination: h5py.Group, key: str) -> None:
    if source is not None and key in source and key not in destination:
        destination.create_dataset(key, data=source[key][()])


def save_preview(
    source_rgb: np.ndarray,
    hamer_rgb: np.ndarray,
    destination: Path,
) -> None:
    source_bgr = source_rgb[:, :, ::-1].copy()
    hamer_bgr = hamer_rgb[:, :, ::-1].copy()
    panel = np.concatenate((source_bgr, hamer_bgr), axis=1)
    font_scale = max(min(source_rgb.shape[:2]) / 320.0, 0.35)
    thickness = max(int(round(font_scale * 2)), 1)
    cv2.putText(
        panel,
        "SOURCE RGB",
        (5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "HAMER CONDITION",
        (source_rgb.shape[1] + 5, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(destination), panel):
        raise RuntimeError(f"Could not write preview image: {destination}")


def process_shard(
    source_path: Path,
    destination_path: Path,
    render_frame,
    args: argparse.Namespace,
    remaining_samples: int | None,
) -> tuple[int, int]:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    completed = failed = 0
    compression = None if args.compression == "none" else args.compression

    with h5py.File(source_path, "r") as source, h5py.File(destination_path, "a") as destination:
        if "video" not in source:
            raise KeyError(f"{source_path} has no 'video' group")
        destination.attrs["source_hdf5"] = str(source_path)
        destination.attrs["format"] = "SignVIP HaMeR front-view mesh on black background"
        hamer_group = destination.require_group("hamer_video")
        length_group = destination.require_group("len")
        text_key_group = destination.require_group("text_key")

        sample_keys = numeric_keys(source["video"])
        if remaining_samples is not None:
            sample_keys = sample_keys[:remaining_samples]
        for sample_position, sample_key in enumerate(
            tqdm(sample_keys, desc=source_path.name, leave=False)
        ):
            video_node = source["video"][sample_key]
            source_length = frame_count(video_node)
            if "len" in source and sample_key in source["len"]:
                source_length = min(source_length, int(source["len"][sample_key][()]))
            output_length = (
                min(source_length, args.max_frames)
                if args.max_frames is not None
                else source_length
            )

            if sample_key in hamer_group and not args.overwrite:
                existing = hamer_group[sample_key]
                if existing.shape[0] == output_length:
                    if args.save_preview and sample_position < args.preview_samples:
                        for frame_index in range(
                            min(output_length, args.preview_frames)
                        ):
                            preview_path = (
                                destination_path.parent
                                / "_preview"
                                / destination_path.stem
                                / f"sample_{sample_key}_frame_{frame_index:04d}.png"
                            )
                            save_preview(
                                read_frame(video_node, frame_index),
                                np.asarray(existing[frame_index], dtype=np.uint8),
                                preview_path,
                            )
                    completed += 1
                    continue
                raise RuntimeError(
                    f"Existing {destination_path}:hamer_video/{sample_key} has "
                    f"{existing.shape[0]} frames, expected {output_length}; use --overwrite"
                )

            temporary_key = f"__tmp__{sample_key}"
            if temporary_key in hamer_group:
                del hamer_group[temporary_key]
            try:
                rendered_frames = []
                for frame_index in tqdm(
                    range(output_length),
                    desc=f"sample {sample_key}",
                    leave=False,
                ):
                    source_frame = read_frame(video_node, frame_index)
                    rendered_frame = render_frame(source_frame)
                    rendered_frames.append(rendered_frame)
                    if (
                        args.save_preview
                        and sample_position < args.preview_samples
                        and frame_index < args.preview_frames
                    ):
                        preview_path = (
                            destination_path.parent
                            / "_preview"
                            / destination_path.stem
                            / f"sample_{sample_key}_frame_{frame_index:04d}.png"
                        )
                        save_preview(source_frame, rendered_frame, preview_path)
                frames = np.stack(rendered_frames, axis=0)
                hamer_group.create_dataset(
                    temporary_key,
                    data=frames,
                    dtype=np.uint8,
                    compression=compression,
                    chunks=(1, *frames.shape[1:]),
                )
                if sample_key in hamer_group:
                    del hamer_group[sample_key]
                hamer_group.move(temporary_key, sample_key)
                if sample_key in length_group:
                    del length_group[sample_key]
                length_group.create_dataset(sample_key, data=output_length)
                copy_scalar(source.get("text_key"), text_key_group, sample_key)
                destination.flush()
                completed += 1
            except Exception as error:
                if temporary_key in hamer_group:
                    del hamer_group[temporary_key]
                failed += 1
                tqdm.write(f"ERROR {source_path}:{sample_key}: {error}")
    return completed, failed


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input dataset does not exist: {input_root}")
    if output_root == input_root or (input_root.is_dir() and input_root in output_root.parents):
        raise ValueError("Output root must be outside the input dataset tree")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be positive")
    if args.preview_samples <= 0 or args.preview_frames <= 0:
        raise ValueError("--preview-samples and --preview-frames must be positive")

    shards = discover_shards(input_root, list(args.splits))
    if not shards:
        raise FileNotFoundError(f"No HDF5 shards found under {input_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this preprocessing job")
    device = torch.device("cuda")

    download_models(CACHE_DIR_HAMER)
    hamer_model, model_cfg = load_hamer(args.checkpoint)
    hamer_model = hamer_model.to(device).eval()
    detector = build_detector()
    pose_model = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=hamer_model.mano.faces)

    def render_frame(frame: np.ndarray) -> np.ndarray:
        return render_condition_frame(
            frame, detector, pose_model, hamer_model, model_cfg, renderer, device, args
        )

    total_completed = total_failed = 0
    for source_path, relative_path in tqdm(shards, desc="HDF5 shards"):
        remaining = (
            max(args.limit_samples - total_completed - total_failed, 0)
            if args.limit_samples is not None
            else None
        )
        if remaining == 0:
            break
        completed, failed = process_shard(
            source_path,
            output_root / relative_path,
            render_frame,
            args,
            remaining,
        )
        total_completed += completed
        total_failed += failed

    print(f"Output root: {output_root}")
    print(f"Completed samples: {total_completed}")
    print(f"Failed samples: {total_failed}")
    if total_failed:
        raise RuntimeError(f"Failed to process {total_failed} samples")


if __name__ == "__main__":
    main()
