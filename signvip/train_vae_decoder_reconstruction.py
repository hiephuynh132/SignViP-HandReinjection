#!/usr/bin/env python3
"""Train only the SD1.5 VAE decoder on region-aware RGB reconstruction."""

import argparse
from collections import OrderedDict
import datetime
import json
import math
import os
import random
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from diffusers.utils import import_utils as diffusers_import_utils
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.checkpoint import checkpoint
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

diffusers_import_utils._xformers_available = False
diffusers_import_utils._xformers_version = None
from diffusers import AutoencoderKL
from perceptual_losses import (
    clip_loss,
    crop_resize_region_pairs,
    dino_loss,
    load_clip_model,
    load_dino_model,
)


MASK_SUFFIX_TEMPLATE = "_grounded_sam2_face_hands_384_to{output_size}_v4.npz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Write a resumed run to this directory instead of the checkpoint parent.",
    )
    return parser.parse_args()


def decode_text(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return decode_text(value.item())
    return str(value)


def numeric_keys(group):
    return sorted(group.keys(), key=lambda value: int(value))


def frame_from_node(node, frame_index):
    if isinstance(node, h5py.Dataset):
        return np.asarray(node[frame_index], dtype=np.uint8)
    return np.asarray(node[str(frame_index)][()], dtype=np.uint8)


def dilate_mask_in_latent(mask, radius, latent_size, threshold):
    """Expand a stored image-space region mask by ``radius`` latent cells."""
    radius = int(radius)
    if radius <= 0:
        return mask
    original = mask.float().clamp(0, 1)
    latent = F.interpolate(
        original[None, None],
        (int(latent_size), int(latent_size)),
        mode="area",
    )
    latent = (latent > float(threshold)).float()
    latent = F.max_pool2d(
        latent,
        kernel_size=2 * radius + 1,
        stride=1,
        padding=radius,
    )
    expanded = F.interpolate(
        latent,
        original.shape[-2:],
        mode="nearest",
    )[0, 0]
    return torch.maximum(original, expanded)


class ReconstructionFrames(Dataset):
    def __init__(
        self,
        hdf5_root,
        mask_root,
        split,
        image_size,
        mask_suffix=None,
        hand_mask_latent_radius=0,
        hand_mask_latent_size=None,
        hand_mask_threshold=0.05,
        face_mask_latent_radius=0,
        face_mask_latent_size=None,
        face_mask_threshold=0.05,
    ):
        self.image_size = int(image_size)
        self.mask_suffix = str(
            mask_suffix
            or MASK_SUFFIX_TEMPLATE.format(output_size=self.image_size)
        )
        self.hand_mask_latent_radius = int(hand_mask_latent_radius)
        self.hand_mask_latent_size = int(
            hand_mask_latent_size or max(1, self.image_size // 8)
        )
        self.hand_mask_threshold = float(hand_mask_threshold)
        self.face_mask_latent_radius = int(face_mask_latent_radius)
        self.face_mask_latent_size = int(
            face_mask_latent_size or max(1, self.image_size // 8)
        )
        self.face_mask_threshold = float(face_mask_threshold)
        self.mask_dir = Path(mask_root) / split
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask split is missing: {self.mask_dir}")
        paths = sorted(Path(hdf5_root).glob("*.hdf5")) + sorted(Path(hdf5_root).glob("*.h5"))
        if not paths:
            raise FileNotFoundError(f"No HDF5 shards under {hdf5_root}")
        self.items = []
        for path in paths:
            with h5py.File(path, "r") as handle:
                for key in numeric_keys(handle["video"]):
                    text_key = decode_text(handle["text_key"][key][()])
                    mask_path = self.mask_dir / f"{text_key}{self.mask_suffix}"
                    if not mask_path.is_file():
                        raise FileNotFoundError(
                            f"Missing Grounded-SAM2 hand/face mask: {mask_path}"
                        )
                    length = int(handle["len"][key][()])
                    for frame_index in range(length):
                        self.items.append((str(path), str(key), text_key, frame_index, str(mask_path)))
        self._handles = {}
        self._pid = None
        self._mask_arrays = OrderedDict()

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_pid"] = None
        state["_mask_arrays"] = OrderedDict()
        return state

    def _handle(self, path):
        pid = os.getpid()
        if self._pid != pid:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
            self._pid = pid
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]

    def _detail_masks(self, mask_path, frame_index):
        cached = self._mask_arrays.pop(mask_path, None)
        if cached is None:
            with np.load(mask_path, allow_pickle=False) as data:
                frame_indices = np.asarray(data["frame_indices"]).astype(np.int64)
                hands = np.asarray(data["hands"]).copy()
                faces = np.asarray(data["face"]).copy()
            cached = (
                {int(frame): i for i, frame in enumerate(frame_indices)},
                hands,
                faces,
            )
        self._mask_arrays[mask_path] = cached
        while len(self._mask_arrays) > 16:
            self._mask_arrays.popitem(last=False)
        positions, hands, faces = cached
        if frame_index not in positions:
            raise KeyError(f"Mask {mask_path} has no frame {frame_index}")
        position = positions[frame_index]
        return (
            torch.from_numpy(hands[position].copy()).float(),
            torch.from_numpy(faces[position].copy()).float(),
        )

    def __getitem__(self, index):
        path, key, text_key, frame_index, mask_path = self.items[index]
        frame = frame_from_node(self._handle(path)["video"][key], frame_index)
        image = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255.0)
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                (self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )[0].clamp_(0, 1)
        hand, face = self._detail_masks(mask_path, frame_index)
        if tuple(hand.shape[-2:]) != (self.image_size, self.image_size):
            hand = F.interpolate(
                hand[None, None], (self.image_size, self.image_size), mode="nearest"
            )[0, 0]
        if tuple(face.shape[-2:]) != (self.image_size, self.image_size):
            face = F.interpolate(
                face[None, None], (self.image_size, self.image_size), mode="nearest"
            )[0, 0]
        hand = dilate_mask_in_latent(
            hand,
            radius=self.hand_mask_latent_radius,
            latent_size=self.hand_mask_latent_size,
            threshold=self.hand_mask_threshold,
        )
        face = dilate_mask_in_latent(
            face,
            radius=self.face_mask_latent_radius,
            latent_size=self.face_mask_latent_size,
            threshold=self.face_mask_threshold,
        )
        return {
            "image": image.mul(2).sub(1),
            "hand_mask": hand.unsqueeze(0).clamp_(0, 1),
            "face_mask": face.unsqueeze(0).clamp_(0, 1),
            "text_key": text_key,
            "frame_index": frame_index,
        }


class ThesisReconstructionFrames(Dataset):
    """Frame dataset backed by the thesis RGB and hand-mask HDF5 shards.

    Face supervision is derived from the fused COCO-WholeBody face landmarks.
    The same square center crop used by ``prepare_signvip_word_hdf5.py`` is
    applied before rasterizing the face hull, so every condition shares the
    exact 256x256 coordinate system.
    """

    def __init__(
        self,
        hdf5_root,
        hand_mask_root,
        fusion_root,
        image_size,
        face_score_threshold=0.30,
        face_margin=0.08,
        hand_mask_latent_radius=0,
        hand_mask_latent_size=None,
        hand_mask_threshold=0.05,
        face_mask_latent_radius=0,
        face_mask_latent_size=None,
        face_mask_threshold=0.05,
    ):
        self.image_size = int(image_size)
        self.face_score_threshold = float(face_score_threshold)
        self.face_margin = float(face_margin)
        self.hand_mask_latent_radius = int(hand_mask_latent_radius)
        self.hand_mask_latent_size = int(
            hand_mask_latent_size or max(1, self.image_size // 8)
        )
        self.hand_mask_threshold = float(hand_mask_threshold)
        self.face_mask_latent_radius = int(face_mask_latent_radius)
        self.face_mask_latent_size = int(
            face_mask_latent_size or max(1, self.image_size // 8)
        )
        self.face_mask_threshold = float(face_mask_threshold)
        self.fusion_root = Path(fusion_root)

        main_paths = sorted(Path(hdf5_root).glob("*.hdf5")) + sorted(
            Path(hdf5_root).glob("*.h5")
        )
        mask_paths = sorted(Path(hand_mask_root).glob("*.hdf5")) + sorted(
            Path(hand_mask_root).glob("*.h5")
        )
        if not main_paths:
            raise FileNotFoundError(f"No thesis RGB HDF5 shards under {hdf5_root}")
        if not mask_paths:
            raise FileNotFoundError(
                f"No thesis hand-mask HDF5 shards under {hand_mask_root}"
            )
        if not self.fusion_root.is_dir():
            raise FileNotFoundError(f"Fused-keypoint root is missing: {fusion_root}")

        mask_index = {}
        for path in mask_paths:
            with h5py.File(path, "r") as handle:
                missing = {"hand_mask", "len", "text_key"} - set(handle.keys())
                if missing:
                    raise KeyError(f"{path} is missing groups: {sorted(missing)}")
                for key in numeric_keys(handle["hand_mask"]):
                    text_key = decode_text(handle["text_key"][key][()])
                    if text_key in mask_index:
                        raise ValueError(f"Duplicate hand-mask ID: {text_key}")
                    mask_index[text_key] = (
                        str(path),
                        str(key),
                        int(handle["len"][key][()]),
                    )

        self.items = []
        for path in main_paths:
            with h5py.File(path, "r") as handle:
                missing = {"video", "len", "text_key"} - set(handle.keys())
                if missing:
                    raise KeyError(f"{path} is missing groups: {sorted(missing)}")
                for key in numeric_keys(handle["video"]):
                    text_key = decode_text(handle["text_key"][key][()])
                    if text_key not in mask_index:
                        raise KeyError(f"No hand-mask HDF5 entry for {text_key}")
                    mask_path, mask_key, mask_length = mask_index[text_key]
                    length = int(handle["len"][key][()])
                    if mask_length != length:
                        raise ValueError(
                            f"Frame-count mismatch for {text_key}: "
                            f"RGB={length}, hand_mask={mask_length}"
                        )
                    fusion_path = self.fusion_root / f"{text_key}.npz"
                    if not fusion_path.is_file():
                        raise FileNotFoundError(
                            f"Missing fused keypoints for {text_key}: {fusion_path}"
                        )
                    for frame_index in range(length):
                        self.items.append(
                            (
                                str(path),
                                str(key),
                                text_key,
                                frame_index,
                                mask_path,
                                mask_key,
                                str(fusion_path),
                            )
                        )

        self._handles = {}
        self._pid = None
        self._fusion_arrays = OrderedDict()

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_pid"] = None
        state["_fusion_arrays"] = OrderedDict()
        return state

    def _handle(self, path):
        pid = os.getpid()
        if self._pid != pid:
            for handle in self._handles.values():
                handle.close()
            self._handles = {}
            self._pid = pid
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]

    def _fusion(self, path):
        cached = self._fusion_arrays.pop(path, None)
        if cached is None:
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata"].item()))
                frame_indices = np.asarray(archive["frame_indices"], dtype=np.int64)
                keypoints = np.asarray(archive["keypoints"], dtype=np.float32)
            cached = (
                {int(frame): i for i, frame in enumerate(frame_indices)},
                keypoints,
                int(metadata["frame_width"]),
                int(metadata["frame_height"]),
            )
        self._fusion_arrays[path] = cached
        while len(self._fusion_arrays) > 16:
            self._fusion_arrays.popitem(last=False)
        return cached

    def _face_mask(self, fusion_path, frame_index):
        positions, keypoints, width, height = self._fusion(fusion_path)
        if frame_index not in positions:
            raise KeyError(f"{fusion_path} has no fused frame {frame_index}")
        face = keypoints[positions[frame_index], 23:91].copy()
        side = min(width, height)
        x0 = (width - side) / 2.0
        y0 = (height - side) / 2.0
        face[:, 0] = (face[:, 0] * width - x0) / side
        face[:, 1] = (face[:, 1] * height - y0) / side
        valid = (
            (face[:, 2] >= self.face_score_threshold)
            & (face[:, 0] >= 0)
            & (face[:, 0] <= 1)
            & (face[:, 1] >= 0)
            & (face[:, 1] <= 1)
        )
        mask = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        if int(valid.sum()) < 6:
            return torch.from_numpy(mask).float()
        points = np.rint(face[valid, :2] * (self.image_size - 1)).astype(np.int32)
        hull = cv2.convexHull(points.reshape(-1, 1, 2))
        cv2.fillConvexPoly(mask, hull, 1)
        x, y, w, h = cv2.boundingRect(hull)
        radius = max(1, int(round(max(w, h) * self.face_margin)))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
        )
        return torch.from_numpy(cv2.dilate(mask, kernel)).float()

    def __getitem__(self, index):
        (
            path,
            key,
            text_key,
            frame_index,
            mask_path,
            mask_key,
            fusion_path,
        ) = self.items[index]
        frame = frame_from_node(self._handle(path)["video"][key], frame_index)
        hand = frame_from_node(
            self._handle(mask_path)["hand_mask"][mask_key], frame_index
        )
        image = torch.from_numpy(frame.copy()).permute(2, 0, 1).float().div_(255.0)
        hand = torch.from_numpy(hand.copy()).float()
        if tuple(image.shape[-2:]) != (self.image_size, self.image_size):
            image = F.interpolate(
                image.unsqueeze(0),
                (self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
            )[0].clamp_(0, 1)
        if tuple(hand.shape[-2:]) != (self.image_size, self.image_size):
            hand = F.interpolate(
                hand[None, None],
                (self.image_size, self.image_size),
                mode="nearest",
            )[0, 0]
        hand = dilate_mask_in_latent(
            hand,
            radius=self.hand_mask_latent_radius,
            latent_size=self.hand_mask_latent_size,
            threshold=self.hand_mask_threshold,
        )
        face = self._face_mask(fusion_path, frame_index)
        face = dilate_mask_in_latent(
            face,
            radius=self.face_mask_latent_radius,
            latent_size=self.face_mask_latent_size,
            threshold=self.face_mask_threshold,
        )
        return {
            "image": image.mul(2).sub(1),
            "hand_mask": hand.unsqueeze(0).clamp_(0, 1),
            "face_mask": face.unsqueeze(0).clamp_(0, 1),
            "text_key": text_key,
            "frame_index": frame_index,
        }


def charbonnier_error(prediction, target, epsilon):
    difference = prediction.float() - target.float()
    return torch.sqrt(difference.square() + epsilon * epsilon) - epsilon


def reconstruction_losses(
    prediction,
    target,
    hand_mask,
    face_mask,
    cfg,
    perceptual_cfg=None,
    clip_model=None,
    dino_model=None,
    lpips_model=None,
):
    # Build differentiable zeros in float32. ``prediction.sum() * 0`` can
    # overflow while prediction is fp16 (inf * 0 = NaN), especially when a
    # batch contains no valid hand mask.
    zero = prediction.float().mean() * 0.0
    error = charbonnier_error(prediction, target, float(cfg.charbonnier_eps))
    global_loss = error.mean()
    mask = hand_mask.float().clamp(0, 1)
    if mask.shape[-2:] != error.shape[-2:]:
        mask = F.interpolate(mask, error.shape[-2:], mode="area")
    areas = mask.sum(dim=(1, 2, 3))
    denominators = (areas * error.shape[1]).clamp_min(1e-6)
    per_image = (error * mask).sum(dim=(1, 2, 3)) / denominators
    valid = areas >= float(cfg.min_mask_pixels)
    hand_loss = per_image[valid].mean() if valid.any() else zero

    face = face_mask.float().clamp(0, 1)
    if face.shape[-2:] != error.shape[-2:]:
        face = F.interpolate(face, error.shape[-2:], mode="area")
    face_areas = face.sum(dim=(1, 2, 3))
    face_denominators = (face_areas * error.shape[1]).clamp_min(1e-6)
    face_per_image = (error * face).sum(dim=(1, 2, 3)) / face_denominators
    valid_faces = face_areas >= float(cfg.min_mask_pixels)
    face_loss = (
        face_per_image[valid_faces].mean()
        if valid_faces.any()
        else zero
    )
    hand_clip = zero
    hand_dino = zero
    hand_lpips = zero
    face_clip = zero
    face_dino = zero
    face_lpips = zero
    valid_hand_rois = 0
    valid_face_rois = 0
    if perceptual_cfg is not None and bool(perceptual_cfg.get("enabled", False)):
        predicted_rgb = prediction.float().add(1).div(2).clamp(0, 1)
        target_rgb = target.float().add(1).div(2).clamp(0, 1)
        predicted_crops, target_crops = crop_resize_region_pairs(
            predicted_rgb,
            target_rgb,
            mask,
            output_size=int(perceptual_cfg.get("crop_size", 128)),
            padding=float(perceptual_cfg.get("crop_padding", 0.2)),
            threshold=float(perceptual_cfg.get("mask_threshold", 0.5)),
            min_mask_pixels=float(cfg.min_mask_pixels),
        )
        if predicted_crops.shape[0] > 0:
            valid_hand_rois = int(predicted_crops.shape[0])
            use_checkpoint = bool(perceptual_cfg.get("gradient_checkpointing", True))
            if float(perceptual_cfg.get("hand_clip_weight", 0.0)) != 0:
                hand_clip = clip_loss(
                    clip_model,
                    predicted_crops,
                    target_crops,
                    gradient_checkpointing=use_checkpoint,
                )
            if float(perceptual_cfg.get("hand_dino_weight", 0.0)) != 0:
                hand_dino = dino_loss(
                    dino_model,
                    predicted_crops,
                    target_crops,
                    gradient_checkpointing=use_checkpoint,
                )
            if float(perceptual_cfg.get("hand_lpips_weight", 0.0)) != 0:
                if lpips_model is None:
                    raise RuntimeError("hand_lpips_weight is nonzero but LPIPS is not loaded")
                if use_checkpoint and predicted_crops.requires_grad:
                    hand_lpips = checkpoint(
                        lambda crops: lpips_model(
                            crops,
                            target_crops,
                            normalize=True,
                        ),
                        predicted_crops,
                        use_reentrant=False,
                    ).float().mean()
                else:
                    hand_lpips = lpips_model(
                        predicted_crops,
                        target_crops,
                        normalize=True,
                    ).float().mean()
        predicted_face_crops, target_face_crops = crop_resize_region_pairs(
            predicted_rgb,
            target_rgb,
            face,
            output_size=int(perceptual_cfg.get("crop_size", 128)),
            padding=float(perceptual_cfg.get("crop_padding", 0.2)),
            threshold=float(perceptual_cfg.get("mask_threshold", 0.5)),
            min_mask_pixels=float(cfg.min_mask_pixels),
        )
        if predicted_face_crops.shape[0] > 0:
            valid_face_rois = int(predicted_face_crops.shape[0])
            use_checkpoint = bool(perceptual_cfg.get("gradient_checkpointing", True))
            if float(perceptual_cfg.get("face_clip_weight", 0.0)) != 0:
                face_clip = clip_loss(
                    clip_model,
                    predicted_face_crops,
                    target_face_crops,
                    gradient_checkpointing=use_checkpoint,
                )
            if float(perceptual_cfg.get("face_dino_weight", 0.0)) != 0:
                face_dino = dino_loss(
                    dino_model,
                    predicted_face_crops,
                    target_face_crops,
                    gradient_checkpointing=use_checkpoint,
                )
            if float(perceptual_cfg.get("face_lpips_weight", 0.0)) != 0:
                if lpips_model is None:
                    raise RuntimeError("face_lpips_weight is nonzero but LPIPS is not loaded")
                if use_checkpoint and predicted_face_crops.requires_grad:
                    face_lpips = checkpoint(
                        lambda crops: lpips_model(
                            crops,
                            target_face_crops,
                            normalize=True,
                        ),
                        predicted_face_crops,
                        use_reentrant=False,
                    ).float().mean()
                else:
                    face_lpips = lpips_model(
                        predicted_face_crops,
                        target_face_crops,
                        normalize=True,
                    ).float().mean()
    c511_loss = (
        float(cfg.global_weight) * global_loss
        + float(cfg.hand_weight) * hand_loss
        + float((perceptual_cfg or {}).get("hand_clip_weight", 0.0)) * hand_clip
        + float((perceptual_cfg or {}).get("hand_dino_weight", 0.0)) * hand_dino
    )
    weighted_hand_lpips = (
        float((perceptual_cfg or {}).get("hand_lpips_weight", 0.0)) * hand_lpips
    )
    c512_loss = c511_loss + weighted_hand_lpips
    weighted_face_loss = float(cfg.get("face_weight", 0.0)) * face_loss
    weighted_face_lpips = (
        float((perceptual_cfg or {}).get("face_lpips_weight", 0.0)) * face_lpips
    )
    face_perceptual_loss = (
        float((perceptual_cfg or {}).get("face_clip_weight", 0.0)) * face_clip
        + float((perceptual_cfg or {}).get("face_dino_weight", 0.0)) * face_dino
        + weighted_face_lpips
    )
    total = c512_loss + weighted_face_loss + face_perceptual_loss
    return (
        total,
        c512_loss,
        c511_loss,
        global_loss,
        hand_loss,
        face_loss,
        weighted_face_loss,
        hand_clip,
        hand_dino,
        hand_lpips,
        weighted_hand_lpips,
        face_clip,
        face_dino,
        face_lpips,
        weighted_face_lpips,
        int(valid.sum().item()),
        int(valid_faces.sum().item()),
        valid_hand_rois,
        valid_face_rois,
    )


def cosine_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def save_vae(vae, output_dir, step, metrics, selection_metric="loss"):
    output_dir.mkdir(parents=True, exist_ok=True)
    vae.save_pretrained(output_dir / "vae", safe_serialization=True)
    record = {"step": int(step), **{key: float(value) for key, value in metrics.items()}}
    record["selection_metric"] = str(selection_metric)
    record["selection_score"] = float(metrics[selection_metric])
    (output_dir / "metrics.json").write_text(json.dumps(record, indent=2) + "\n")


def validation_preview(images, reconstructions, hand_masks, face_masks, count):
    count = min(int(count), images.shape[0])
    ground_truth = images[:count].float().add(1).div(2).clamp(0, 1)
    predicted = reconstructions[:count].float().add(1).div(2).clamp(0, 1)
    errors = (ground_truth - predicted).abs().mul(5).clamp(0, 1)
    overlay = ground_truth.clone()
    masks = hand_masks[:count].float().clamp(0, 1)
    faces = face_masks[:count].float().clamp(0, 1)
    red = overlay.new_tensor((1.0, 0.05, 0.05)).view(1, 3, 1, 1)
    blue = overlay.new_tensor((0.05, 0.35, 1.0)).view(1, 3, 1, 1)
    overlay = overlay * (1 - 0.55 * masks) + red * (0.55 * masks)
    overlay = overlay * (1 - 0.45 * faces) + blue * (0.45 * faces)
    rows = [
        torch.cat((ground_truth[i], predicted[i], errors[i], overlay[i]), dim=2)
        for i in range(count)
    ]
    return torch.cat(rows, dim=1).cpu()


@torch.no_grad()
def validate(
    vae,
    loader,
    device,
    dtype,
    cfg,
    clip_model=None,
    dino_model=None,
    lpips_model=None,
):
    vae.eval()
    totals = {
        "loss": 0.0,
        "c5_12_loss": 0.0,
        "c5_11_loss": 0.0,
        "global_charbonnier": 0.0,
        "hand_charbonnier": 0.0,
        "face_charbonnier": 0.0,
        "face_charbonnier_weighted": 0.0,
        "hand_clip": 0.0,
        "hand_dino": 0.0,
        "hand_lpips_raw": 0.0,
        "hand_lpips_weighted": 0.0,
        "face_clip": 0.0,
        "face_dino": 0.0,
        "face_lpips_raw": 0.0,
        "face_lpips_weighted": 0.0,
    }
    batches = 0
    valid_hands = 0
    valid_faces = 0
    valid_hand_rois = 0
    valid_face_rois = 0
    preview = None
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["hand_mask"].to(device, non_blocking=True)
        face_masks = batch["face_mask"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            latents = vae.encode(images.to(dtype)).latent_dist.mode()
            reconstructions = vae.decode(latents).sample
        (
            loss,
            c512_loss,
            c511_loss,
            global_loss,
            hand_loss,
            face_loss,
            weighted_face_loss,
            hand_clip,
            hand_dino,
            hand_lpips,
            weighted_hand_lpips,
            face_clip,
            face_dino,
            face_lpips,
            weighted_face_lpips,
            valid,
            valid_face,
            valid_rois,
            valid_face_roi,
        ) = reconstruction_losses(
            reconstructions,
            images,
            masks,
            face_masks,
            cfg.loss,
            cfg.get("perceptual_loss", None),
            clip_model,
            dino_model,
            lpips_model,
        )
        totals["loss"] += float(loss)
        totals["c5_12_loss"] += float(c512_loss)
        totals["c5_11_loss"] += float(c511_loss)
        totals["global_charbonnier"] += float(global_loss)
        totals["hand_charbonnier"] += float(hand_loss)
        totals["face_charbonnier"] += float(face_loss)
        totals["face_charbonnier_weighted"] += float(weighted_face_loss)
        totals["hand_clip"] += float(hand_clip)
        totals["hand_dino"] += float(hand_dino)
        totals["hand_lpips_raw"] += float(hand_lpips)
        totals["hand_lpips_weighted"] += float(weighted_hand_lpips)
        totals["face_clip"] += float(face_clip)
        totals["face_dino"] += float(face_dino)
        totals["face_lpips_raw"] += float(face_lpips)
        totals["face_lpips_weighted"] += float(weighted_face_lpips)
        valid_hands += valid
        valid_faces += valid_face
        valid_hand_rois += valid_rois
        valid_face_rois += valid_face_roi
        if preview is None and int(cfg.get("log_validation_images", 0)) > 0:
            preview = validation_preview(
                images,
                reconstructions,
                masks,
                face_masks,
                int(cfg.log_validation_images),
            )
        batches += 1
        if batches >= int(cfg.validation_max_batches):
            break
    vae.train()
    vae.encoder.eval()
    vae.quant_conv.eval()

    if not batches:
        raise RuntimeError("Validation loader is empty")
    result = {key: value / batches for key, value in totals.items()}
    result["valid_hand_frames"] = valid_hands
    result["valid_face_frames"] = valid_faces
    result["valid_hand_rois"] = valid_hand_rois
    result["valid_face_rois"] = valid_face_rois
    return result, preview


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)
    override = os.environ.get("SIGNVIP_MAX_TRAIN_STEPS")
    if override:
        cfg.solver.max_train_steps = int(override)
    continuation_base_lr_override = os.environ.get(
        "SIGNVIP_CONTINUATION_BASE_LR"
    )
    batch_size_override = os.environ.get("SIGNVIP_BATCH_SIZE")
    if batch_size_override:
        cfg.dataloader.batch_size = int(batch_size_override)
    hand_radius_override = os.environ.get("SIGNVIP_HAND_MASK_LATENT_RADIUS")
    if hand_radius_override:
        cfg.dataset.hand_mask_latent_radius = int(hand_radius_override)
    face_radius_override = os.environ.get("SIGNVIP_FACE_MASK_LATENT_RADIUS")
    if face_radius_override:
        cfg.dataset.face_mask_latent_radius = int(face_radius_override)
    pretrained_vae_override = os.environ.get("SIGNVIP_PRETRAINED_VAE")
    if pretrained_vae_override:
        cfg.pretrained_vae = pretrained_vae_override
    experiment_override = os.environ.get("SIGNVIP_EXPERIMENT_NAME")
    if experiment_override:
        cfg.experiment_name = experiment_override
    workspace_override = os.environ.get("SIGNVIP_WORKSPACE_ROOT")
    if workspace_override:
        cfg.workspace_root = workspace_override
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[str(cfg.weight_dtype)]
    random.seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))

    dataset_format = str(cfg.dataset.get("format", "legacy_detail_masks"))
    if dataset_format == "thesis_hdf5":
        dataset_kwargs = {
            "fusion_root": cfg.dataset.fusion_root,
            "image_size": cfg.dataset.image_size,
            "face_score_threshold": cfg.dataset.get("face_score_threshold", 0.30),
            "face_margin": cfg.dataset.get("face_margin", 0.08),
            "hand_mask_latent_radius": cfg.dataset.get("hand_mask_latent_radius", 0),
            "hand_mask_latent_size": cfg.dataset.get(
                "hand_mask_latent_size", int(cfg.dataset.image_size) // 8
            ),
            "hand_mask_threshold": cfg.dataset.get("hand_mask_threshold", 0.05),
            "face_mask_latent_radius": cfg.dataset.get(
                "face_mask_latent_radius", 0
            ),
            "face_mask_latent_size": cfg.dataset.get(
                "face_mask_latent_size", int(cfg.dataset.image_size) // 8
            ),
            "face_mask_threshold": cfg.dataset.get("face_mask_threshold", 0.05),
        }
        train_dataset = ThesisReconstructionFrames(
            cfg.dataset.train_root,
            cfg.dataset.train_hand_mask_root,
            **dataset_kwargs,
        )
        dev_dataset = ThesisReconstructionFrames(
            cfg.dataset.dev_root,
            cfg.dataset.dev_hand_mask_root,
            **dataset_kwargs,
        )
    elif dataset_format in {"legacy_detail_masks", "grounded_sam2_npz"}:
        detail_mask_kwargs = {
            "mask_suffix": cfg.dataset.get("mask_suffix", None),
            "hand_mask_latent_radius": cfg.dataset.get(
                "hand_mask_latent_radius", 0
            ),
            "hand_mask_latent_size": cfg.dataset.get(
                "hand_mask_latent_size", int(cfg.dataset.image_size) // 8
            ),
            "hand_mask_threshold": cfg.dataset.get("hand_mask_threshold", 0.05),
            "face_mask_latent_radius": cfg.dataset.get(
                "face_mask_latent_radius", 0
            ),
            "face_mask_latent_size": cfg.dataset.get(
                "face_mask_latent_size", int(cfg.dataset.image_size) // 8
            ),
            "face_mask_threshold": cfg.dataset.get("face_mask_threshold", 0.05),
        }
        train_dataset = ReconstructionFrames(
            cfg.dataset.train_root,
            cfg.dataset.detail_mask_cache,
            "train",
            cfg.dataset.image_size,
            **detail_mask_kwargs,
        )
        dev_dataset = ReconstructionFrames(
            cfg.dataset.dev_root,
            cfg.dataset.detail_mask_cache,
            "dev",
            cfg.dataset.image_size,
            **detail_mask_kwargs,
        )
    else:
        raise ValueError(
            "dataset.format must be thesis_hdf5, grounded_sam2_npz, "
            "or legacy_detail_masks, "
            f"got {dataset_format!r}"
        )
    generator = torch.Generator().manual_seed(int(cfg.seed))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=True,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.dataloader.num_workers) > 0,
        drop_last=True,
        generator=generator,
    )
    validation_frames = min(
        len(dev_dataset),
        int(cfg.validation_max_batches) * int(cfg.dataloader.batch_size),
    )
    validation_indices = np.linspace(
        0, len(dev_dataset) - 1, validation_frames, dtype=np.int64
    ).tolist()
    dev_subset = Subset(dev_dataset, validation_indices)
    dev_loader = DataLoader(
        dev_subset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataloader.num_workers),
        pin_memory=True,
        persistent_workers=int(cfg.dataloader.num_workers) > 0,
    )

    vae = AutoencoderKL.from_pretrained(str(cfg.pretrained_vae), local_files_only=True).to(device, dtype)
    vae.requires_grad_(False)
    vae.post_quant_conv.requires_grad_(True)
    vae.decoder.requires_grad_(True)
    trainable = [parameter for parameter in vae.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    vae.train()
    vae.encoder.eval()
    vae.quant_conv.eval()

    perceptual_cfg = cfg.get("perceptual_loss", {})
    perceptual_enabled = bool(perceptual_cfg.get("enabled", False))
    clip_model = None
    dino_model = None
    lpips_model = None
    if perceptual_enabled:
        clip_model = load_clip_model(str(perceptual_cfg.clip_model), device, dtype)
        dino_model = load_dino_model(str(perceptual_cfg.dino_model), device, dtype)
        print(
            "Frozen hand-crop perceptual models: "
            f"CLIP={perceptual_cfg.clip_model}, DINO={perceptual_cfg.dino_model}"
        )
    if (
        float(perceptual_cfg.get("hand_lpips_weight", 0.0)) != 0
        or float(perceptual_cfg.get("face_lpips_weight", 0.0)) != 0
    ):
        import lpips

        lpips_model = lpips.LPIPS(net="alex", version="0.1").to(device)
        lpips_model.eval().requires_grad_(False)
        print(
            "Frozen local LPIPS: AlexNet v0.1, RGB [0,1], normalize=True, "
            f"hand_weight={perceptual_cfg.get('hand_lpips_weight', 0.0)}, "
            f"face_weight={perceptual_cfg.get('face_lpips_weight', 0.0)}"
        )

    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.solver.learning_rate),
        betas=(float(cfg.solver.adam_beta1), float(cfg.solver.adam_beta2)),
        weight_decay=float(cfg.solver.adam_weight_decay),
        eps=float(cfg.solver.adam_epsilon),
    )
    total_steps = int(cfg.solver.max_train_steps)
    selection_metric = str(cfg.get("checkpoint_selection", {}).get("metric", "loss"))
    early_stopping_cfg = cfg.get("early_stopping", {})
    early_stopping_enabled = bool(early_stopping_cfg.get("enabled", False))
    early_stopping_patience = int(early_stopping_cfg.get("patience", 20))
    early_stopping_min_delta = float(early_stopping_cfg.get("min_delta", 0.0))
    early_stopping_start_step = int(early_stopping_cfg.get("start_after_step", 0))
    if early_stopping_enabled and early_stopping_patience <= 0:
        raise ValueError("early_stopping.patience must be positive")
    supported_selection_metrics = {"loss", "hand_lpips_raw"}
    if selection_metric not in supported_selection_metrics:
        raise ValueError(
            f"Unsupported checkpoint-selection metric {selection_metric!r}; "
            f"choose one of {sorted(supported_selection_metrics)}"
        )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: cosine_multiplier(step, total_steps, int(cfg.solver.warmup_steps)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)

    resume_dir = Path(args.resume).resolve() if args.resume else None
    if resume_dir is not None:
        if not (resume_dir / "training_state.pt").is_file():
            raise FileNotFoundError(f"Resume state is missing: {resume_dir / 'training_state.pt'}")
        run_dir = (
            Path(args.output_dir).resolve()
            if args.output_dir
            else resume_dir.parent
        )
        run_dir.mkdir(parents=True, exist_ok=True)
        if not (run_dir / "config.yaml").is_file():
            OmegaConf.save(cfg, run_dir / "config.yaml")
    else:
        if args.output_dir:
            raise ValueError("--output-dir is only valid together with --resume")
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = Path(str(cfg.workspace_root)) / str(cfg.experiment_name) / timestamp
        run_dir.mkdir(parents=True, exist_ok=False)
        OmegaConf.save(cfg, run_dir / "config.yaml")
    status_path = run_dir / "training_status.json"
    status_path.write_text(
        json.dumps(
            {
                "status": "running",
                "target_step": int(total_steps),
                "resume_checkpoint": str(resume_dir) if resume_dir else None,
            },
            indent=2,
        )
        + "\n"
    )
    writer = SummaryWriter(run_dir / "log")
    start_step, best_score = 0, float("inf")
    early_stopping_bad_validations = 0
    if resume_dir is not None:
        state = torch.load(resume_dir / "training_state.pt", map_location="cpu")
        vae.post_quant_conv.load_state_dict(state["post_quant_conv"])
        vae.decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_step = int(state["step"])
        saved_selection_metric = str(state.get("selection_metric", "loss"))
        if saved_selection_metric != selection_metric:
            raise ValueError(
                "Cannot resume with a different checkpoint-selection metric: "
                f"checkpoint={saved_selection_metric}, config={selection_metric}"
            )
        best_score = float(state.get("best_score", state.get("best_loss", float("inf"))))
        early_stopping_bad_validations = int(
            state.get("early_stopping_bad_validations", 0)
        )
        # A process may stop after saving a newer versioned best but before the
        # next resumable checkpoint. Keep best.json authoritative so resuming
        # from the older checkpoint never replaces that newer best with a worse
        # validation result.
        best_metadata_path = run_dir / "best.json"
        try:
            best_metadata = json.loads(best_metadata_path.read_text(encoding="utf-8"))
            metadata_metric = str(best_metadata.get("selection_metric", selection_metric))
            metadata_score = float(best_metadata["selection_score"])
            metadata_step = int(best_metadata.get("step", 0))
            if (
                metadata_metric == selection_metric
                and math.isfinite(metadata_score)
                and metadata_score < best_score
            ):
                best_score = metadata_score
                if metadata_step > start_step:
                    early_stopping_bad_validations = 0
                print(
                    "Restored newer best.json state: "
                    f"step={metadata_step}, best_{selection_metric}={best_score:.8f}"
                )
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            pass
        # Extending the target horizon would otherwise make the cosine
        # multiplier large again (for example, 183k/1M) and cause a sudden LR
        # jump.  An explicit continuation base LR preserves a safe effective
        # LR while retaining optimizer moments and the global scheduler step.
        if continuation_base_lr_override:
            continuation_base_lr = float(continuation_base_lr_override)
            if continuation_base_lr <= 0:
                raise ValueError(
                    "SIGNVIP_CONTINUATION_BASE_LR must be positive, got "
                    f"{continuation_base_lr_override!r}"
                )
            scheduler.base_lrs = [
                continuation_base_lr for _ in optimizer.param_groups
            ]
            for group in optimizer.param_groups:
                group["initial_lr"] = continuation_base_lr

        # Recompute the LR at the same global step against the requested
        # total-step horizon. Optimizer moments and scheduler epoch remain
        # continuous.
        multiplier = cosine_multiplier(
            start_step,
            total_steps,
            int(cfg.solver.warmup_steps),
        )
        for group, base_lr in zip(optimizer.param_groups, scheduler.base_lrs):
            group["lr"] = float(base_lr) * multiplier
        scheduler._last_lr = [group["lr"] for group in optimizer.param_groups]
        print(
            f"Resuming in place from {resume_dir}: step={start_step}, "
            f"target={total_steps}, lr={scheduler.get_last_lr()[0]:.3e}, "
            f"best_{selection_metric}={best_score:.8f}"
        )
        if continuation_base_lr_override:
            print(
                "Continuation cosine base LR: "
                f"{float(continuation_base_lr_override):.3e} "
                "(SIGNVIP_CONTINUATION_BASE_LR)"
            )

    print(f"Run directory: {run_dir}")
    print(f"Train/dev frames: {len(train_dataset)}/{len(dev_dataset)}")
    print(f"Batch size: {int(cfg.dataloader.batch_size)}")
    if dataset_format in {"thesis_hdf5", "grounded_sam2_npz"}:
        print(
            "Hand-mask dilation: "
            f"r={int(cfg.dataset.get('hand_mask_latent_radius', 0))} on "
            f"{int(cfg.dataset.get('hand_mask_latent_size', int(cfg.dataset.image_size) // 8))}x"
            f"{int(cfg.dataset.get('hand_mask_latent_size', int(cfg.dataset.image_size) // 8))} latent, "
            f"threshold>{float(cfg.dataset.get('hand_mask_threshold', 0.05)):g}"
        )
        print(
            "Face-mask dilation: "
            f"r={int(cfg.dataset.get('face_mask_latent_radius', 0))} on "
            f"{int(cfg.dataset.get('face_mask_latent_size', int(cfg.dataset.image_size) // 8))}x"
            f"{int(cfg.dataset.get('face_mask_latent_size', int(cfg.dataset.image_size) // 8))} latent, "
            f"threshold>{float(cfg.dataset.get('face_mask_threshold', 0.05)):g}"
        )
    print("Frozen: encoder + quant_conv; trainable: post_quant_conv + decoder")
    print(f"Best-checkpoint selection: minimum validation {selection_metric}")
    if early_stopping_enabled:
        print(
            "Early stopping: "
            f"patience={early_stopping_patience} validations, "
            f"min_delta={early_stopping_min_delta:g}, "
            f"start_after_step={early_stopping_start_step}"
        )
    print(
        f"Loss = {cfg.loss.global_weight} * global Charbonnier + "
        f"{cfg.loss.hand_weight} * normalized hand Charbonnier + "
        f"{cfg.loss.get('face_weight', 0.0)} * normalized face Charbonnier"
    )
    iterator = iter(train_loader)
    progress = tqdm(
        range(start_step, total_steps),
        desc=f"{cfg.experiment_name} VAE reconstruction",
        initial=start_step,
        total=total_steps,
    )
    last_step = start_step
    stopped_early = False
    for zero_step in progress:
        step = zero_step + 1
        last_step = step
        should_stop = False
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train_loader)
            batch = next(iterator)
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["hand_mask"].to(device, non_blocking=True)
        face_masks = batch["face_mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            latents = vae.encode(images.to(dtype)).latent_dist.mode()
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            reconstructions = vae.decode(latents).sample
            (
                loss,
                c512_loss,
                c511_loss,
                global_loss,
                hand_loss,
                face_loss,
                weighted_face_loss,
                hand_clip,
                hand_dino,
                hand_lpips,
                weighted_hand_lpips,
                face_clip,
                face_dino,
                face_lpips,
                weighted_face_lpips,
                valid_hands,
                valid_faces,
                valid_hand_rois,
                valid_face_rois,
            ) = reconstruction_losses(
                reconstructions,
                images,
                masks,
                face_masks,
                cfg.loss,
                perceptual_cfg,
                clip_model,
                dino_model,
                lpips_model,
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        decoder_grad_norm = None
        if step % int(cfg.log_steps) == 0:
            squared_grad_norms = [
                parameter.grad.detach().float().norm().square()
                for parameter in vae.decoder.parameters()
                if parameter.grad is not None
            ]
            if squared_grad_norms:
                decoder_grad_norm = float(torch.stack(squared_grad_norms).sum().sqrt())
        torch.nn.utils.clip_grad_norm_(trainable, float(cfg.solver.max_grad_norm))
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        if step % int(cfg.log_steps) == 0:
            values = {
                "train/loss": float(loss),
                "train/c5_12_loss": float(c512_loss),
                "train/c5_11_loss": float(c511_loss),
                "train/global_charbonnier": float(global_loss),
                "train/hand_charbonnier": float(hand_loss),
                "train/face_charbonnier": float(face_loss),
                "train/face_charbonnier_weighted": float(weighted_face_loss),
                "train/hand_clip": float(hand_clip),
                "train/hand_dino": float(hand_dino),
                "train/hand_lpips_raw": float(hand_lpips),
                "train/hand_lpips_weighted": float(weighted_hand_lpips),
                "train/face_clip": float(face_clip),
                "train/face_dino": float(face_dino),
                "train/face_lpips_raw": float(face_lpips),
                "train/face_lpips_weighted": float(weighted_face_lpips),
                "train/valid_hand_frames": valid_hands,
                "train/valid_face_frames": valid_faces,
                "train/valid_hand_rois": valid_hand_rois,
                "train/valid_face_rois": valid_face_rois,
                "train/lr": scheduler.get_last_lr()[0],
            }
            if decoder_grad_norm is not None:
                values["train/decoder_grad_norm"] = decoder_grad_norm
            for key, value in values.items():
                writer.add_scalar(key, value, step)
            progress.set_postfix(loss=f"{float(loss):.4f}", hand=f"{float(hand_loss):.4f}")

        if step % int(cfg.validation_steps) == 0 or step == total_steps:
            metrics, preview = validate(
                vae,
                dev_loader,
                device,
                dtype,
                cfg,
                clip_model,
                dino_model,
                lpips_model,
            )
            for key, value in metrics.items():
                writer.add_scalar(f"validation/{key}", value, step)
            if preview is not None:
                writer.add_image("validation/GT_recon_error_handmask_facemask", preview, step)
            print(f"step={step} validation={metrics}")
            selection_score = float(metrics[selection_metric])
            if not math.isfinite(selection_score):
                raise FloatingPointError(
                    f"Validation {selection_metric} is not finite at step {step}: "
                    f"{selection_score}"
                )
            improved = selection_score < best_score - early_stopping_min_delta
            if improved:
                best_score = selection_score
                early_stopping_bad_validations = 0
                if bool(cfg.get("versioned_best", False)):
                    best_dir = run_dir / f"best-step-{step}"
                    (run_dir / "best.json").write_text(
                        json.dumps(
                            {
                                "step": int(step),
                                "path": best_dir.name,
                                "selection_metric": selection_metric,
                                "selection_score": selection_score,
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                else:
                    best_dir = run_dir / "best"
                    if best_dir.exists():
                        shutil.rmtree(best_dir)
                save_vae(vae, best_dir, step, metrics, selection_metric)
            elif early_stopping_enabled and step >= early_stopping_start_step:
                early_stopping_bad_validations += 1
                print(
                    "Early stopping: no significant improvement "
                    f"({early_stopping_bad_validations}/{early_stopping_patience})"
                )
                should_stop = (
                    early_stopping_bad_validations >= early_stopping_patience
                )

        if (
            step % int(cfg.checkpointing_steps) == 0
            or step == total_steps
            or should_stop
        ):
            checkpoint = run_dir / f"checkpoint-{step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "step": step,
                    "best_loss": best_score if selection_metric == "loss" else float("inf"),
                    "best_score": best_score,
                    "selection_metric": selection_metric,
                    "early_stopping_bad_validations": early_stopping_bad_validations,
                    "post_quant_conv": vae.post_quant_conv.state_dict(),
                    "decoder": vae.decoder.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                },
                checkpoint / "training_state.pt",
            )
            max_checkpoints = cfg.get("max_checkpoints", None)
            if max_checkpoints is not None and int(max_checkpoints) > 0:
                checkpoints = sorted(
                    run_dir.glob("checkpoint-*"),
                    key=lambda path: int(path.name.rsplit("-", 1)[1]),
                )
                for old in checkpoints[: -int(max_checkpoints)]:
                    shutil.rmtree(old)
        if should_stop:
            stopped_early = True
            print(
                f"Early stopping triggered at step {step}; "
                f"best validation {selection_metric}={best_score:.8f}"
            )
            break
    writer.close()
    status_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "reason": "early_stopping" if stopped_early else "max_steps",
                "last_step": int(last_step),
                "target_step": int(total_steps),
                "best_score": float(best_score),
                "selection_metric": selection_metric,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Training complete: {run_dir}")


if __name__ == "__main__":
    main()
