"""HDF5 adapter for the center-cropped SignGen/SignViP data."""

import glob
import json
import os
import random
from typing import List, Tuple, Union

import h5py
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import (
    Compose,
    Normalize,
    PILToTensor,
    RandomResizedCrop,
)
from transformers import CLIPImageProcessor


class _ZeroOneNormalize:
    def __call__(self, image):
        return image.float().div(255)


def _as_list(value):
    return [value] if isinstance(value, (str, os.PathLike)) else list(value or [])


def _shards(root: str) -> List[str]:
    root = os.fspath(root)
    if os.path.isfile(root):
        return [root]
    paths = sorted(glob.glob(os.path.join(root, "*.hdf5")))
    paths += sorted(glob.glob(os.path.join(root, "*.h5")))
    if not paths:
        raise FileNotFoundError(f"No HDF5 shards found under: {root}")
    return paths


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode(value.item())
    return str(value)


def _numeric_keys(group):
    return sorted(group.keys(), key=lambda key: int(key))


class HDF5SignLangVideoDataset(Dataset):
    """Return the same tensors as ``SignLangVideoDataset`` from paired HDF5.

    HDF5 handles are opened lazily per worker/process and are deliberately not
    serialized. Main and HAMER shards are joined by ``text_key``, so shard and
    item ordering do not need to match.
    """

    def __init__(
        self,
        frame_size: Tuple[int, int] = (128, 128),
        frame_scale: Tuple[float, float] = (1.0, 1.0),
        frame_ratio: Tuple[float, float] = (1.0, 1.0),
        roots: Union[List[str], str] = None,
        hamer_roots: Union[List[str], str] = None,
        hand_mask_roots: Union[List[str], str] = None,
        fusion_keypoint_root: str = None,
        fusion_keypoint_strict: bool = True,
        face_score_threshold: float = 0.30,
        face_margin: float = 0.08,
        sample_rate: int = 1,
        num_frames: int = 1,
        ref_margin: int = 5,
        uncond_ratio: float = 0.1,
        mask_ratio: float = 0,
        mask_thershold: float = 0.85,
        skip_ratio: float = 0,
        use_sample_random: bool = False,
        random_sample: bool = True,
        sk_mask_ratio: float = 0,
        hamer_mask_ratio: float = 0,
        both_mask_ratio: float = 0,
        detail_mask_cache: str = None,
        detail_mask_split: str = "train",
        detail_mask_strict: bool = True,
        hand_mask_strict: bool = True,
        pose_keypoint_cache: str = None,
        pose_keypoint_split: str = "train",
        pose_keypoint_strict: bool = True,
        hand_roi_score_threshold: float = 0.30,
        hand_roi_margin: float = 0.20,
        full_video: bool = False,
        centered_clip: bool = False,
        reference_mode: str = "outside_clip",
        conditions_only: bool = False,
        **_unused,
    ):
        super().__init__()
        del uncond_ratio, mask_thershold, skip_ratio
        self.frame_size = tuple(frame_size)
        self.frame_scale = tuple(frame_scale)
        self.frame_ratio = tuple(frame_ratio)
        self.sample_rate = sample_rate if not use_sample_random or random.random() <= 0.5 else 1
        self.num_frames = int(num_frames)
        self.ref_margin = int(ref_margin)
        self.mask_ratio = float(mask_ratio)
        self.random_sample = bool(random_sample)
        self.sk_mask_ratio = float(sk_mask_ratio)
        self.hamer_mask_ratio = float(hamer_mask_ratio)
        self.both_mask_ratio = float(both_mask_ratio)
        self.detail_mask_strict = bool(detail_mask_strict)
        self.hand_mask_strict = bool(hand_mask_strict)
        self.fusion_keypoint_strict = bool(fusion_keypoint_strict)
        self.face_score_threshold = float(face_score_threshold)
        self.face_margin = float(face_margin)
        self.pose_keypoint_strict = bool(pose_keypoint_strict)
        self.hand_roi_score_threshold = float(hand_roi_score_threshold)
        self.hand_roi_margin = float(hand_roi_margin)
        self.full_video = bool(full_video)
        self.centered_clip = bool(centered_clip)
        self.reference_mode = str(reference_mode)
        self.conditions_only = bool(conditions_only)
        if self.reference_mode not in {"outside_clip", "first"}:
            raise ValueError(
                "reference_mode must be either 'outside_clip' or 'first'"
            )
        self.detail_mask_paths = {}
        self.pose_keypoint_paths = {}
        self.fusion_keypoint_paths = {}
        if fusion_keypoint_root:
            fusion_root = os.path.abspath(
                os.path.expanduser(str(fusion_keypoint_root))
            )
            if not os.path.isdir(fusion_root):
                raise FileNotFoundError(
                    f"Fused-keypoint root does not exist: {fusion_root}"
                )
            self.fusion_keypoint_paths = {
                os.path.splitext(filename)[0]: os.path.join(fusion_root, filename)
                for filename in os.listdir(fusion_root)
                if filename.endswith(".npz")
            }
        if detail_mask_cache:
            split_dir = os.path.join(
                os.path.abspath(os.path.expanduser(str(detail_mask_cache))),
                str(detail_mask_split),
            )
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"Detail-mask split does not exist: {split_dir}")
            if self.frame_size[0] != self.frame_size[1]:
                raise ValueError(
                    "Grounded-SAM2 detail masks require a square frame_size, "
                    f"got {self.frame_size}"
                )
            suffix = (
                "_grounded_sam2_face_hands_384_to"
                f"{self.frame_size[0]}_v4.npz"
            )
            self.detail_mask_paths = {
                filename[: -len(suffix)]: os.path.join(split_dir, filename)
                for filename in os.listdir(split_dir)
                if filename.endswith(suffix)
            }
        if pose_keypoint_cache:
            split_dir = os.path.join(
                os.path.abspath(os.path.expanduser(str(pose_keypoint_cache))),
                str(pose_keypoint_split),
            )
            if not os.path.isdir(split_dir):
                raise FileNotFoundError(f"DWPose-keypoint split does not exist: {split_dir}")
            suffix = "_dwpose384_keypoints_v1.npz"
            self.pose_keypoint_paths = {
                filename[: -len(suffix)]: os.path.join(split_dir, filename)
                for filename in os.listdir(split_dir)
                if filename.endswith(suffix)
            }
        if self.num_frames < 1:
            raise ValueError("num_frames must be at least 1")

        main_paths = [path for root in _as_list(roots) for path in _shards(root)]
        hamer_paths = [path for root in _as_list(hamer_roots) for path in _shards(root)]
        if not hamer_paths:
            raise ValueError("hamer_roots is required for SignViP two-condition training")

        hamer_by_text_key = {}
        for path in hamer_paths:
            with h5py.File(path, "r") as handle:
                for key in _numeric_keys(handle["len"]):
                    text_key = _decode(handle["text_key"][key][()])
                    if text_key in hamer_by_text_key:
                        raise ValueError(f"Duplicate HAMER text_key: {text_key}")
                    hamer_by_text_key[text_key] = (path, key, int(handle["len"][key][()]))

        hand_mask_by_text_key = {}
        hand_mask_paths = [
            path for root in _as_list(hand_mask_roots) for path in _shards(root)
        ]
        for path in hand_mask_paths:
            with h5py.File(path, "r") as handle:
                missing_groups = {"hand_mask", "len", "text_key"} - set(handle.keys())
                if missing_groups:
                    raise KeyError(
                        f"{path} is missing hand-mask groups: {sorted(missing_groups)}"
                    )
                for key in _numeric_keys(handle["len"]):
                    text_key = _decode(handle["text_key"][key][()])
                    if text_key in hand_mask_by_text_key:
                        raise ValueError(f"Duplicate hand-mask text_key: {text_key}")
                    hand_mask_by_text_key[text_key] = (
                        path,
                        key,
                        int(handle["len"][key][()]),
                    )

        self.items = []
        for path in main_paths:
            with h5py.File(path, "r") as handle:
                missing_groups = {"video", "pose_video", "len", "text_key"} - set(handle.keys())
                if missing_groups:
                    raise KeyError(f"{path} is missing groups: {sorted(missing_groups)}")
                for key in _numeric_keys(handle["len"]):
                    text_key = _decode(handle["text_key"][key][()])
                    if text_key not in hamer_by_text_key:
                        raise KeyError(f"No HAMER sample for text_key: {text_key}")
                    hamer_path, hamer_key, hamer_len = hamer_by_text_key[text_key]
                    length = int(handle["len"][key][()])
                    if length != hamer_len:
                        raise ValueError(
                            f"Frame count mismatch for {text_key}: main={length}, HAMER={hamer_len}"
                        )
                    mask_item = hand_mask_by_text_key.get(text_key)
                    if hand_mask_paths and mask_item is None and self.hand_mask_strict:
                        raise KeyError(f"No hand mask for text_key: {text_key}")
                    if mask_item is not None and length != mask_item[2]:
                        raise ValueError(
                            f"Frame count mismatch for {text_key}: "
                            f"main={length}, hand_mask={mask_item[2]}"
                        )
                    required = (
                        1
                        if self.full_video
                        else (self.num_frames - 1) * self.sample_rate + 1
                    )
                    if length >= required:
                        self.items.append(
                            (
                                path,
                                key,
                                hamer_path,
                                hamer_key,
                                mask_item[0] if mask_item else None,
                                mask_item[1] if mask_item else None,
                                length,
                                text_key,
                            )
                        )
        if not self.items:
            raise ValueError("No HDF5 samples are long enough for the requested clip")

        aspect = self.frame_size[1] / self.frame_size[0]
        self.image_transform = Compose(
            [
                PILToTensor(),
                _ZeroOneNormalize(),
                RandomResizedCrop(
                    self.frame_size,
                    scale=self.frame_scale,
                    ratio=(aspect, aspect),
                    antialias=True,
                ),
                Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        self.clip_image_processor = CLIPImageProcessor()
        self._handles = {}
        self._handle_pid = None

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_handles"] = {}
        state["_handle_pid"] = None
        return state

    def _handle(self, path):
        pid = os.getpid()
        if self._handle_pid != pid:
            self._close_handles()
            self._handle_pid = pid
        if path not in self._handles:
            self._handles[path] = h5py.File(path, "r", swmr=True)
        return self._handles[path]

    def _close_handles(self):
        for handle in getattr(self, "_handles", {}).values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}

    def __del__(self):
        self._close_handles()

    def _transform(self, array, rng_state):
        torch.set_rng_state(rng_state)
        return self.image_transform(Image.fromarray(np.asarray(array, dtype=np.uint8)))

    def _transform_mask(self, array, rng_state):
        # Reuse the exact crop parameters applied to RGB/conditions. Undo the
        # final [-1, 1] normalization and retain one soft mask channel.
        transformed = self._transform(np.asarray(array, dtype=np.uint8) * 255, rng_state)
        return transformed[0].add(1).div(2).clamp(0, 1)

    def _detail_masks(self, text_key, frame_ids, rng_state):
        if not self.detail_mask_paths:
            return None, None
        path = self.detail_mask_paths.get(text_key)
        if path is None:
            if self.detail_mask_strict:
                raise FileNotFoundError(f"No Grounded-SAM2 mask for {text_key}")
            empty = torch.zeros((1, self.num_frames, *self.frame_size))
            return empty, empty.clone()
        with np.load(path) as data:
            positions = {
                int(frame): position
                for position, frame in enumerate(data["frame_indices"])
            }
            hands = []
            faces = []
            for frame in frame_ids:
                frame = int(frame)
                if frame not in positions:
                    if self.detail_mask_strict:
                        raise KeyError(f"Mask {path} does not contain frame {frame}")
                    hands.append(torch.zeros(self.frame_size))
                    faces.append(torch.zeros(self.frame_size))
                    continue
                position = positions[frame]
                hands.append(self._transform_mask(data["hands"][position], rng_state))
                faces.append(self._transform_mask(data["face"][position], rng_state))
        return torch.stack(hands).unsqueeze(0), torch.stack(faces).unsqueeze(0)

    def _fused_face_masks(self, text_key, frame_ids, rng_state):
        """Rasterize thesis face ROIs from fused COCO-WholeBody landmarks."""
        if not self.fusion_keypoint_paths:
            return None
        path = self.fusion_keypoint_paths.get(text_key)
        if path is None:
            if self.fusion_keypoint_strict:
                raise FileNotFoundError(f"No fused keypoints for {text_key}")
            return torch.zeros((1, len(frame_ids), *self.frame_size))

        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            indices = np.asarray(archive["frame_indices"], dtype=np.int64)
            keypoints = np.asarray(archive["keypoints"], dtype=np.float32)
        positions = {int(frame): i for i, frame in enumerate(indices)}
        source_width = int(metadata["frame_width"])
        source_height = int(metadata["frame_height"])
        side = min(source_width, source_height)
        crop_x = (source_width - side) / 2.0
        crop_y = (source_height - side) / 2.0
        mask_height, mask_width = self.frame_size
        masks = []

        for frame_id in frame_ids:
            frame_id = int(frame_id)
            if frame_id not in positions:
                if self.fusion_keypoint_strict:
                    raise KeyError(f"{path} has no fused frame {frame_id}")
                masks.append(torch.zeros(self.frame_size))
                continue
            face = keypoints[positions[frame_id], 23:91].copy()
            face[:, 0] = (face[:, 0] * source_width - crop_x) / side
            face[:, 1] = (face[:, 1] * source_height - crop_y) / side
            valid = (
                (face[:, 2] >= self.face_score_threshold)
                & np.isfinite(face).all(axis=1)
                & (face[:, 0] >= 0.0) & (face[:, 0] <= 1.0)
                & (face[:, 1] >= 0.0) & (face[:, 1] <= 1.0)
            )
            mask = np.zeros((mask_height, mask_width), dtype=np.uint8)
            if int(valid.sum()) >= 6:
                points = np.stack(
                    (
                        face[valid, 0] * (mask_width - 1),
                        face[valid, 1] * (mask_height - 1),
                    ),
                    axis=1,
                )
                points = np.rint(points).astype(np.int32)
                hull = cv2.convexHull(points.reshape(-1, 1, 2))
                cv2.fillConvexPoly(mask, hull, 1)
                _, _, width, height = cv2.boundingRect(hull)
                radius = max(
                    1, int(round(max(width, height) * self.face_margin))
                )
                kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (2 * radius + 1, 2 * radius + 1),
                )
                mask = cv2.dilate(mask, kernel)
            masks.append(self._transform_mask(mask, rng_state))
        return torch.stack(masks).unsqueeze(0)

    def _hand_roi_masks(self, text_key, frame_ids, rng_state):
        """Rasterize separate left/right DWPose ROIs before the shared crop.

        DWPose stores 21 joints for each hand at indices 92:113 and 113:134.
        The person with the strongest mean body confidence is selected per
        frame.  A rectangular margin is added before applying exactly the same
        RandomResizedCrop RNG state as the RGB conditions.
        """
        if not self.pose_keypoint_paths:
            return None, None
        path = self.pose_keypoint_paths.get(text_key)
        if path is None:
            if self.pose_keypoint_strict:
                raise FileNotFoundError(f"No DWPose keypoint cache for {text_key}")
            empty = torch.zeros((1, len(frame_ids), *self.frame_size))
            return empty, empty.clone()

        with np.load(path, allow_pickle=False) as data:
            keypoints = np.asarray(data["keypoints"], dtype=np.float32)
            scores = np.asarray(data["scores"], dtype=np.float32)
            offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
            height, width = map(int, np.asarray(data["inference_size"]).tolist())
            side_masks = {"left": [], "right": []}
            for frame_id in frame_ids:
                start, end = int(offsets[int(frame_id)]), int(offsets[int(frame_id) + 1])
                people_xy = keypoints[start:end]
                people_score = scores[start:end]
                masks = {}
                if len(people_xy):
                    person_index = int(np.nanmean(people_score[:, :18], axis=1).argmax())
                else:
                    person_index = None
                for side, joint_slice in (("left", slice(92, 113)), ("right", slice(113, 134))):
                    mask = np.zeros((height, width), dtype=np.uint8)
                    if person_index is not None:
                        xy = people_xy[person_index, joint_slice]
                        confidence = people_score[person_index, joint_slice]
                        valid = (
                            np.isfinite(xy).all(axis=1)
                            & np.isfinite(confidence)
                            & (confidence >= self.hand_roi_score_threshold)
                            & (xy[:, 0] >= 0.0) & (xy[:, 0] <= 1.0)
                            & (xy[:, 1] >= 0.0) & (xy[:, 1] <= 1.0)
                        )
                        if valid.any():
                            points = xy[valid]
                            x0, y0 = points.min(axis=0)
                            x1, y1 = points.max(axis=0)
                            # Ensure a usable ROI even when only a few joints survive.
                            span_x = max(float(x1 - x0), 0.06)
                            span_y = max(float(y1 - y0), 0.06)
                            pad_x = self.hand_roi_margin * span_x
                            pad_y = self.hand_roi_margin * span_y
                            ix0 = max(0, int(np.floor((x0 - pad_x) * width)))
                            ix1 = min(width, int(np.ceil((x1 + pad_x) * width)))
                            iy0 = max(0, int(np.floor((y0 - pad_y) * height)))
                            iy1 = min(height, int(np.ceil((y1 + pad_y) * height)))
                            if ix1 > ix0 and iy1 > iy0:
                                mask[iy0:iy1, ix0:ix1] = 1
                    masks[side] = self._transform_mask(mask, rng_state)
                for side in side_masks:
                    side_masks[side].append(masks[side])
        return (
            torch.stack(side_masks["left"]).unsqueeze(0),
            torch.stack(side_masks["right"]).unsqueeze(0),
        )

    def __getitem__(self, index):
        (
            main_path,
            key,
            hamer_path,
            hamer_key,
            hand_mask_path,
            hand_mask_key,
            length,
            text_key,
        ) = self.items[index]
        main = self._handle(main_path)
        hamer = self._handle(hamer_path)

        if self.full_video:
            frame_ids = np.arange(length, dtype=int)
        else:
            clip_length = (self.num_frames - 1) * self.sample_rate + 1
            max_start = length - clip_length
            if self.centered_clip:
                middle = length // 2
                centered_start = middle - (clip_length // 2)
                start = min(max(centered_start, 0), max(max_start, 0))
            else:
                start = random.randint(0, max_start) if self.random_sample else 0
            frame_ids = np.linspace(
                start, start + clip_length - 1, self.num_frames, dtype=int
            )

        if self.reference_mode == "first":
            ref_id = 0
        else:
            forbidden_start = max(0, int(frame_ids.min()) - self.ref_margin)
            forbidden_end = min(length, int(frame_ids.max()) + self.ref_margin + 1)
            reference_ids = list(range(0, forbidden_start)) + list(
                range(forbidden_end, length)
            )
            if not reference_ids:
                reference_ids = [
                    i for i in range(length) if i not in set(frame_ids.tolist())
                ]
            ref_id = (
                random.choice(reference_ids) if reference_ids else int(frame_ids[0])
            )

        video = main["video"][key]
        pose = main["pose_video"][key]
        hamer_video = hamer["hamer_video"][hamer_key]
        text_embedding = torch.as_tensor(
            np.asarray(main["text"][key][()]), dtype=torch.float32
        ).squeeze(0)

        rng_state = torch.get_rng_state()
        tgt_sk_frames = torch.stack(
            [self._transform(pose[i], rng_state) for i in frame_ids], dim=1
        )
        tgt_hamer_frames = torch.stack(
            [self._transform(hamer_video[i], rng_state) for i in frame_ids], dim=1
        )
        if self.conditions_only:
            return {
                "tgt_sk_frames": tgt_sk_frames,
                "tgt_hamer_frames": tgt_hamer_frames,
                "tgt_frame_ids": torch.as_tensor(frame_ids, dtype=torch.long),
                "vid_len": length,
                "text_key": text_key,
                "text_embedding": text_embedding,
                "path": text_key + ".pkl",
            }

        ref_image = Image.fromarray(np.asarray(video[ref_id], dtype=np.uint8))
        clip_image = self.clip_image_processor(images=ref_image, return_tensors="pt").pixel_values[0]

        ref_frame = self._transform(video[ref_id], rng_state)
        tgt_frames = torch.stack([self._transform(video[i], rng_state) for i in frame_ids], dim=1)
        hand_masks, face_masks = self._detail_masks(
            text_key, frame_ids, rng_state
        )
        if hand_masks is None and hand_mask_path is not None:
            hand_mask_video = self._handle(hand_mask_path)["hand_mask"][hand_mask_key]
            hand_masks = torch.stack(
                [self._transform_mask(hand_mask_video[i], rng_state) for i in frame_ids]
            ).unsqueeze(0)
        if face_masks is None:
            face_masks = self._fused_face_masks(text_key, frame_ids, rng_state)
        left_hand_roi, right_hand_roi = self._hand_roi_masks(
            text_key, frame_ids, rng_state
        )

        # The rendered-pose HDF5 does not contain DWPose confidence scores.
        # A full-valid mask preserves the conditions; C5.1 keeps mask_ratio=0.
        tgt_hand_masks = torch.ones((len(frame_ids), *self.frame_size), dtype=torch.float32)
        if self.mask_ratio and random.random() < self.mask_ratio:
            mask = tgt_hand_masks.unsqueeze(0)
            tgt_sk_frames = tgt_sk_frames * mask + (1 - mask) * -1
            tgt_hamer_frames = tgt_hamer_frames * mask + (1 - mask) * -1

        for frame_idx in range(len(frame_ids)):
            draw = random.random()
            if draw < self.both_mask_ratio:
                tgt_sk_frames[:, frame_idx].fill_(-1)
                tgt_hamer_frames[:, frame_idx].fill_(-1)
            elif draw < self.both_mask_ratio + self.sk_mask_ratio:
                tgt_sk_frames[:, frame_idx].fill_(-1)
            elif draw < self.both_mask_ratio + self.sk_mask_ratio + self.hamer_mask_ratio:
                tgt_hamer_frames[:, frame_idx].fill_(-1)

        result = {
            "clip_image": clip_image,
            "ref_frame": ref_frame,
            "tgt_frames": tgt_frames,
            "tgt_sk_frames": tgt_sk_frames,
            "tgt_hamer_frames": tgt_hamer_frames,
            "tgt_hand_masks": tgt_hand_masks,
            "tgt_frame_ids": torch.as_tensor(frame_ids, dtype=torch.long),
            "reference_frame_id": int(ref_id),
            "vid_len": length,
            "text_key": text_key,
            "text_embedding": text_embedding,
        }
        if hand_masks is not None:
            result["hand_mask"] = hand_masks
        if face_masks is not None:
            result["face_mask"] = face_masks
        if left_hand_roi is not None:
            result["left_hand_roi_mask"] = left_hand_roi
            result["right_hand_roi_mask"] = right_hand_roi
        return result
