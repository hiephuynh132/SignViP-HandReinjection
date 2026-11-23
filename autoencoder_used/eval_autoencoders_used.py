#!/usr/bin/env python3
"""Evaluate reconstruction quality for first-stage autoencoders used by exp runs."""

import argparse
import csv
import itertools
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw

try:
    import h5py
except Exception:  # pragma: no cover
    h5py = None

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover
    skimage_ssim = None


REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path):
    if path is None:
        return None
    path = Path(os.path.expanduser(str(path)))
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path


def install_dummy_lightning_checkpoint_class():
    lightning = sys.modules.setdefault("pytorch_lightning", types.ModuleType("pytorch_lightning"))
    callbacks = sys.modules.setdefault(
        "pytorch_lightning.callbacks", types.ModuleType("pytorch_lightning.callbacks")
    )
    model_checkpoint = sys.modules.setdefault(
        "pytorch_lightning.callbacks.model_checkpoint",
        types.ModuleType("pytorch_lightning.callbacks.model_checkpoint"),
    )

    class ModelCheckpoint:
        pass

    lightning.callbacks = callbacks
    callbacks.model_checkpoint = model_checkpoint
    model_checkpoint.ModelCheckpoint = ModelCheckpoint
    torch.serialization.add_safe_globals([ModelCheckpoint])


def load_checkpoint(path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception:
        install_dummy_lightning_checkpoint_class()
        try:
            return torch.load(path, map_location="cpu")
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)


class TamingVQGANFirstStage(nn.Module):
    def __init__(self, spec):
        super().__init__()
        taming_root = REPO_ROOT / "taming-transformers"
        if str(taming_root) not in sys.path:
            sys.path.insert(0, str(taming_root))

        from taming.modules.diffusionmodules.model import Decoder, Encoder
        from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer

        image_size = int(spec.get("image_size", 128))
        latent_channels = int(spec.get("latent_channels", 4))
        downsample = int(spec.get("latent_downsample", 8))
        ch_mult = list(spec.get("ch_mult") or {
            4: [1, 2, 4],
            8: [1, 2, 2, 4],
            16: [1, 1, 2, 2, 4],
            32: [1, 1, 2, 2, 4, 4],
        }[downsample])
        attn_resolutions = (
            list(spec["attn_resolutions"])
            if "attn_resolutions" in spec
            else [image_size // downsample]
        )

        ddconfig = {
            "double_z": False,
            "z_channels": latent_channels,
            "resolution": image_size,
            "in_channels": 3,
            "out_ch": 3,
            "ch": 128,
            "ch_mult": ch_mult,
            "num_res_blocks": 2,
            "attn_resolutions": attn_resolutions,
            "dropout": 0.0,
        }
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.quantize = VectorQuantizer(int(spec.get("n_embed", 16384)), latent_channels, beta=0.25)
        self.quant_conv = nn.Conv2d(ddconfig["z_channels"], latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, ddconfig["z_channels"], 1)

        ckpt = load_checkpoint(resolve_path(spec["checkpoint"]))
        state = ckpt.get("state_dict", ckpt)
        missing, unexpected = self.load_state_dict(state, strict=False)
        unexpected_real = [
            key for key in unexpected
            if not key.startswith("loss.") and not key.startswith("discriminator.")
        ]
        if missing or unexpected_real:
            raise RuntimeError(
                f"Could not load VQGAN cleanly: missing={missing[:8]}, "
                f"unexpected={unexpected_real[:8]}"
            )

    def forward(self, x):
        h = self.quant_conv(self.encoder(x))
        quant, _, _ = self.quantize(h)
        return self.decoder(self.post_quant_conv(quant)), quant


def kl_ddconfig(spec):
    z_channels = int(spec.get("latent_channels", 4))
    downsample = int(spec.get("latent_downsample", 8))
    image_size = int(spec.get("image_size", 128))
    ch_mult = list(spec.get("ch_mult") or {
        4: [1, 2, 4],
        8: [1, 2, 2, 4],
        16: [1, 1, 2, 2, 4],
        32: [1, 1, 2, 2, 4, 4],
    }.get(downsample, [1, 2, 2, 4]))
    attn_resolutions = list(spec.get("attn_resolutions") or {
        4: [],
        8: [image_size // downsample],
        16: [image_size // downsample],
        32: [16, 8],
    }.get(downsample, []))
    return {
        "double_z": True,
        "z_channels": z_channels,
        "resolution": image_size,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": ch_mult,
        "num_res_blocks": 2,
        "attn_resolutions": attn_resolutions,
        "dropout": 0.0,
    }


class AutoencoderKLFirstStage(nn.Module):
    def __init__(self, spec):
        super().__init__()
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from models.autoencoder import AutoencoderKL

        self.model = AutoencoderKL(
            kl_ddconfig(spec),
            int(spec.get("latent_channels", 4)),
            ckpt_path=str(resolve_path(spec["checkpoint"])),
            partial_init=bool(spec.get("partial_init", False)),
        )

    def forward(self, x):
        posterior = self.model.encode(x)
        z = posterior.mode()
        return self.model.decode(z), z


class DiffusersTinyVAEFirstStage(nn.Module):
    """Diffusers AutoencoderTiny wrapper used for TAESD/TAESD3 checks."""

    def __init__(self, spec):
        super().__init__()
        try:
            # Match the compatibility shims used by runners/ncsn_runner.py.
            import transformers
            if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
                transformers.EncoderDecoderCache = transformers.DynamicCache
            if not hasattr(transformers, "Dinov2WithRegistersConfig"):
                class _Dinov2WithRegistersConfig:
                    pass
                transformers.Dinov2WithRegistersConfig = _Dinov2WithRegistersConfig
            if not hasattr(transformers, "Dinov2WithRegistersModel"):
                class _Dinov2WithRegistersModel:
                    pass
                transformers.Dinov2WithRegistersModel = _Dinov2WithRegistersModel
            import diffusers.utils.import_utils as diffusers_import_utils
            diffusers_import_utils._xformers_available = False
            diffusers_import_utils._xformers_version = None
            from diffusers import AutoencoderTiny
        except Exception as exc:
            raise ImportError(
                "backend=diffusers_tiny_vae requires diffusers/transformers. "
                "Use the same environment as training or set TAESD3_MODEL to a local cache."
            ) from exc

        checkpoint = str(spec["checkpoint"])
        if spec.get("name") == "taesd3" and os.environ.get("TAESD3_MODEL"):
            checkpoint = os.environ["TAESD3_MODEL"]
        dtype = None
        dtype_name = str(spec.get("torch_dtype", "")).lower()
        if dtype_name in {"float16", "fp16", "half"}:
            dtype = torch.float16
        elif dtype_name in {"bfloat16", "bf16"}:
            dtype = torch.bfloat16
        elif dtype_name in {"float32", "fp32"}:
            dtype = torch.float32
        elif dtype_name:
            raise ValueError(f"Unsupported torch_dtype for {spec['name']}: {dtype_name}")

        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoencoderTiny.from_pretrained(checkpoint, **kwargs)
        self.model.eval()

    def encode(self, x):
        encoded = self.model.encode(x)
        if hasattr(encoded, "latents"):
            return encoded.latents
        if isinstance(encoded, (tuple, list)) and encoded:
            first = encoded[0]
            if hasattr(first, "latents"):
                return first.latents
            return first
        raise RuntimeError(f"Unexpected AutoencoderTiny encode output type: {type(encoded)}")

    def decode(self, latents):
        decoded = self.model.decode(latents)
        if hasattr(decoded, "sample"):
            return decoded.sample
        if isinstance(decoded, (tuple, list)) and decoded:
            return decoded[0]
        return decoded

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class DiffusersAutoencoderKLFirstStage(nn.Module):
    """Wrapper for the pretrained SD1.x ``diffusers.AutoencoderKL`` VAE."""

    def __init__(self, spec):
        super().__init__()
        try:
            # Keep this evaluator compatible with the pinned C3 transformers
            # and diffusers overlays, matching the training environment.
            import transformers
            if not hasattr(transformers, "EncoderDecoderCache") and hasattr(transformers, "DynamicCache"):
                transformers.EncoderDecoderCache = transformers.DynamicCache
            if not hasattr(transformers, "Dinov2WithRegistersConfig"):
                class _Dinov2WithRegistersConfig:
                    pass
                transformers.Dinov2WithRegistersConfig = _Dinov2WithRegistersConfig
            if not hasattr(transformers, "Dinov2WithRegistersModel"):
                class _Dinov2WithRegistersModel:
                    pass
                transformers.Dinov2WithRegistersModel = _Dinov2WithRegistersModel
            import diffusers.utils.import_utils as diffusers_import_utils
            diffusers_import_utils._xformers_available = False
            diffusers_import_utils._xformers_version = None
            from diffusers import AutoencoderKL
        except Exception as exc:
            raise ImportError(
                "backend=diffusers_autoencoder_kl requires the C3 "
                "diffusers/transformers environment"
            ) from exc

        checkpoint = resolve_path(spec["checkpoint"])
        kwargs = {"local_files_only": True}
        subfolder = str(spec.get("subfolder", "") or "").strip()
        if subfolder:
            kwargs["subfolder"] = subfolder
        self.model = AutoencoderKL.from_pretrained(str(checkpoint), **kwargs)
        self.model.eval()

    def forward(self, x):
        # Posterior mode gives a deterministic reconstruction upper bound.  Do
        # not apply scaling_factor here: SD applies it only at the diffusion
        # boundary, and encode->decode cancels that scaling exactly.
        posterior = self.model.encode(x).latent_dist
        z = posterior.mode()
        return self.model.decode(z).sample, z


def build_model(spec, device):
    backend = str(spec["backend"]).lower()
    if backend == "taming_vqgan":
        model = TamingVQGANFirstStage(spec)
    elif backend == "autoencoder_kl":
        model = AutoencoderKLFirstStage(spec)
    elif backend in {"diffusers_tiny_vae", "taesd", "taesd3"}:
        model = DiffusersTinyVAEFirstStage(spec)
    elif backend in {"diffusers_autoencoder_kl", "sd_vae", "sd15_vae"}:
        model = DiffusersAutoencoderKLFirstStage(spec)
    else:
        raise ValueError(f"Unknown backend: {backend}")
    model.eval().requires_grad_(False)
    return model.to(device)


def load_image(path, image_size):
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return resize_rgb_tensor(tensor, image_size)


def resize_rgb_tensor(tensor, image_size):
    """Match C2.2's torch bicubic RGB preprocessing exactly."""
    if tuple(tensor.shape[-2:]) == (image_size, image_size):
        return tensor
    return F.interpolate(
        tensor.unsqueeze(0),
        size=(image_size, image_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0).clamp(0.0, 1.0)


def image_paths(input_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(path for path in Path(input_dir).rglob("*") if path.suffix.lower() in exts)


def hdf5_paths(path):
    path = Path(path)
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.hdf5")) + sorted(path.rglob("*.h5"))


def is_dense_dataset(node):
    return node is not None and hasattr(node, "shape") and hasattr(node, "dtype")


def reached_limit(records, max_samples):
    return max_samples > 0 and len(records) >= max_samples


def detail_mask_suffix(image_size):
    """Return the Grounded-SAM2 cache suffix matching evaluation resolution."""
    return f"_grounded_sam2_face_hands_384_to{int(image_size)}_v4.npz"


def decode_hdf5_text(value):
    value = np.asarray(value)
    if value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)


def read_hdf5_text_key(handle, video_key):
    text_keys = handle.get("text_key")
    if text_keys is None:
        return None
    if is_dense_dataset(text_keys):
        return decode_hdf5_text(text_keys[int(video_key)])
    node = text_keys.get(str(video_key))
    return decode_hdf5_text(node[()]) if node is not None else None


def load_external_detail_masks(mask_path, image_size):
    with np.load(mask_path, allow_pickle=False) as masks:
        required = {"hands", "face", "frame_indices"}
        missing = required.difference(masks.files)
        if missing:
            raise KeyError(f"Mask file {mask_path} is missing arrays: {sorted(missing)}")
        hands = np.asarray(masks["hands"], dtype=np.bool_)
        face = np.asarray(masks["face"], dtype=np.bool_)
        frame_indices = np.asarray(masks["frame_indices"], dtype=np.int64)

    if hands.shape != face.shape or hands.ndim != 3:
        raise ValueError(
            f"Invalid hand/face mask shapes in {mask_path}: "
            f"hands={hands.shape}, face={face.shape}"
        )
    if frame_indices.shape != (hands.shape[0],):
        raise ValueError(
            f"Invalid frame_indices shape in {mask_path}: {frame_indices.shape} "
            f"for masks {hands.shape}"
        )
    hand_tensor = torch.from_numpy(hands).unsqueeze(1)
    face_tensor = torch.from_numpy(face).unsqueeze(1)
    if tuple(hand_tensor.shape[-2:]) != (image_size, image_size):
        hand_tensor = F.interpolate(
            hand_tensor.float(), size=(image_size, image_size), mode="nearest"
        ).bool()
        face_tensor = F.interpolate(
            face_tensor.float(), size=(image_size, image_size), mode="nearest"
        ).bool()
    positions = {int(frame_id): index for index, frame_id in enumerate(frame_indices)}
    return hand_tensor.squeeze(1).bool(), face_tensor.squeeze(1).bool(), positions


def iter_hdf5_frames(
    path,
    image_size,
    frames_per_video,
    max_samples,
    detail_mask_cache=None,
    detail_mask_split=None,
    detail_mask_strict=False,
    load_pose=True,
):
    if h5py is None:
        raise RuntimeError("h5py is required for --hdf5 input")
    mask_dir = None
    mask_paths = {}
    if detail_mask_cache:
        mask_suffix = detail_mask_suffix(image_size)
        mask_dir = resolve_path(detail_mask_cache)
        if detail_mask_split:
            mask_dir = mask_dir / detail_mask_split
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Detail-mask directory does not exist: {mask_dir}")
        mask_paths = {
            item.name[:-len(mask_suffix)]: item
            for item in mask_dir.glob(f"*{mask_suffix}")
        }
        if not mask_paths:
            raise FileNotFoundError(f"No Grounded-SAM2 v4 masks found in {mask_dir}")
    yielded = 0
    for shard in hdf5_paths(path):
        with h5py.File(shard, "r") as handle:
            if "video" not in handle:
                continue
            for video_key in sorted(handle["video"].keys(), key=lambda value: int(value)):
                video_group = handle["video"][video_key]
                pose_group = handle["pose_video"].get(video_key) if "pose_video" in handle else None
                sample_key = read_hdf5_text_key(handle, video_key) if mask_dir else None
                external_masks = None
                if mask_dir:
                    mask_path = mask_paths.get(sample_key)
                    if mask_path is None:
                        if detail_mask_strict:
                            raise FileNotFoundError(
                                f"No Grounded-SAM2 v4 mask for sample {sample_key!r} in {mask_dir}"
                            )
                    else:
                        external_masks = load_external_detail_masks(mask_path, image_size)
                if is_dense_dataset(video_group):
                    frame_keys = range(min(len(video_group), frames_per_video))
                else:
                    frame_keys = sorted(
                        video_group.keys(), key=lambda value: int(value)
                    )[:frames_per_video]
                for frame_key in frame_keys:
                    frame_idx = int(frame_key)
                    raw = video_group[frame_key]
                    image = Image.fromarray(np.asarray(raw).astype(np.uint8)).convert("RGB")
                    arr = np.asarray(image).astype(np.float32) / 255.0
                    image_tensor = resize_rgb_tensor(
                        torch.from_numpy(arr).permute(2, 0, 1), image_size
                    )
                    pose_tensor = None
                    if load_pose and pose_group is not None:
                        try:
                            pose_raw = pose_group[frame_idx] if is_dense_dataset(pose_group) else pose_group[str(frame_idx)][()]
                            pose_img = Image.fromarray(np.asarray(pose_raw).astype(np.uint8)).convert("RGB")
                            pose_img = pose_img.resize((image_size, image_size), Image.NEAREST)
                            pose_arr = np.asarray(pose_img).astype(np.float32) / 255.0
                            pose_tensor = torch.from_numpy(pose_arr).permute(2, 0, 1)
                        except Exception:
                            pose_tensor = None
                    hand_mask = None
                    face_mask = None
                    if external_masks is not None:
                        hands, faces, positions = external_masks
                        position = positions.get(int(frame_idx))
                        if position is None:
                            if detail_mask_strict:
                                raise KeyError(
                                    f"Mask for sample {sample_key!r} has no frame_index={frame_idx}"
                                )
                        else:
                            hand_mask = hands[position]
                            face_mask = faces[position]
                    yield {
                        "image": image_tensor,
                        "pose": pose_tensor,
                        "hand_mask": hand_mask,
                        "face_mask": face_mask,
                    }
                    yielded += 1
                    if max_samples > 0 and yielded >= max_samples:
                        return


def read_hdf5_frames(
    path,
    image_size,
    frames_per_video,
    max_samples,
    detail_mask_cache=None,
    detail_mask_split=None,
    detail_mask_strict=False,
):
    """Materialize HDF5 records for compatibility with callers and tests.

    Full evaluation uses :func:`iter_hdf5_frames` through ``iter_inputs`` so
    frames are released after each inference batch instead of retaining the
    complete split in RAM.
    """
    return list(
        iter_hdf5_frames(
            path,
            image_size,
            frames_per_video,
            max_samples,
            detail_mask_cache=detail_mask_cache,
            detail_mask_split=detail_mask_split,
            detail_mask_strict=detail_mask_strict,
            load_pose=True,
        )
    )


def iter_inputs(args, image_size):
    if args.input_dir:
        paths = image_paths(args.input_dir)
        if args.max_samples > 0:
            paths = paths[:args.max_samples]
        return (
            {"image": load_image(path, image_size), "pose": None}
            for path in paths
        )
    if args.hdf5:
        # Pose is only a fallback source for ROI masks. Strict external masks
        # make pose loading redundant and would otherwise double CPU RAM.
        load_pose = not (args.detail_mask_cache and args.detail_mask_strict)
        return iter_hdf5_frames(
            args.hdf5,
            image_size,
            args.frames_per_video,
            args.max_samples,
            detail_mask_cache=args.detail_mask_cache,
            detail_mask_split=args.detail_mask_split,
            detail_mask_strict=args.detail_mask_strict,
            load_pose=load_pose,
        )
    raise ValueError("Provide --input-dir or --hdf5")


def iter_batches(records, batch_size):
    """Yield bounded record lists without materializing the input iterator."""
    iterator = iter(records)
    while True:
        batch = list(itertools.islice(iterator, batch_size))
        if not batch:
            return
        yield batch


def to_model_space(x01):
    return x01 * 2.0 - 1.0


def to_rgb_space(x_model):
    return ((x_model.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


def compute_ssim_batch(x, y):
    if skimage_ssim is not None:
        values = []
        x_np = x.detach().cpu().permute(0, 2, 3, 1).numpy()
        y_np = y.detach().cpu().permute(0, 2, 3, 1).numpy()
        for a, b in zip(x_np, y_np):
            if a.shape[-1] == 1:
                a_gray = a[..., 0]
                b_gray = b[..., 0]
            else:
                a_gray = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
                b_gray = 0.299 * b[..., 0] + 0.587 * b[..., 1] + 0.114 * b[..., 2]
            values.append(
                skimage_ssim(
                    np.clip(a_gray * 255.0, 0, 255).astype(np.uint8),
                    np.clip(b_gray * 255.0, 0, 255).astype(np.uint8),
                    data_range=255,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
            )
        return float(np.mean(values))

    # Lightweight fallback: global SSIM over each image.
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    dims = (1, 2, 3)
    mux = x.mean(dim=dims)
    muy = y.mean(dim=dims)
    sigx = ((x - mux[:, None, None, None]) ** 2).mean(dim=dims)
    sigy = ((y - muy[:, None, None, None]) ** 2).mean(dim=dims)
    sigxy = ((x - mux[:, None, None, None]) * (y - muy[:, None, None, None])).mean(dim=dims)
    ssim = ((2 * mux * muy + c1) * (2 * sigxy + c2)) / (
        (mux ** 2 + muy ** 2 + c1) * (sigx + sigy + c2)
    )
    return float(ssim.mean().item())


def compute_ssim_single_channel_batch(x, y):
    if skimage_ssim is not None:
        values = []
        x_np = x.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        for a, b in zip(x_np, y_np):
            values.append(
                skimage_ssim(
                    np.clip(a * 255.0, 0, 255).astype(np.uint8),
                    np.clip(b * 255.0, 0, 255).astype(np.uint8),
                    data_range=255,
                    gaussian_weights=True,
                    use_sample_covariance=False,
                )
            )
        return float(np.mean(values))
    return math.nan


def _dilate_np(mask, radius):
    radius = int(radius or 0)
    if radius <= 0:
        return mask.astype(bool)
    try:
        import cv2

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
        return cv2.dilate(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    except Exception:
        return mask.astype(bool)


def _bbox_mask_from_seed(seed, padding, min_pixels=1):
    seed = seed.astype(bool)
    out = np.zeros_like(seed, dtype=bool)
    if int(seed.sum()) < min_pixels:
        return out
    ys, xs = np.where(seed)
    height, width = seed.shape
    padding = int(padding or 0)
    y0 = max(int(ys.min()) - padding, 0)
    y1 = min(int(ys.max()) + padding + 1, height)
    x0 = max(int(xs.min()) - padding, 0)
    x1 = min(int(xs.max()) + padding + 1, width)
    out[y0:y1, x0:x1] = True
    return out


def build_roi_masks_from_pose(pose_chw, args):
    if pose_chw is None:
        return None, None
    pose = pose_chw.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
    red, green, blue = pose[..., 0], pose[..., 1], pose[..., 2]
    max_ch = pose.max(axis=-1)
    min_ch = pose.min(axis=-1)
    pose_fg = max_ch > float(args.roi_pose_threshold)

    face_seed = (
        (red >= float(args.face_white_threshold))
        & (green >= float(args.face_white_threshold))
        & (blue >= float(args.face_white_threshold))
        & ((max_ch - min_ch) <= float(args.face_chroma_tolerance))
    )
    face_seed = _dilate_np(face_seed, args.face_dilation)
    face_mask = _bbox_mask_from_seed(face_seed, args.face_padding, min_pixels=args.face_min_pixels) & pose_fg

    hand_seed = (
        (blue >= float(args.hand_blue_threshold))
        & (red <= float(args.hand_other_max))
        & (green <= float(args.hand_other_max))
    )
    hand_seed = _dilate_np(hand_seed, args.hand_seed_dilation)
    hand_seed = _dilate_np(hand_seed, args.hand_final_dilation)
    hand_mask = _bbox_mask_from_seed(hand_seed, args.hand_padding, min_pixels=args.hand_min_pixels) & pose_fg & ~face_mask
    return torch.from_numpy(hand_mask), torch.from_numpy(face_mask)


def masked_mse_per_sample(recon, target, masks):
    if masks is None or not masks:
        return None
    masks = torch.stack(masks).to(device=recon.device, dtype=recon.dtype).unsqueeze(1)
    denom = (masks.sum(dim=(1, 2, 3)) * recon.shape[1]).clamp_min(1.0)
    return (((recon - target).pow(2) * masks).sum(dim=(1, 2, 3)) / denom).detach().cpu()


def crop_roi_batch(images, masks, pad_value=0.0):
    crops = []
    if masks is None or not masks:
        return None
    for image, mask in zip(images.detach().cpu(), masks):
        mask = mask.detach().cpu().bool()
        if not bool(mask.any()):
            continue
        ys, xs = torch.where(mask)
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        crop = image[:, y0:y1, x0:x1]
        if crop.numel() == 0:
            continue
        crops.append(F.interpolate(crop.unsqueeze(0), size=(64, 64), mode="bilinear", align_corners=False)[0])
    if not crops:
        return None
    return torch.stack(crops).clamp(0.0, 1.0)


def build_lpips_metric(device, enabled=True):
    if not enabled:
        return None
    try:
        # Use the official implementation.  The legacy local wrapper can emit
        # small negative distances, which are invalid for final comparisons.
        import lpips

        metric = lpips.LPIPS(net="alex", version="0.1").to(device)
        metric.eval().requires_grad_(False)
        return metric
    except Exception as exc:
        print(f"[warn] LPIPS unavailable, writing NaN: {type(exc).__name__}: {exc}")
        return None


def compute_lpips_batch(lpips_metric, target, recon):
    if lpips_metric is None:
        return math.nan
    with torch.no_grad():
        value = lpips_metric(target, recon, normalize=True)
    return float(value.detach().mean().item())


def append_finite(values, value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return
    if math.isfinite(value):
        values.append(value)


def save_grid(original, recon, path, max_items=8):
    count = min(max_items, original.shape[0])
    cell_h, cell_w = original.shape[2], original.shape[3]
    canvas = Image.new("RGB", (cell_w * count, cell_h * 3))
    for idx in range(count):
        error = (recon[idx] - original[idx]).abs().mean(dim=0, keepdim=True)
        if error.max().item() > 0:
            error = error / error.max()
        for row, tensor in enumerate((original[idx], recon[idx], error.repeat(3, 1, 1))):
            arr = (tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
            image = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB")
            canvas.paste(image, (idx * cell_w, row * cell_h))
    canvas.save(path)


def error_heatmap(error, max_error=0.02):
    """Map absolute RGB error to a fixed blue->cyan->yellow->red scale."""
    value = (error / float(max_error)).clamp(0.0, 1.0)
    # Piecewise-linear "jet" map: zero stays dark blue and saturated errors
    # become red. A fixed scale keeps different frames directly comparable.
    red = (1.5 - (4.0 * value - 3.0).abs()).clamp(0.0, 1.0)
    green = (1.5 - (4.0 * value - 2.0).abs()).clamp(0.0, 1.0)
    blue = (1.5 - (4.0 * value - 1.0).abs()).clamp(0.0, 1.0)
    return torch.stack((red, green, blue), dim=0)


def save_error_heatmap_grid(original, recon, path, max_items=8, max_error=0.02):
    count = min(max_items, original.shape[0])
    cell_h, cell_w = original.shape[2], original.shape[3]
    legend_h = 28
    canvas = Image.new("RGB", (cell_w * count, cell_h + legend_h), "white")
    for idx in range(count):
        error = (recon[idx] - original[idx]).abs().mean(dim=0)
        heatmap = error_heatmap(error, max_error=max_error)
        arr = (heatmap.detach().cpu().permute(1, 2, 0).numpy() * 255.0).round()
        canvas.paste(
            Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGB"),
            (idx * cell_w, 0),
        )

    legend_width = min(320, canvas.width - 20)
    legend_x = 10
    legend_y = cell_h + 5
    values = torch.linspace(0.0, max_error, legend_width)
    legend = error_heatmap(values, max_error=max_error)
    legend_arr = (legend.permute(1, 0).unsqueeze(0).numpy() * 255.0).round()
    legend_image = Image.fromarray(
        legend_arr.clip(0, 255).astype(np.uint8), "RGB"
    ).resize((legend_width, 10))
    canvas.paste(legend_image, (legend_x, legend_y))
    draw = ImageDraw.Draw(canvas)
    draw.text((legend_x, legend_y + 11), "0", fill="black")
    draw.text(
        (legend_x + legend_width - 58, legend_y + 11),
        f">={max_error:.3f}",
        fill="black",
    )
    draw.text(
        (legend_x + legend_width + 8, legend_y),
        "mean |RGB error|",
        fill="black",
    )
    canvas.save(path)


def evaluate_one(spec, frames, args, device, output_dir):
    backend = str(spec["backend"]).lower()
    checkpoint_value = str(spec["checkpoint"])
    if spec.get("name") == "taesd3" and os.environ.get("TAESD3_MODEL"):
        checkpoint_value = os.environ["TAESD3_MODEL"]

    if backend in {
        "diffusers_tiny_vae", "taesd", "taesd3",
        "diffusers_autoencoder_kl", "sd_vae", "sd15_vae",
    }:
        checkpoint_display = checkpoint_value
        checkpoint_path = Path(os.path.expanduser(checkpoint_value))
        looks_like_local_path = (
            checkpoint_path.is_absolute()
            or checkpoint_value.startswith(".")
            or checkpoint_value.startswith("~")
        )
        if looks_like_local_path and not checkpoint_path.exists():
            if spec.get("optional", False) or args.skip_missing:
                print(f"[skip] {spec['name']}: missing checkpoint/model dir {checkpoint_path}")
                return None
            raise FileNotFoundError(f"Missing checkpoint/model dir for {spec['name']}: {checkpoint_path}")
    else:
        checkpoint = resolve_path(checkpoint_value)
        checkpoint_display = str(checkpoint)
        if not checkpoint.exists():
            if spec.get("optional", False) or args.skip_missing:
                print(f"[skip] {spec['name']}: missing checkpoint {checkpoint}")
                return None
            raise FileNotFoundError(f"Missing checkpoint for {spec['name']}: {checkpoint}")

    image_size = int(spec.get("image_size", 128))
    if frames is None:
        records = iter_inputs(args, image_size)
    else:
        records = iter(frames)

    model = build_model(spec, device)
    lpips_metric = build_lpips_metric(device, enabled=args.lpips)
    l1_sum = 0.0
    mse_sum = 0.0
    psnr_sum = 0.0
    channel_mse_sum = torch.zeros(3, dtype=torch.float64)
    channel_psnr_sum = torch.zeros(3, dtype=torch.float64)
    channel_ssim_values = [[], [], []]
    ssim_values = []
    lpips_values = []
    hand_mse_values = []
    hand_psnr_values = []
    hand_ssim_values = []
    hand_lpips_values = []
    face_mse_values = []
    face_psnr_values = []
    face_ssim_values = []
    face_lpips_values = []
    hand_frames = 0
    face_frames = 0
    total = 0
    latent_shape = None
    first_original = None
    first_recon = None

    frame_count = 0
    for batch_records in iter_batches(records, args.batch_size):
        batch = torch.stack([record["image"] for record in batch_records]).to(device)
        pose_items = [record.get("pose") for record in batch_records]
        x_model = to_model_space(batch)
        with torch.no_grad():
            recon_model, latents = model(x_model)
        recon = to_rgb_space(recon_model)
        target = batch.clamp(0.0, 1.0)
        if latent_shape is None:
            latent_shape = "x".join(str(v) for v in latents.shape[1:])
        l1_sum += F.l1_loss(recon, target, reduction="sum").item()
        mse_sum += F.mse_loss(recon, target, reduction="sum").item()
        frame_mse = (recon - target).pow(2).mean(dim=(1, 2, 3))
        psnr_sum += float((-10.0 * torch.log10(frame_mse.clamp_min(1e-12))).sum().item())
        ssim_values.append(compute_ssim_batch(target, recon))
        channel_mse = (recon - target).pow(2).mean(dim=(2, 3))
        channel_mse_sum += channel_mse.detach().cpu().double().sum(dim=0)
        channel_psnr_sum += (-10.0 * torch.log10(channel_mse.clamp_min(1e-12))).detach().cpu().double().sum(dim=0)
        for channel_idx in range(min(3, recon.shape[1])):
            append_finite(
                channel_ssim_values[channel_idx],
                compute_ssim_single_channel_batch(target[:, channel_idx], recon[:, channel_idx]),
            )
        lpips_value = compute_lpips_batch(lpips_metric, target, recon)
        if not math.isnan(lpips_value):
            lpips_values.append(lpips_value)

        hand_masks = []
        face_masks = []
        for record, pose in zip(batch_records, pose_items):
            hand_mask = record.get("hand_mask")
            face_mask = record.get("face_mask")
            if hand_mask is None or face_mask is None:
                hand_mask, face_mask = build_roi_masks_from_pose(pose, args)
            hand_masks.append(hand_mask if hand_mask is not None else torch.zeros(target.shape[-2:], dtype=torch.bool))
            face_masks.append(face_mask if face_mask is not None else torch.zeros(target.shape[-2:], dtype=torch.bool))

        for prefix, masks, mse_values, psnr_values, ssim_roi_values, lpips_roi_values in (
            ("hand", hand_masks, hand_mse_values, hand_psnr_values, hand_ssim_values, hand_lpips_values),
            ("face", face_masks, face_mse_values, face_psnr_values, face_ssim_values, face_lpips_values),
        ):
            valid_count = sum(1 for mask in masks if mask is not None and bool(mask.any()))
            if prefix == "hand":
                hand_frames += valid_count
            else:
                face_frames += valid_count
            roi_mse = masked_mse_per_sample(recon, target, masks)
            if roi_mse is not None:
                for value in roi_mse.tolist():
                    if value > 0:
                        mse_values.append(float(value))
                        psnr_values.append(10.0 * math.log10(1.0 / max(float(value), 1e-12)))
            target_crops = crop_roi_batch(target, masks)
            recon_crops = crop_roi_batch(recon, masks)
            if target_crops is not None and recon_crops is not None:
                append_finite(ssim_roi_values, compute_ssim_batch(target_crops, recon_crops))
                append_finite(lpips_roi_values, compute_lpips_batch(lpips_metric, target_crops.to(device), recon_crops.to(device)))
        total += target.numel()
        frame_count += len(batch_records)
        if first_original is None:
            first_original = target.detach().cpu()
            first_recon = recon.detach().cpu()

    if frame_count == 0:
        raise RuntimeError("No input frames found")
    l1 = l1_sum / total
    mse = mse_sum / total
    psnr = psnr_sum / frame_count
    ssim = float(np.mean(ssim_values))
    lpips = float(np.mean(lpips_values)) if lpips_values else math.nan
    channel_mse_avg = channel_mse_sum / max(frame_count, 1)
    channel_psnr_avg = channel_psnr_sum / max(frame_count, 1)

    def mean_or_nan(values):
        return float(np.mean(values)) if values else math.nan

    sample_path = output_dir / "samples" / f"{spec['name']}_grid.png"
    save_grid(first_original, first_recon, sample_path)
    heatmap_path = output_dir / "samples" / f"{spec['name']}_error_heatmap.png"
    save_error_heatmap_grid(first_original, first_recon, heatmap_path)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model": spec["name"],
        "backend": spec["backend"],
        "frames": frame_count,
        "image_size": image_size,
        "checkpoint": checkpoint_display,
        "latent_shape": latent_shape,
        "l1": l1,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "mse_r": float(channel_mse_avg[0]),
        "mse_g": float(channel_mse_avg[1]),
        "mse_b": float(channel_mse_avg[2]),
        "psnr_r": float(channel_psnr_avg[0]),
        "psnr_g": float(channel_psnr_avg[1]),
        "psnr_b": float(channel_psnr_avg[2]),
        "ssim_r": mean_or_nan(channel_ssim_values[0]),
        "ssim_g": mean_or_nan(channel_ssim_values[1]),
        "ssim_b": mean_or_nan(channel_ssim_values[2]),
        "hand_frames": int(hand_frames),
        "mse_hand": mean_or_nan(hand_mse_values),
        "psnr_hand": mean_or_nan(hand_psnr_values),
        "ssim_hand": mean_or_nan(hand_ssim_values),
        "lpips_hand": mean_or_nan(hand_lpips_values),
        "face_frames": int(face_frames),
        "mse_face": mean_or_nan(face_mse_values),
        "psnr_face": mean_or_nan(face_psnr_values),
        "ssim_face": mean_or_nan(face_ssim_values),
        "lpips_face": mean_or_nan(face_lpips_values),
        "sample": str(sample_path),
        "metric_reduction": "mean_frames",
        "ssim_mode": "grayscale_gaussian_data_range_255",
        "psnr_mode": "mean_frame_psnr",
    }


def write_reports(rows, output_dir):
    csv_path = output_dir / "metrics.csv"
    md_path = output_dir / "metrics.md"
    fields = [
        "model",
        "backend",
        "frames",
        "image_size",
        "latent_shape",
        "l1",
        "mse",
        "psnr",
        "ssim",
        "lpips",
        "mse_r",
        "mse_g",
        "mse_b",
        "psnr_r",
        "psnr_g",
        "psnr_b",
        "ssim_r",
        "ssim_g",
        "ssim_b",
        "hand_frames",
        "mse_hand",
        "psnr_hand",
        "ssim_hand",
        "lpips_hand",
        "face_frames",
        "mse_face",
        "psnr_face",
        "ssim_face",
        "lpips_face",
        "checkpoint",
        "sample",
        "metric_reduction",
        "ssim_mode",
        "psnr_mode",
    ]

    def format_float(value, digits=6):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if math.isnan(number):
            return "nan"
        if math.isinf(number):
            return "inf" if number > 0 else "-inf"
        return f"{number:.{digits}f}"

    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    with open(md_path, "w", encoding="utf-8") as handle:
        handle.write("# Autoencoder Reconstruction Metrics\n\n")
        handle.write(
            "Metric protocol: PSNR is averaged per frame/image; SSIM is grayscale "
            "Gaussian with data_range=255.\n\n"
        )
        handle.write("| Model | Backend | Frames | Image | Latent | L1 ↓ | MSE ↓ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | PSNR hand ↑ | SSIM hand ↑ | LPIPS hand ↓ | PSNR face ↑ | SSIM face ↑ | LPIPS face ↓ |\n")
        handle.write("|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['model']}` | `{row['backend']}` | {row['frames']} | {row['image_size']} | "
                f"`{row['latent_shape']}` | {format_float(row['l1'])} | {format_float(row['mse'])} | "
                f"{format_float(row['psnr'])} | {format_float(row['ssim'])} | {format_float(row['lpips'])} | "
                f"{format_float(row['psnr_hand'])} | {format_float(row['ssim_hand'])} | {format_float(row['lpips_hand'])} | "
                f"{format_float(row['psnr_face'])} | {format_float(row['ssim_face'])} | {format_float(row['lpips_face'])} |\n"
            )
        handle.write("\n## Per-Channel Metrics\n\n")
        handle.write("| Model | MSE_R ↓ | MSE_G ↓ | MSE_B ↓ | PSNR_R ↑ | PSNR_G ↑ | PSNR_B ↑ | SSIM_R ↑ | SSIM_G ↑ | SSIM_B ↑ |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['model']}` | {format_float(row['mse_r'])} | {format_float(row['mse_g'])} | "
                f"{format_float(row['mse_b'])} | {format_float(row['psnr_r'])} | {format_float(row['psnr_g'])} | "
                f"{format_float(row['psnr_b'])} | {format_float(row['ssim_r'])} | {format_float(row['ssim_g'])} | "
                f"{format_float(row['ssim_b'])} |\n"
            )
        handle.write("\n## ROI Coverage\n\n")
        handle.write("| Model | Hand ROI frames | Face ROI frames |\n")
        handle.write("|---|---:|---:|\n")
        for row in rows:
            handle.write(f"| `{row['model']}` | {row['hand_frames']} | {row['face_frames']} |\n")
        handle.write("\n## Checkpoints\n\n")
        for row in rows:
            handle.write(f"- `{row['model']}`: `{row['checkpoint']}`\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


def read_existing_rows(output_dir):
    csv_path = output_dir / "metrics.csv"
    if not csv_path.exists():
        return []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_specs(path, selected):
    with open(path, "r", encoding="utf-8") as handle:
        specs = yaml.safe_load(handle)["autoencoders"]
    if selected:
        selected = set(selected)
        specs = [spec for spec in specs if spec["name"] in selected]
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(REPO_ROOT / "autoencoder_used" / "autoencoders.yml"))
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--hdf5", default=None)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum frames to evaluate. Use 0 or a negative value for the full input set.",
    )
    parser.add_argument("--frames-per-video", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "autoencoder_used" / "results"))
    parser.add_argument(
        "--detail-mask-cache",
        default=None,
        help="Grounded-SAM2 v4 mask-cache root; masks override pose-derived ROIs.",
    )
    parser.add_argument(
        "--detail-mask-split",
        default=None,
        help="Mask-cache split subdirectory, for example train/dev/test.",
    )
    parser.add_argument(
        "--detail-mask-strict",
        action="store_true",
        help="Fail if a sample or selected frame has no external hand/face mask.",
    )
    parser.add_argument("--skip-missing", action="store_true", default=True)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip models already present in output-dir/metrics.csv and preserve their rows.",
    )
    parser.add_argument("--lpips", action="store_true", help="Compute LPIPS with local models/eval_models weights when available.")
    parser.add_argument("--roi-pose-threshold", type=float, default=0.02)
    parser.add_argument("--hand-blue-threshold", type=float, default=0.58)
    parser.add_argument("--hand-other-max", type=float, default=0.45)
    parser.add_argument("--hand-seed-dilation", type=int, default=9)
    parser.add_argument("--hand-final-dilation", type=int, default=4)
    parser.add_argument("--hand-padding", type=int, default=8)
    parser.add_argument("--hand-min-pixels", type=int, default=3)
    parser.add_argument("--face-white-threshold", type=float, default=0.70)
    parser.add_argument("--face-chroma-tolerance", type=float, default=0.18)
    parser.add_argument("--face-dilation", type=int, default=5)
    parser.add_argument("--face-padding", type=int, default=5)
    parser.add_argument("--face-min-pixels", type=int, default=3)
    args = parser.parse_args()

    output_dir = resolve_path(args.output_dir)
    (output_dir / "samples").mkdir(parents=True, exist_ok=True)
    specs = load_specs(resolve_path(args.config), args.models)
    rows = read_existing_rows(output_dir) if args.skip_existing else []
    existing_models = {row.get("model") for row in rows}

    # Inputs may need different image sizes, so load per-spec.
    for spec in specs:
        if args.skip_existing and spec["name"] in existing_models:
            print(f"[skip-existing] {spec['name']}: already in {output_dir / 'metrics.csv'}")
            continue
        print(f"Evaluating {spec['name']}...")
        row = evaluate_one(spec, None, args, torch.device(args.device), output_dir)
        if row is not None:
            rows.append(row)
            print(
                f"  L1={row['l1']:.6f} MSE={row['mse']:.6f} "
                f"PSNR={row['psnr']:.3f} SSIM={row['ssim']:.4f} "
                f"LPIPS={row['lpips']:.6f}"
            )

    if not rows:
        raise RuntimeError("No models were evaluated")
    write_reports(rows, output_dir)


if __name__ == "__main__":
    main()
