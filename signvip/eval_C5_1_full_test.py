"""Deterministic full-test evaluation for C5.1 Stage I on paired HDF5 data."""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from PIL import Image, ImageDraw
from skimage.metrics import structural_similarity
from tqdm import tqdm

from train_stage1_multi_cond import load_config_file, load_latent_refiner, load_modules
from metrics.calculate_ssim import calculate_ssim_function
from pipelines.pipeline_static import SignViPStaticPipeline
from signdatasets import HDF5SignLangVideoDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--test-root", required=True)
    parser.add_argument("--hamer-root", required=True)
    parser.add_argument("--detail-mask-cache", required=True)
    parser.add_argument("--pose-keypoint-cache", required=True)
    parser.add_argument("--paper-hand-crop-size", type=int, default=60)
    parser.add_argument("--paper-hand-score-threshold", type=float, default=0.3)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-frames", type=int, default=5)
    parser.add_argument("--sample-rate", type=int, default=4)
    parser.add_argument("--num-inference-steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--only-index", type=int, default=None)
    parser.add_argument(
        "--indices-file",
        default=None,
        help="Optional newline-delimited dataset indices for deterministic sharded evaluation.",
    )
    parser.add_argument("--save-previews", type=int, default=20)
    parser.add_argument(
        "--top-hand-previews",
        type=int,
        default=10,
        help="Keep comparison images for the N videos with the best hand metric.",
    )
    parser.add_argument(
        "--top-hand-metric",
        choices=(
            "hand_ssim",
            "hand_masked_ssim",
            "paper_hand_ssim",
            "hand_psnr",
            "hand_lpips",
            "hand_mse",
        ),
        default="hand_ssim",
    )
    parser.add_argument("--signvip-protocol", action="store_true")
    parser.add_argument("--centered-clip", action="store_true")
    parser.add_argument("--inference-chunk-size", type=int, default=5)
    parser.add_argument("--no-lpips", action="store_true")
    parser.add_argument(
        "--disable-hand-edge",
        action="store_true",
        help="Set gamma_hand_edge to zero after loading for a fixed-seed paired ablation.",
    )
    return parser.parse_args()


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def read_records(path):
    records = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def frame_metrics(real, generated):
    real_np = real.detach().float().cpu().numpy()
    generated_np = generated.detach().float().cpu().numpy()
    ssim = float(calculate_ssim_function(real_np, generated_np))
    mse = float(np.mean((real_np - generated_np) ** 2))
    psnr = float("inf") if mse == 0 else -10.0 * math.log10(mse)
    return {"mse": mse, "psnr": psnr, "ssim": ssim}


def build_lpips(device, enabled):
    if not enabled:
        return None
    import lpips

    metric = lpips.LPIPS(net="alex", version="0.1").to(device)
    metric.eval().requires_grad_(False)
    return metric


def lpips_values(metric, pred, target, device, batch_size=32):
    if metric is None or pred.numel() == 0:
        return [math.nan] * len(pred)
    output = []
    with torch.inference_mode():
        for start in range(0, len(pred), batch_size):
            values = metric(
                pred[start : start + batch_size].to(device=device, dtype=torch.float32),
                target[start : start + batch_size].to(device=device, dtype=torch.float32),
                normalize=True,
            )
            output.extend(float(value) for value in values.detach().flatten().cpu())
    return output


def gray_ssim(pred, target):
    pred_np = pred.detach().cpu().permute(1, 2, 0).numpy()
    target_np = target.detach().cpu().permute(1, 2, 0).numpy()
    pred_gray = 0.299 * pred_np[..., 0] + 0.587 * pred_np[..., 1] + 0.114 * pred_np[..., 2]
    target_gray = 0.299 * target_np[..., 0] + 0.587 * target_np[..., 1] + 0.114 * target_np[..., 2]
    return float(
        structural_similarity(
            target_gray,
            pred_gray,
            data_range=1,
            gaussian_weights=True,
            use_sample_covariance=False,
        )
    )


def masked_ssim(pred, target, mask):
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()
    weights = mask.detach().float().cpu().numpy()
    denominator = float(weights.sum())
    if denominator <= 0:
        return math.nan
    channel_maps = []
    for channel in range(pred_np.shape[0]):
        _, ssim_map = structural_similarity(
            target_np[channel],
            pred_np[channel],
            data_range=1,
            gaussian_weights=True,
            use_sample_covariance=False,
            full=True,
        )
        channel_maps.append(ssim_map)
    return float((np.mean(channel_maps, axis=0) * weights).sum() / denominator)


def crop_pair(pred, target, mask):
    positions = torch.where(mask.detach().cpu().bool())
    if len(positions[0]) == 0:
        return None
    y0, y1 = int(positions[0].min()), int(positions[0].max()) + 1
    x0, x1 = int(positions[1].min()), int(positions[1].max()) + 1
    pred_crop = F.interpolate(
        pred[:, y0:y1, x0:x1].unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )[0]
    target_crop = F.interpolate(
        target[:, y0:y1, x0:x1].unsqueeze(0),
        size=(64, 64),
        mode="bilinear",
        align_corners=False,
    )[0]
    return pred_crop, target_crop


def load_detail_masks(cache_root, text_key, frame_ids, output_size):
    suffix = "_grounded_sam2_face_hands_384_to128_v4.npz"
    path = cache_root / "test" / f"{text_key}{suffix}"
    if not path.is_file():
        raise FileNotFoundError(f"Missing detail mask: {path}")
    with np.load(path) as data:
        positions = {int(frame): index for index, frame in enumerate(data["frame_indices"])}
        try:
            hand = torch.from_numpy(
                np.stack([data["hands"][positions[int(frame)]] for frame in frame_ids])
            ).float()
            face = torch.from_numpy(
                np.stack([data["face"][positions[int(frame)]] for frame in frame_ids])
            ).float()
        except KeyError as error:
            raise KeyError(f"Mask {path} does not contain requested frame {error}") from error
    if tuple(hand.shape[-2:]) != tuple(output_size):
        hand = F.interpolate(hand[:, None], size=output_size, mode="nearest")[:, 0]
        face = F.interpolate(face[:, None], size=output_size, mode="nearest")[:, 0]
    return hand, face


def load_paper_hand_centers(cache_root, text_key, frame_ids, output_size, score_threshold):
    matches = sorted((cache_root / "test").glob(f"{text_key}_dwpose*_keypoints_v1.npz"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one DWPose keypoint cache for {text_key!r}, found {len(matches)}"
        )
    with np.load(matches[0], allow_pickle=False) as data:
        keypoints = np.asarray(data["keypoints"], dtype=np.float32)
        scores = np.asarray(data["scores"], dtype=np.float32)
        offsets = np.asarray(data["frame_offsets"], dtype=np.int64)
    height, width = map(int, output_size)
    centers = []
    for frame_id in frame_ids:
        frame_id = int(frame_id)
        if frame_id < 0 or frame_id + 1 >= len(offsets):
            raise IndexError(f"Frame {frame_id} is outside keypoint cache {matches[0]}")
        start, end = int(offsets[frame_id]), int(offsets[frame_id + 1])
        if start == end:
            centers.append([])
            continue
        frame_scores = scores[start:end]
        person_index = int(np.nanmean(frame_scores[:, :18], axis=1).argmax())
        person_keypoints = keypoints[start + person_index]
        person_scores = frame_scores[person_index]
        frame_centers = []
        for side, joint_index in (("left", 101), ("right", 122)):
            xy = person_keypoints[joint_index]
            confidence = float(person_scores[joint_index])
            if confidence < score_threshold or not np.isfinite(xy).all():
                continue
            if not (0.0 <= float(xy[0]) <= 1.0 and 0.0 <= float(xy[1]) <= 1.0):
                continue
            frame_centers.append(
                {
                    "side": side,
                    "x": float(xy[0]) * width,
                    "y": float(xy[1]) * height,
                    "confidence": confidence,
                }
            )
        centers.append(frame_centers)
    return centers


def fixed_square_crop(image, center_x, center_y, crop_size):
    _, height, width = image.shape
    if crop_size < 11:
        raise ValueError("paper hand crop must be at least 11 pixels for SSIM")
    if crop_size > height or crop_size > width:
        raise ValueError(f"paper hand crop {crop_size} exceeds frame size {height}x{width}")
    x0 = min(max(int(round(float(center_x) - crop_size / 2)), 0), width - crop_size)
    y0 = min(max(int(round(float(center_y) - crop_size / 2)), 0), height - crop_size)
    return image[:, y0 : y0 + crop_size, x0 : x0 + crop_size]


def paper_hand_ssim_values(real, generated, hand_centers, crop_size):
    values, roi_records = [], []
    for frame_index, centers in enumerate(hand_centers):
        for center in centers:
            target_crop = fixed_square_crop(real[frame_index], center["x"], center["y"], crop_size)
            pred_crop = fixed_square_crop(generated[frame_index], center["x"], center["y"], crop_size)
            value = float(
                calculate_ssim_function(
                    target_crop.detach().float().cpu().numpy(),
                    pred_crop.detach().float().cpu().numpy(),
                )
            )
            values.append(value)
            roi_records.append(
                {
                    "frame": frame_index,
                    "side": center["side"],
                    "confidence": center["confidence"],
                    "ssim": value,
                }
            )
    return values, roi_records


def detailed_metrics(
    real,
    generated,
    hand_masks,
    face_masks,
    lpips_metric,
    device,
    paper_hand_centers=None,
    paper_hand_crop_size=60,
):
    per_frame = [frame_metrics(real[i], generated[i]) for i in range(len(real))]
    for record, value in zip(per_frame, lpips_values(lpips_metric, generated, real, device)):
        record["lpips"] = value
    region_values = {name: [] for name in ("hand", "face", "detail")}
    for frame, (pred, target, hand, face) in enumerate(
        zip(generated, real, hand_masks, face_masks)
    ):
        for name, mask in (
            ("hand", hand),
            ("face", face),
            ("detail", torch.maximum(hand, face)),
        ):
            mask = mask.to(dtype=pred.dtype)
            denominator = float(mask.sum().item()) * pred.shape[0]
            if denominator <= 0:
                continue
            mse = float(((pred - target).pow(2) * mask).sum().item() / denominator)
            crops = crop_pair(pred, target, mask)
            if crops is None:
                continue
            pred_crop, target_crop = crops
            region_values[name].append(
                {
                    "frame": frame,
                    "mse": mse,
                    "psnr": -10.0 * math.log10(max(mse, 1e-12)),
                    "ssim": float(
                        calculate_ssim_function(
                            target_crop.detach().float().cpu().numpy(),
                            pred_crop.detach().float().cpu().numpy(),
                        )
                    ),
                    "ssim_gray": gray_ssim(pred_crop, target_crop),
                    "masked_ssim": masked_ssim(pred, target, mask),
                    "mask_fraction": float(mask.mean().item()),
                    "pred_crop": pred_crop,
                    "target_crop": target_crop,
                }
            )
    result = {
        name: float(np.mean([record[name] for record in per_frame]))
        for name in ("mse", "psnr", "ssim", "lpips")
    }
    for name in ("mse", "psnr", "ssim", "lpips"):
        result[f"{name}_values"] = [record[name] for record in per_frame]
    for region, values in region_values.items():
        if not values:
            continue
        crop_lpips = lpips_values(
            lpips_metric,
            torch.stack([value["pred_crop"] for value in values]),
            torch.stack([value["target_crop"] for value in values]),
            device,
        )
        for value, lpips_value in zip(values, crop_lpips):
            value["lpips"] = lpips_value
        for metric_name in (
            "mse",
            "psnr",
            "ssim",
            "ssim_gray",
            "masked_ssim",
            "lpips",
            "mask_fraction",
        ):
            metric_values = [value[metric_name] for value in values]
            result[f"{region}_{metric_name}"] = float(np.mean(metric_values))
            result[f"{region}_{metric_name}_values"] = metric_values
        result[f"{region}_frames"] = len(values)
    if paper_hand_centers is not None:
        values, roi_records = paper_hand_ssim_values(
            real, generated, paper_hand_centers, paper_hand_crop_size
        )
        result["paper_hand_ssim_values"] = values
        result["paper_hand_ssim"] = float(np.mean(values)) if values else math.nan
        result["paper_hand_rois"] = len(values)
        result["paper_hand_roi_records"] = roi_records
    return result


def save_preview(path, real, generated, title=None):
    rows = [torch.cat((real[i], generated[i]), dim=2) for i in range(real.shape[0])]
    grid = torch.cat(rows, dim=1).clamp(0, 1)
    array = (grid.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    preview = Image.fromarray(array)
    if title:
        canvas = Image.new("RGB", (preview.width, preview.height + 24), "white")
        canvas.paste(preview, (0, 24))
        ImageDraw.Draw(canvas).text((5, 5), title, fill="black")
        preview = canvas
    preview.save(path)


def _hand_rank_score(metric_name, metric_value):
    value = float(metric_value)
    if not math.isfinite(value):
        return None
    return -value if metric_name in ("hand_mse", "hand_lpips") else value


def _safe_filename_component(value):
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def update_top_hand_previews(
    candidates, *, limit, metric_name, record, real, generated, output_dir
):
    if limit <= 0 or metric_name not in record:
        return
    metric_value = float(record[metric_name])
    rank_score = _hand_rank_score(metric_name, metric_value)
    if rank_score is None:
        return
    index = int(record["index"])
    existing = next((item for item in candidates if item["index"] == index), None)
    if existing is not None:
        candidates.remove(existing)
        Path(existing["image"]).unlink(missing_ok=True)
    qualifies = len(candidates) < limit or rank_score > min(
        item["rank_score"] for item in candidates
    )
    if not qualifies:
        return
    safe_key = _safe_filename_component(record["text_key"])
    image_path = output_dir / (
        f"sample_{index:06d}_{safe_key}_{metric_name}_{metric_value:.6f}.png"
    )
    save_preview(
        image_path,
        real,
        generated,
        title=(
            f"GT (left) | Prediction (right) | index={index} | "
            f"{metric_name}={metric_value:.6f}"
        ),
    )
    candidates.append(
        {
            "index": index,
            "text_key": record["text_key"],
            "frame_ids": record["frame_ids"],
            "metric": metric_name,
            "value": metric_value,
            "rank_score": rank_score,
            "image": str(image_path.resolve()),
        }
    )
    if len(candidates) > limit:
        removed = min(candidates, key=lambda item: item["rank_score"])
        candidates.remove(removed)
        Path(removed["image"]).unlink(missing_ok=True)
    ranked = sorted(candidates, key=lambda item: item["rank_score"], reverse=True)
    manifest = {
        "metric": metric_name,
        "higher_is_better": metric_name not in ("hand_mse", "hand_lpips"),
        "requested_count": int(limit),
        "samples": [{**item, "rank": rank} for rank, item in enumerate(ranked, start=1)],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def load_top_hand_previews(output_dir, metric_name, limit):
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        return []
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("metric") != metric_name:
        return []
    candidates = []
    for item in manifest.get("samples", []):
        if Path(item["image"]).is_file():
            candidate = dict(item)
            candidate.pop("rank", None)
            candidates.append(candidate)
    candidates.sort(key=lambda item: item["rank_score"], reverse=True)
    return candidates[: max(int(limit), 0)]


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    top_hand_dir = output_dir / "top10_hand"
    top_hand_dir.mkdir(exist_ok=True)
    if args.top_hand_previews < 0:
        raise ValueError("--top-hand-previews must be non-negative")
    top_hand_candidates = load_top_hand_previews(
        top_hand_dir, args.top_hand_metric, args.top_hand_previews
    )
    records_path = output_dir / "per_video_metrics.jsonl"
    detail_mask_cache = Path(args.detail_mask_cache).expanduser().resolve()
    pose_keypoint_cache = Path(args.pose_keypoint_cache).expanduser().resolve()
    if args.paper_hand_crop_size < 11:
        raise ValueError("--paper-hand-crop-size must be at least 11")
    if args.inference_chunk_size < 1:
        raise ValueError("--inference-chunk-size must be at least 1")

    cfg = load_config_file(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_vae = checkpoint_dir / "vae"
    if (checkpoint_vae / "config.json").is_file():
        cfg.modules.vae = str(checkpoint_vae)
    cfg.modules.apperance_encoder = str(checkpoint_dir / "appearance_encoder")
    cfg.modules.condition_encoder = str(checkpoint_dir / "condition_encoder" / "model.bin")
    checkpoint_unet = checkpoint_dir / "unet" / "model.bin"
    if checkpoint_unet.is_file():
        cfg.modules.unet = str(checkpoint_unet)
    reinjector_path = checkpoint_dir / "hamer_reinjector" / "model.bin"
    if reinjector_path.is_file():
        cfg.modules.hamer_reinjector = str(reinjector_path)
    latent_refiner_path = checkpoint_dir / "latent_refiner" / "model.bin"
    if latent_refiner_path.is_file():
        cfg.modules.latent_refiner = str(latent_refiner_path)
    elif cfg.modules.get("latent_refiner_kwargs", None):
        raise FileNotFoundError(
            f"C5.19 checkpoint is missing latent refiner: {latent_refiner_path}"
        )

    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    weight_dtype = torch.float16
    (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        _noise_scheduler,
        scheduler,
        empty_text_emb,
    ) = load_modules(cfg, device, weight_dtype)
    # Stage-I checkpoints may store trainable condition/appearance modules in
    # FP32 while the inference inputs and diffusion backbone use FP16.
    # Explicitly align both encoders before the pipeline forward pass.
    appearance_encoder = appearance_encoder.to(device=device, dtype=weight_dtype)
    condition_encoder = condition_encoder.to(device=device, dtype=weight_dtype)
    latent_refiner = load_latent_refiner(cfg, device, weight_dtype)
    if args.disable_hand_edge:
        reinjector = getattr(unet, "hamer_reinjector", None)
        if reinjector is None or not getattr(reinjector, "hand_edge_enabled", False):
            raise RuntimeError("--disable-hand-edge requires a hand-edge checkpoint")
        with torch.no_grad():
            reinjector.gamma_hand_edge.zero_()
        print("Paired ablation: Edge OFF (gamma_hand_edge=0).")
    for module in (vae, unet, appearance_encoder, condition_encoder):
        module.requires_grad_(False)
        module.eval()
    if latent_refiner is not None:
        latent_refiner.requires_grad_(False)
        latent_refiner.eval()

    pipeline = SignViPStaticPipeline(
        vae=vae,
        denoising_unet=unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=appearance_encoder,
        condition_encoder=condition_encoder,
        latent_refiner=latent_refiner,
    ).to(device=device, dtype=weight_dtype)
    pipeline.set_progress_bar_config(disable=True)
    lpips_metric = build_lpips(device, enabled=not args.no_lpips)

    dataset = HDF5SignLangVideoDataset(
        roots=args.test_root,
        hamer_roots=args.hamer_root,
        frame_size=cfg.dataset.frame_size,
        frame_scale=(1.0, 1.0),
        frame_ratio=(1.0, 1.0),
        sample_rate=args.sample_rate,
        num_frames=args.num_frames,
        ref_margin=cfg.dataset.ref_margin,
        random_sample=False,
        mask_ratio=0,
        sk_mask_ratio=0,
        hamer_mask_ratio=0,
        both_mask_ratio=0,
        full_video=args.signvip_protocol,
        centered_clip=args.centered_clip,
        reference_mode="first" if args.signvip_protocol else "outside_clip",
        pose_keypoint_cache=str(pose_keypoint_cache),
        pose_keypoint_split="test",
        pose_keypoint_strict=True,
        hand_roi_score_threshold=float(
            cfg.dataset.get("hand_roi_score_threshold", 0.30)
        ),
        hand_roi_margin=float(cfg.dataset.get("hand_roi_margin", 0.20)),
    )
    total = len(dataset) if args.max_samples <= 0 else min(len(dataset), args.max_samples)
    existing = read_records(records_path)
    completed = {int(record["index"]) for record in existing}
    selected_indices = None
    if args.indices_file is not None:
        if args.only_index is not None:
            raise ValueError("--indices-file and --only-index are mutually exclusive")
        indices_path = Path(args.indices_file)
        selected_indices = [
            int(line.strip())
            for line in indices_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(selected_indices) != len(set(selected_indices)):
            raise ValueError(f"Duplicate indices in {indices_path}")
        invalid = [index for index in selected_indices if not 0 <= index < total]
        if invalid:
            raise ValueError(
                f"Indices in {indices_path} must be in [0, {total - 1}]: {invalid[:5]}"
            )
        evaluation_indices = selected_indices
    elif args.only_index is not None:
        if not 0 <= args.only_index < len(dataset):
            raise ValueError(f"--only-index must be in [0, {len(dataset) - 1}]")
        evaluation_indices = [args.only_index]
    else:
        evaluation_indices = range(total)

    for index in tqdm(evaluation_indices, desc="C5.1 full-test"):
        if index in completed:
            continue
        sample_seed = args.seed + index
        random.seed(sample_seed)
        np.random.seed(sample_seed % (2**32))
        torch.manual_seed(sample_seed)
        sample = dataset[index]
        reference = sample["ref_frame"].unsqueeze(0).to(device)
        pose = sample["tgt_sk_frames"].unsqueeze(0)
        hamer = sample["tgt_hamer_frames"].unsqueeze(0)
        left_hand_roi = sample.get("left_hand_roi_mask")
        right_hand_roi = sample.get("right_hand_roi_mask")
        if left_hand_roi is not None:
            left_hand_roi = left_hand_roi.unsqueeze(0)
            right_hand_roi = right_hand_roi.unsqueeze(0)
        real = (sample["tgt_frames"].permute(1, 0, 2, 3) / 2 + 0.5).clamp(0, 1)
        generator = torch.Generator(device="cpu").manual_seed(sample_seed)
        frame_count = int(pose.shape[2])
        latent_height = cfg.dataset.frame_size[0] // pipeline.vae_scale_factor
        latent_width = cfg.dataset.frame_size[1] // pipeline.vae_scale_factor
        initial_latents = torch.randn(
            (1, unet.in_channels, frame_count, latent_height, latent_width),
            generator=generator,
            device="cpu",
            dtype=weight_dtype,
        )
        generated_chunks = []
        chunk_size = args.inference_chunk_size if args.signvip_protocol else frame_count
        for chunk_start in range(0, frame_count, chunk_size):
            chunk_end = min(chunk_start + chunk_size, frame_count)
            with torch.no_grad(), accelerator.autocast():
                chunk = pipeline(
                    prompt=cfg.validation_data.prompt,
                    reference_image=reference,
                    sk_image=pose[:, :, chunk_start:chunk_end].to(device),
                    hamer_image=hamer[:, :, chunk_start:chunk_end].to(device),
                    left_hand_roi_mask=(
                        left_hand_roi[:, :, chunk_start:chunk_end].to(device)
                        if left_hand_roi is not None else None
                    ),
                    right_hand_roi_mask=(
                        right_hand_roi[:, :, chunk_start:chunk_end].to(device)
                        if right_hand_roi is not None else None
                    ),
                    width=cfg.dataset.frame_size[1],
                    height=cfg.dataset.frame_size[0],
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    latents=initial_latents[:, :, chunk_start:chunk_end],
                )
            generated_chunks.append(chunk)
        generated = torch.cat(generated_chunks, dim=2)
        generated = generated[0].permute(1, 0, 2, 3).float().cpu().clamp(0, 1)
        hand_masks, face_masks = load_detail_masks(
            detail_mask_cache,
            sample["text_key"],
            sample["tgt_frame_ids"].tolist(),
            real.shape[-2:],
        )
        paper_hand_centers = load_paper_hand_centers(
            pose_keypoint_cache,
            sample["text_key"],
            sample["tgt_frame_ids"].tolist(),
            real.shape[-2:],
            args.paper_hand_score_threshold,
        )
        metrics = detailed_metrics(
            real,
            generated,
            hand_masks,
            face_masks,
            lpips_metric,
            device,
            paper_hand_centers=paper_hand_centers,
            paper_hand_crop_size=args.paper_hand_crop_size,
        )
        record = {
            "index": index,
            "text_key": sample["text_key"],
            "frame_ids": sample["tgt_frame_ids"].tolist(),
            "num_frames": int(real.shape[0]),
            "seed": sample_seed,
            "reference_frame_id": int(sample["reference_frame_id"]),
            **metrics,
        }
        append_jsonl(records_path, record)
        update_top_hand_previews(
            top_hand_candidates,
            limit=args.top_hand_previews,
            metric_name=args.top_hand_metric,
            record=record,
            real=real,
            generated=generated,
            output_dir=top_hand_dir,
        )
        if args.only_index is not None or index < args.save_previews:
            save_preview(
                preview_dir / f"{index:04d}_{sample['text_key']}.png",
                real,
                generated,
            )

    records = read_records(records_path)
    if selected_indices is not None:
        selected_set = set(selected_indices)
        records = [record for record in records if int(record["index"]) in selected_set]
    elif args.only_index is not None:
        records = [record for record in records if int(record["index"]) == args.only_index]
    else:
        records = [record for record in records if int(record["index"]) < total]
    records.sort(key=lambda record: int(record["index"]))
    metric_names = (
        "mse", "psnr", "ssim", "lpips", "paper_hand_ssim",
        "hand_mse", "hand_psnr", "hand_ssim", "hand_ssim_gray",
        "hand_masked_ssim", "hand_lpips", "hand_mask_fraction",
        "face_mse", "face_psnr", "face_ssim", "face_ssim_gray",
        "face_masked_ssim", "face_lpips", "face_mask_fraction",
        "detail_mse", "detail_psnr", "detail_ssim", "detail_ssim_gray",
        "detail_masked_ssim", "detail_lpips", "detail_mask_fraction",
    )
    summary = {
        "experiment": str(cfg.get("exp_name", Path(args.config).stem)),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "test_root": str(Path(args.test_root).resolve()),
        "hamer_root": str(Path(args.hamer_root).resolve()),
        "detail_mask_cache": str(detail_mask_cache),
        "pose_keypoint_cache": str(pose_keypoint_cache),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "hand_edge_disabled": args.disable_hand_edge,
        "num_frames_per_video": args.num_frames,
        "sample_rate": args.sample_rate,
        "signvip_protocol": args.signvip_protocol,
        "inference_chunk_size": args.inference_chunk_size,
        "num_frames_total": int(sum(record["num_frames"] for record in records)),
        "only_index": args.only_index,
        "indices_file": args.indices_file,
        "num_videos": len(records),
        "top_hand_previews": {
            "metric": args.top_hand_metric,
            "requested_count": args.top_hand_previews,
            "saved_count": len(top_hand_candidates),
            "directory": str(top_hand_dir.resolve()),
            "manifest": str((top_hand_dir / "manifest.json").resolve()),
        },
        "metric_protocol": {
            "global_ssim": "RGB Gaussian SSIM, data_range=1",
            "roi_ssim": "RGB Gaussian SSIM on Grounded-SAM2 v4 tight mask bounding-box resized to 64x64",
            "masked_ssim": "mean RGB Gaussian SSIM map weighted only by Grounded-SAM2 v4 mask pixels at 128x128",
            "roi_ssim_gray": "grayscale Gaussian SSIM on the same crop; matches the local C3.7 evaluator",
            "roi_lpips": "AlexNet LPIPS on the same crop",
            "roi_mse_psnr": "soft-mask-weighted RGB error on the full 128x128 frame",
            "lpips": "official lpips 0.1.4, AlexNet v0.1, RGB [0,1] normalized to [-1,1]",
            "paper_hand_ssim": (
                f"RGB Gaussian SSIM on separate fixed {args.paper_hand_crop_size}x"
                f"{args.paper_hand_crop_size} crops centered at GT DWPose left/right "
                "middle-finger MCP (hand index 9); confidence threshold "
                f"{args.paper_hand_score_threshold:g}; boundary crops shifted to remain fixed-size"
            ),
            "aggregation": "frame-weighted means across all evaluated frames/ROIs",
            "reference": (
                "first frame of each video"
                if args.signvip_protocol
                else "deterministic frame outside the target clip"
            ),
        },
    }
    for name in metric_names:
        values = []
        values_key = f"{name}_values"
        for record in records:
            candidates = record.get(values_key)
            if candidates is None and name in record:
                candidates = [record[name]]
            for value in candidates or []:
                value = float(value)
                if math.isfinite(value):
                    values.append(value)
        values = np.asarray(values, dtype=np.float64)
        summary[f"{name}_mean"] = float(values.mean()) if len(values) else math.nan
        summary[f"{name}_std"] = float(values.std()) if len(values) else math.nan
        summary[f"{name}_count"] = int(len(values))
    summary_path = output_dir / "metrics_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
