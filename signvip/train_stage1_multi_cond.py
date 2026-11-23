import argparse
import datetime
import itertools
import json
import logging
import math
import os
import pathlib
import random
import shutil
import subprocess
import time
import warnings
from typing import Optional

import accelerate
import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.nn.functional as F
import torchvision.transforms as transforms
from accelerate.logging import get_logger
from diffusers.utils import import_utils as diffusers_import_utils

# C3's Blackwell PyTorch overlay is intentionally newer than the xFormers wheel
# in its base environment. Stage I never enables xFormers attention, so prevent
# Diffusers from importing that incompatible optional extension.
diffusers_import_utils._xformers_available = False

from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, UNet2DConditionModel
from diffusers.configuration_utils import FrozenDict
from diffusers.optimization import get_scheduler
from einops import rearrange
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

# from metrics.calculate_fid import calculate_fid as calc_fid  # 添加 FID 计算导入
from metrics.calculate_ssim import (
    calculate_hand_nonhand_ssim,
    calculate_ssim as calc_ssim,
)
from models.appearance_encoder import AppearanceEncoderModel
from models.condition_encoder import ConditionEncoder
from models.hand_latent_refiner import HandLatentResidualRefiner
from models.mutual_self_attention import ReferenceAttentionControl
from models.unet import UNet3DConditionModel
from pipelines.pipeline_static import SignViPStaticPipeline
from perceptual_losses import (
    clip_loss,
    crop_resize_region_pairs,
    decode_latents,
    dino_loss,
    flatten_region_masks,
    flatten_target_frames,
    global_reconstruction_loss,
    load_clip_model,
    load_dino_model,
    normalized_masked_reconstruction_loss,
    predict_x0,
)
import signdatasets
from utils import get_num_params, seed_everything

warnings.filterwarnings("ignore")
logger = get_logger(__name__, log_level="INFO")


def save_resumable_checkpoint(accelerator, workspace_dir, step, max_ckpt):
    """Publish a complete checkpoint before pruning any previous state."""
    root = pathlib.Path(workspace_dir)
    destination = root / f"checkpoint-{step}"
    staging = root / f".checkpoint-{step}.incomplete"
    required = ["model.safetensors", "optimizer.bin", "scheduler.bin",
                "random_states_0.pkl"]
    if accelerator.scaler is not None:
        required.append("scaler.pt")
    # An immediate early stop after resume can end at an already saved step.
    if destination.exists():
        if all((destination / name).is_file() and (destination / name).stat().st_size
               for name in required):
            accelerator.wait_for_everyone()
            return
        raise RuntimeError(f"Existing checkpoint is incomplete: {destination}")
    accelerator.save_state(str(staging))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if not all((staging / name).is_file() and (staging / name).stat().st_size
                   for name in required):
            raise RuntimeError(f"Incomplete checkpoint; previous state retained: {staging}")
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing checkpoint: {destination}")
        staging.rename(destination)
        if max_ckpt is not None:
            keep_count = int(max_ckpt)
            if keep_count < 1:
                raise ValueError("max_ckpt must be at least 1")
            checkpoints = sorted(
                (p for p in root.iterdir() if p.is_dir() and not p.is_symlink()
                 and p.name.startswith("checkpoint-")
                 and p.name.removeprefix("checkpoint-").isdigit()),
                key=lambda p: int(p.name.removeprefix("checkpoint-")),
            )
            for previous in checkpoints[:-keep_count]:
                logger.info("Removing superseded checkpoint after successful save: %s", previous)
                shutil.rmtree(previous)
    accelerator.wait_for_everyone()


def notify_best_checkpoint(workspace_dir, metrics_path, experiment_name):
    """Send a non-fatal Telegram notification for a newly saved best model."""
    if os.environ.get("SIGNVIP_TELEGRAM_BEST_NOTIFY", "0") != "1":
        return
    notify_script = pathlib.Path(__file__).resolve().parent / "notify_telegram.sh"
    command = [
        str(notify_script),
        "--event",
        "best-checkpoint",
        "--experiment",
        str(experiment_name),
        "--run-dir",
        str(workspace_dir),
        "--best-metrics-json",
        str(metrics_path),
    ]
    try:
        result = subprocess.run(command, check=False, timeout=30)
        if result.returncode != 0:
            logger.warning(
                "Best-checkpoint Telegram notifier exited with code %d",
                result.returncode,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Best-checkpoint Telegram notification failed: %s", exc)


def notify_training_progress(workspace_dir, experiment_name, step, max_steps):
    """Send a non-fatal Telegram heartbeat at configured step intervals."""
    if os.environ.get("TELEGRAM_NOTIFY", "1") != "1":
        return
    interval = int(os.environ.get("SIGNVIP_TELEGRAM_PROGRESS_STEPS", "0"))
    if interval <= 0 or step <= 0 or step % interval != 0:
        return
    progress_percent = min(int(step * 100 / max(max_steps, 1)), 100)
    notify_script = pathlib.Path(__file__).resolve().parent / "notify_telegram.sh"
    command = [
        str(notify_script),
        "--event",
        "progress",
        "--experiment",
        str(experiment_name),
        "--stage",
        f"Stage I training: step {step:,}/{max_steps:,}",
        "--progress-percent",
        str(progress_percent),
        "--run-dir",
        str(workspace_dir),
        "--note",
        f"Job vẫn đang chạy; đã hoàn thành step {step:,}.",
    ]
    try:
        result = subprocess.run(command, check=False, timeout=30)
        if result.returncode != 0:
            logger.warning(
                "Progress Telegram notifier exited with code %d",
                result.returncode,
            )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("Progress Telegram notification failed: %s", exc)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def build_soft_hand_latent_mask(
    hand_mask,
    target_size,
    *,
    threshold=0.05,
    dilation_radius=2,
    ring1_weight=0.5,
    ring2_weight=0.25,
):
    """Downsample an RGB hand mask and add up to two soft latent rings."""
    if hand_mask.ndim != 5 or hand_mask.shape[1] != 1:
        raise ValueError(f"Expected hand mask [B,1,F,H,W], got {tuple(hand_mask.shape)}")
    batch, _, frames, _, _ = hand_mask.shape
    flat = rearrange(hand_mask.float(), "b 1 f h w -> (b f) 1 h w")
    core = (
        F.interpolate(flat, size=tuple(target_size), mode="area")
        > float(threshold)
    ).float()
    radius = int(dilation_radius)
    if radius not in (0, 1, 2):
        raise ValueError("hand latent dilation_radius must be 0, 1, or 2")
    roi = core
    if radius >= 1:
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        roi = roi + float(ring1_weight) * (dilated_1 - core)
    if radius >= 2:
        dilated_2 = F.max_pool2d(core, kernel_size=5, stride=1, padding=2)
        roi = roi + float(ring2_weight) * (dilated_2 - dilated_1)
    return rearrange(roi, "(b f) 1 h w -> b 1 f h w", b=batch, f=frames)


def prediction_from_x0(noisy_latents, predicted_x0, timesteps, scheduler):
    """Convert a corrected clean latent back to the scheduler prediction type."""
    alphas = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha = alphas[timesteps].sqrt().view(-1, 1, 1, 1, 1)
    sigma = (1 - alphas[timesteps]).sqrt().view(-1, 1, 1, 1, 1)
    if scheduler.prediction_type == "epsilon":
        return (noisy_latents - alpha * predicted_x0) / sigma.clamp_min(1e-6)
    if scheduler.prediction_type == "v_prediction":
        return (alpha * noisy_latents - predicted_x0) / sigma.clamp_min(1e-6)
    raise ValueError(f"Unsupported prediction type: {scheduler.prediction_type}")


def hand_frequency_losses(prediction, target, mask):
    """Area-normalized Sobel-gradient and Laplacian L1 losses in a hand ROI."""
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("Hand-frequency loss expects matching NCHW RGB tensors")
    mask = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="nearest")
    channels = prediction.shape[1]
    sobel_x = prediction.new_tensor(((-1, 0, 1), (-2, 0, 2), (-1, 0, 1)))
    sobel_y = sobel_x.t()
    laplacian = prediction.new_tensor(((0, 1, 0), (1, -4, 1), (0, 1, 0)))

    def filtered(image, kernel):
        weight = kernel.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        return F.conv2d(image.float(), weight.float(), padding=1, groups=channels)

    pred_dx, target_dx = filtered(prediction, sobel_x), filtered(target, sobel_x)
    pred_dy, target_dy = filtered(prediction, sobel_y), filtered(target, sobel_y)
    pred_lap, target_lap = filtered(prediction, laplacian), filtered(target, laplacian)
    denominator = mask.sum().clamp_min(1.0) * channels
    gradient_loss = (
        ((pred_dx - target_dx).abs() + (pred_dy - target_dy).abs()) * mask
    ).sum() / (2.0 * denominator)
    laplacian_loss = ((pred_lap - target_lap).abs() * mask).sum() / denominator
    return gradient_loss, laplacian_loss


class Model(nn.Module):

    def __init__(
        self,
        vae: AutoencoderKL,
        unet: UNet3DConditionModel,
        appearance_encoder: UNet2DConditionModel,
        condition_encoder: ConditionEncoder,
        batch_size: int,
        latent_refiner: Optional[HandLatentResidualRefiner] = None,
    ):
        super().__init__()

        self.vae = vae
        self.unet = unet
        self.appearance_encoder = appearance_encoder
        self.reference_control_writer = ReferenceAttentionControl(
            appearance_encoder,
            do_classifier_free_guidance=False,
            mode="write",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        self.reference_control_reader = ReferenceAttentionControl(
            self.unet,
            do_classifier_free_guidance=False,
            mode="read",
            batch_size=batch_size,
            fusion_blocks="full",
        )
        self.condition_encoder = condition_encoder
        self.latent_refiner = latent_refiner


def load_latent_refiner(cfg, device, weight_dtype):
    refiner_cfg = cfg.modules.get("latent_refiner_kwargs", None)
    if not refiner_cfg:
        return None
    refiner = HandLatentResidualRefiner(
        **OmegaConf.to_container(refiner_cfg, resolve=True)
    )
    checkpoint = str(cfg.modules.get("latent_refiner", "") or "").strip()
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        missing, unexpected = refiner.load_state_dict(state, strict=False)
        # F/G may deliberately promote a simpler winning ablation to the
        # required superset architecture; only newly enabled modules may miss.
        allowed_new_prefixes = (
            "hamer_conditioner.",
            "spatial_gate.",
            "stage2.",
            "highres_stage.",
        )
        illegal_missing = [
            key for key in missing if not key.startswith(allowed_new_prefixes)
        ]
        if illegal_missing or unexpected:
            raise RuntimeError(
                "Incompatible latent refiner checkpoint: "
                f"missing={illegal_missing}, unexpected={unexpected}"
            )
        logger.info("Loaded latent refiner from %s", checkpoint)
        if missing:
            logger.info("Initialized %d newly enabled refiner tensors", len(missing))
    return refiner.to(device=device, dtype=weight_dtype)


def load_config_file(config_path):
    """Load recursive relative ``base_config`` overlays with cycle detection."""
    config_path = pathlib.Path(config_path).resolve()

    def load_recursive(path, stack):
        path = pathlib.Path(path).resolve()
        if path in stack:
            chain = " -> ".join(str(item) for item in (*stack, path))
            raise ValueError(f"Cyclic base_config chain: {chain}")
        cfg = OmegaConf.load(path)
        base_config = cfg.get("base_config", None)
        if not base_config:
            return cfg
        base_path = pathlib.Path(str(base_config))
        if not base_path.is_absolute():
            base_path = path.parent / base_path
        child = OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))
        del child["base_config"]
        return OmegaConf.merge(load_recursive(base_path, (*stack, path)), child)

    cfg = load_recursive(config_path, ())
    cfg.exp_name = pathlib.Path(config_path).stem
    return cfg


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train/stage_1.yaml")
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    args = parser.parse_args()

    # Load the configuration file
    cfg = load_config_file(args.config)
    max_train_steps_override = os.environ.get("SIGNVIP_MAX_TRAIN_STEPS")
    if max_train_steps_override:
        cfg.solver.max_train_steps = int(max_train_steps_override)
    seed_override = os.environ.get("SIGNVIP_SEED")
    if seed_override:
        cfg.seed = int(seed_override)
    early_patience_steps_override = os.environ.get(
        "SIGNVIP_EARLY_STOP_PATIENCE_STEPS"
    )
    early_min_steps_override = os.environ.get("SIGNVIP_EARLY_STOP_MIN_STEPS")
    if early_patience_steps_override:
        if not cfg.get("early_stopping", None):
            cfg.early_stopping = OmegaConf.create({})
        cfg.early_stopping.enabled = True
        cfg.early_stopping.patience_steps = int(early_patience_steps_override)
    if early_min_steps_override:
        if not cfg.get("early_stopping", None):
            cfg.early_stopping = OmegaConf.create({})
        cfg.early_stopping.min_steps = int(early_min_steps_override)
    resume_override = os.environ.get("SIGNVIP_RESUME_FROM_CHECKPOINT")
    if resume_override:
        cfg.resume_from_checkpoint = resume_override

    # Update cfg with values from args
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    args_dict = vars(args)
    merged_dict = {**cfg_dict, **args_dict}

    # Convert merged_dict back to OmegaConf object
    merged_cfg = OmegaConf.create(merged_dict)

    return merged_cfg


def load_modules(cfg, device, weight_dtype):
    modules_cfg = cfg.modules

    vae = AutoencoderKL.from_pretrained(modules_cfg.vae).to(device, weight_dtype)
    logger.info(f"Loaded VAE from {modules_cfg.vae}.")

    unet = UNet3DConditionModel.from_pretrained_2d(
        modules_cfg.unet_2d,
        unet_additional_kwargs=OmegaConf.to_container(
            modules_cfg.unet_additional_kwargs
        ),
    )
    unet = unet.to(device, weight_dtype)
    logger.info(f"Loaded U-Net from {modules_cfg.unet_2d} (unet_2d)")
    appearance_encoder = AppearanceEncoderModel.from_pretrained(
        modules_cfg.apperance_encoder
    ).to(device)
    logger.info(f"Loaded appearance encoder from {modules_cfg.apperance_encoder}.")

    condition_encoder_image_finetune = bool(
        modules_cfg.get("condition_encoder_image_finetune", True)
    )
    condition_encoder = ConditionEncoder(
        conditioning_channels=3,
        image_finetune=condition_encoder_image_finetune,
        num_conds=2,
        motion_module_type=modules_cfg.unet_additional_kwargs.get(
            "motion_module_type", None
        ),
        motion_module_kwargs=modules_cfg.unet_additional_kwargs.get(
            "motion_module_kwargs", None
        ),
    )
    if modules_cfg.condition_encoder:
        condition_state = torch.load(modules_cfg.condition_encoder, map_location="cpu")
        missing, unexpected = condition_encoder.load_state_dict(
            condition_state, strict=condition_encoder_image_finetune
        )
        if unexpected:
            raise RuntimeError(
                "Unexpected condition-encoder checkpoint keys: "
                + ", ".join(unexpected)
            )
        illegal_missing = [
            key for key in missing if not key.startswith("motion_modules.")
        ]
        if illegal_missing:
            raise RuntimeError(
                "Condition-encoder checkpoint is missing non-temporal keys: "
                + ", ".join(illegal_missing)
            )
        logger.info(f"Loaded condition encoder from {modules_cfg.condition_encoder}.")
        if missing:
            logger.info(
                "Initialized %d new condition-encoder temporal tensors.", len(missing)
            )
    condition_motion_path = modules_cfg.get("condition_encoder_motion", None)
    if condition_motion_path:
        motion_state = torch.load(condition_motion_path, map_location="cpu")
        missing, unexpected = condition_encoder.load_state_dict(
            motion_state, strict=False
        )
        if unexpected:
            raise RuntimeError(
                "Unexpected condition-motion checkpoint keys: "
                + ", ".join(unexpected)
            )
        logger.info(f"Loaded condition motion modules from {condition_motion_path}.")
    condition_encoder.to(device)

    motion_module_path = modules_cfg.get("mm", None)
    if motion_module_path:
        motion_state = torch.load(motion_module_path, map_location="cpu")
        if isinstance(motion_state, dict) and "state_dict" in motion_state:
            motion_state = motion_state["state_dict"]
        normalized_motion_state = {}
        for key, value in motion_state.items():
            normalized_key = key
            if normalized_key.startswith("module."):
                normalized_key = normalized_key[len("module.") :]
            if normalized_key.startswith("unet."):
                normalized_key = normalized_key[len("unet.") :]
            if "motion_modules" in normalized_key:
                normalized_motion_state[normalized_key] = value
        if not normalized_motion_state:
            raise RuntimeError(
                f"No motion_modules tensors found in checkpoint: {motion_module_path}"
            )
        _, unexpected = unet.load_state_dict(normalized_motion_state, strict=False)
        if unexpected:
            raise RuntimeError(
                "Unexpected U-Net motion checkpoint keys: " + ", ".join(unexpected)
            )
        logger.info(
            "Loaded %d U-Net motion tensors from %s.",
            len(normalized_motion_state),
            motion_module_path,
        )

    if modules_cfg.unet:
        state = torch.load(modules_cfg.unet, map_location="cpu")
        missing, unexpected = unet.load_state_dict(state, strict=False)
        conditioning_prefixes = (
            "hamer_reinjector.geometry_",
            "hamer_reinjector.gamma_geometry",
            "hamer_reinjector.hand_edge_",
            "hamer_reinjector.gamma_hand_edge",
            "hamer_reinjector.pixel_geometry_",
            "hamer_reinjector.gamma_pixel_geometry",
            "hamer_reinjector.wavelet_edge_",
            "hamer_reinjector.gamma_wavelet_edge",
            "hamer_reinjector.c518_",
        )
        illegal_missing = [
            key for key in missing if not key.startswith(conditioning_prefixes)
        ]
        if illegal_missing or unexpected:
            raise RuntimeError(
                "Incompatible full U-Net checkpoint. Missing: "
                + ", ".join(illegal_missing)
                + "; unexpected: "
                + ", ".join(unexpected)
            )
        unet.to(device, weight_dtype)
        logger.info(f"Loaded full UNET from {modules_cfg.unet}.")
        if missing:
            logger.info("Initialized %d new C5.17 geometry tensors.", len(missing))
    hamer_reinjector_path = modules_cfg.get("hamer_reinjector", None)
    if hamer_reinjector_path:
        if unet.hamer_reinjector is None:
            raise RuntimeError(
                "A HAMER reinjector checkpoint was provided but the U-Net config "
                "does not instantiate hamer_reinjector"
            )
        state = torch.load(hamer_reinjector_path, map_location="cpu")
        missing, unexpected = unet.hamer_reinjector.load_state_dict(
            state, strict=False
        )
        illegal_missing = [
            key
            for key in missing
            if not key.startswith(
                (
                    "geometry_",
                    "gamma_geometry",
                    "hand_edge_",
                    "gamma_hand_edge",
                    "pixel_geometry_",
                    "gamma_pixel_geometry",
                    "wavelet_edge_",
                    "gamma_wavelet_edge",
                    "c518_",
                )
            )
        ]
        if illegal_missing or unexpected:
            raise RuntimeError(
                "Incompatible HAMER reinjector checkpoint. Missing: "
                + ", ".join(illegal_missing)
                + "; unexpected: "
                + ", ".join(unexpected)
            )
        logger.info(f"Loaded HAMER reinjector from {hamer_reinjector_path}.")
        if missing:
            logger.info("Initialized %d new C5.17 reinjector tensors.", len(missing))

    scheduler_kwargs = OmegaConf.to_container(cfg.noise_scheduler_kwargs)
    if cfg.enable_zero_snr:
        scheduler_kwargs.update(
            rescale_betas_zero_snr=True,
            timestep_spacing="trailing",
            prediction_type="v_prediction",
        )
    val_noise_scheduler = DDIMScheduler(**scheduler_kwargs)
    scheduler_kwargs.update({"beta_schedule": "scaled_linear"})
    noise_scheduler = DDIMScheduler.from_pretrained(
        cfg.modules.scheduler,
        **scheduler_kwargs,
    )

    empty_text_emb_path = str(cfg.modules.get("empty_text_emb", "") or "").strip()
    if empty_text_emb_path:
        empty_text_emb = torch.load(empty_text_emb_path, map_location="cpu").to(
            device, weight_dtype
        )
        logger.info(f"Loaded empty-text embedding from {empty_text_emb_path}.")
    else:
        tokenizer_path = str(cfg.modules.tokenizer)
        text_encoder_path = str(cfg.modules.text_encoder)
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path)
        text_encoder = CLIPTextModel.from_pretrained(text_encoder_path).to(
            device, weight_dtype
        )
        tokens = tokenizer(
            "",
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            empty_text_emb = text_encoder(tokens.input_ids.to(device))[0]
        del text_encoder, tokenizer
        logger.info(
            "Built empty-text embedding from the configured SD1.5 tokenizer "
            "and text encoder."
        )

    return (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        noise_scheduler,
        val_noise_scheduler,
        empty_text_emb,
    )


def count_params(model: Model):
    logger.info("***** Parameters Counting *****")

    # VAE
    params = list(model.vae.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(f"  VAE: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)")

    # unet
    params = list(model.unet.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  U-Net (unet_2d + mm): {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
    )

    # appearance_encoder
    params = list(model.appearance_encoder.parameters())
    num_params, num_trainable_params = get_num_params(params)
    logger.info(
        f"  Appearance Encoder: {num_params:.3f} M (trainable: {num_trainable_params:.3f}M)"
    )


def compute_snr(noisy_scheduler, timesteps):
    alphas_cumprod = noisy_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    snr = (alpha / sigma) ** 2
    return snr


def save_model(model, cfg, model_path, is_main_process, weight_dtype):
    if getattr(model, "latent_refiner", None) is not None and is_main_process:
        refiner_path = os.path.join(model_path, "latent_refiner")
        pathlib.Path(refiner_path).mkdir(parents=True, exist_ok=True)
        torch.save(
            model.latent_refiner.state_dict(),
            os.path.join(refiner_path, "model.bin"),
        )
        with open(
            os.path.join(refiner_path, "metadata.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(
                OmegaConf.to_container(
                    cfg.modules.latent_refiner_kwargs, resolve=True
                ),
                handle,
                indent=2,
            )
        logger.info("Saved latent refiner to %s", refiner_path)
    if (
        cfg.grad.vae
        or bool(cfg.grad.get("vae_decoder", False))
        or bool(cfg.get("save_frozen_vae", False))
    ):
        model.vae.save_pretrained(
            os.path.join(model_path, "vae"),
            is_main_process=is_main_process,
            safe_serialization=True,
        )
        logger.info(f"Saved VAE to {os.path.join(model_path, 'vae')}.")

    save_frozen_modules = bool(cfg.get("save_frozen_modules", False))
    if cfg.grad.appearance_encoder or save_frozen_modules:
        model.appearance_encoder.save_pretrained(
            os.path.join(model_path, "appearance_encoder"),
            is_main_process=is_main_process,
            safe_serialization=True,
        )
        logger.info(
            f"Saved appearance encoder to {os.path.join(model_path, 'appearance_encoder')}."
        )
    if cfg.grad.condition_encoder or save_frozen_modules:
        pathlib.Path(os.path.join(model_path, "condition_encoder")).mkdir(
            parents=True, exist_ok=True
        )
        torch.save(
            model.condition_encoder.state_dict(),
            os.path.join(model_path, "condition_encoder/model.bin"),
        )
        logger.info(
            f"Saved condition encoder to {os.path.join(model_path, 'condition_encoder')}."
        )
    if (cfg.grad.unet_2d) and is_main_process:
        pathlib.Path(os.path.join(model_path, "unet")).mkdir(
            parents=True, exist_ok=True
        )
        torch.save(model.unet.state_dict(), os.path.join(model_path, "unet/model.bin"))
        logger.info(f"Saved unet to {os.path.join(model_path, 'unet')}.")
    elif cfg.grad.get("selected_unet_name_patterns", None) and is_main_process:
        pathlib.Path(os.path.join(model_path, "unet")).mkdir(
            parents=True, exist_ok=True
        )
        patterns = list(cfg.grad.selected_unet_name_patterns)
        motion_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.unet.state_dict().items()
            if any(pattern in name for pattern in patterns)
        }
        torch.save(motion_state, os.path.join(model_path, "unet/model.bin"))
        logger.info(
            "Saved %d selected U-Net tensors to %s.",
            len(motion_state),
            os.path.join(model_path, "unet/model.bin"),
        )
    if (
        cfg.grad.get("selected_condition_encoder_name_patterns", None)
        and is_main_process
    ):
        pathlib.Path(os.path.join(model_path, "condition_encoder_motion")).mkdir(
            parents=True, exist_ok=True
        )
        patterns = list(cfg.grad.selected_condition_encoder_name_patterns)
        motion_state = {
            name: parameter.detach().cpu()
            for name, parameter in model.condition_encoder.state_dict().items()
            if any(pattern in name for pattern in patterns)
        }
        torch.save(
            motion_state,
            os.path.join(model_path, "condition_encoder_motion/model.bin"),
        )
        logger.info(
            "Saved %d selected condition-encoder tensors to %s.",
            len(motion_state),
            os.path.join(model_path, "condition_encoder_motion/model.bin"),
        )
    if getattr(model.unet, "hamer_reinjector", None) is not None and is_main_process:
        reinjector_path = os.path.join(model_path, "hamer_reinjector")
        pathlib.Path(reinjector_path).mkdir(parents=True, exist_ok=True)
        torch.save(
            model.unet.hamer_reinjector.state_dict(),
            os.path.join(reinjector_path, "model.bin"),
        )
        with open(os.path.join(reinjector_path, "metadata.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "up_second_highest_index": model.unet.hamer_up_second_highest_index,
                    "up_highest_index": model.unet.hamer_up_highest_index,
                    "up16_block_index": model.unet.hamer_up16_block_index,
                    "hamer_up16_injection_location": (
                        model.unet.hamer_reinjector.hamer_up16_injection_location
                    ),
                    "site_enabled": model.unet.hamer_reinjector.site_enabled,
                    "geometry_enabled": model.unet.hamer_reinjector.geometry_enabled,
                    "geometry_site": model.unet.hamer_reinjector.geometry_site,
                    "geometry_cross_attention": (
                        model.unet.hamer_reinjector.geometry_cross_attention_enabled
                    ),
                    "geometry_timestep_gate": (
                        model.unet.hamer_reinjector.geometry_timestep_gate_enabled
                    ),
                    "hand_edge_enabled": model.unet.hamer_reinjector.hand_edge_enabled,
                    "hand_edge_site": model.unet.hamer_reinjector.hand_edge_site,
                    "hand_edge_residual_scale": (
                        model.unet.hamer_reinjector.hand_edge_residual_scale
                    ),
                    "pixel_geometry_enabled": (
                        model.unet.hamer_reinjector.pixel_geometry_enabled
                    ),
                    "pixel_geometry_site": (
                        model.unet.hamer_reinjector.pixel_geometry_site
                    ),
                    "wavelet_edge_enabled": (
                        model.unet.hamer_reinjector.wavelet_edge_enabled
                    ),
                    "wavelet_edge_site": (
                        model.unet.hamer_reinjector.wavelet_edge_site
                    ),
                    "c518_multiscale_enabled": (
                        model.unet.hamer_reinjector.c518_multiscale_enabled
                    ),
                    "c518_separate_hand_roi": (
                        model.unet.hamer_reinjector.c518_separate_hand_roi
                    ),
                    "c518_local_cross_attention": (
                        model.unet.hamer_reinjector.c518_local_cross_attention
                    ),
                    "parameters": model.unet.hamer_reinjector.extra_parameter_count(),
                },
                handle,
                indent=2,
            )
        logger.info(f"Saved HAMER reinjector to {reinjector_path}.")


@torch.no_grad()
def log_valid(
    cfg,
    accelerator,
    valid_loader,
    model,
    empty_text_emb,
    val_noise_scheduler,
    device,
    weight_dtype,
    global_step,
    out_path,
):
    model.eval()

    scheduler = val_noise_scheduler

    pipeline = SignViPStaticPipeline(
        vae=model.vae,
        denoising_unet=model.unet,
        scheduler=scheduler,
        empty_text_emb=empty_text_emb,
        appearance_encoder=model.appearance_encoder,
        condition_encoder=model.condition_encoder,
        latent_refiner=getattr(model, "latent_refiner", None),
    ).to(device=device)

    # 用于存储所有生成和真实图像
    all_generated = []
    all_real = []
    all_hand_masks = []
    validation_has_hand_masks = None

    # 遍历验证集
    validation_seed = int(cfg.validation_data.get("seed", cfg.get("seed", 42)))
    for batch_index, batch in enumerate(
        tqdm(valid_loader, total=len(valid_loader), desc="Validating")
    ):
        ref_frame = batch["ref_frame"].to(device)
        tgt_frames = batch["tgt_frames"].to(device)  # 保持原始范围 [-1, 1]
        tgt_frames = (tgt_frames / 2 + 0.5).clamp(0, 1)  # 转换到 [0, 1] 范围
        tgt_sk_frames = batch["tgt_sk_frames"].to(device)
        tgt_hamer_frames = batch["tgt_hamer_frames"].to(device)
        hand_masks = batch.get("hand_mask")
        if hand_masks is None:
            if validation_has_hand_masks is True:
                raise RuntimeError(
                    "Validation batches inconsistently provide hand_mask"
                )
            validation_has_hand_masks = False
            if batch_index == 0:
                logger.warning(
                    "Validation dataset has no hand_mask; reporting global SSIM "
                    "only and leaving hand/non-hand SSIM as NaN."
                )
        else:
            if validation_has_hand_masks is False:
                raise RuntimeError(
                    "Validation batches inconsistently provide hand_mask"
                )
            validation_has_hand_masks = True
            hand_masks = hand_masks.to(device=device, dtype=torch.float32)
        left_hand_roi_mask = batch.get("left_hand_roi_mask")
        right_hand_roi_mask = batch.get("right_hand_roi_mask")
        if left_hand_roi_mask is not None:
            left_hand_roi_mask = left_hand_roi_mask.to(device=device, dtype=torch.float32)
            right_hand_roi_mask = right_hand_roi_mask.to(device=device, dtype=torch.float32)

        # convert from bcfhw to (b f) c h w
        tgt_frames = rearrange(tgt_frames, "b c f h w -> (b f) c h w")
        if hand_masks is not None:
            hand_masks = rearrange(hand_masks, "b 1 f h w -> (b f) 1 h w")

        # 生成图像
        with accelerator.autocast():
            generated_image = pipeline(
                prompt=cfg.validation_data.prompt,
                reference_image=ref_frame,
                sk_image=tgt_sk_frames,
                hamer_image=tgt_hamer_frames,
                left_hand_roi_mask=left_hand_roi_mask,
                right_hand_roi_mask=right_hand_roi_mask,
                width=cfg.dataset.frame_size[1],
                height=cfg.dataset.frame_size[0],
                num_inference_steps=cfg.validation_data.num_inference_steps,
                guidance_scale=cfg.validation_data.guidance_scale,
                batch_size=ref_frame.shape[0],
                generator=torch.Generator(device="cpu").manual_seed(
                    validation_seed + batch_index
                ),
            )
        generated_image = (
            rearrange(generated_image, "b c f h w -> (b f) c h w")
            .contiguous()
            .to(device)
        )
        real_image = tgt_frames.contiguous().to(device)

        if hand_masks is None:
            generated_image, real_image = accelerator.gather_for_metrics(
                (generated_image, real_image)
            )
        else:
            generated_image, real_image, hand_masks = accelerator.gather_for_metrics(
                (generated_image, real_image, hand_masks)
            )

        # 收集生成和真实图像
        all_generated.append(generated_image)
        all_real.append(real_image)
        if hand_masks is not None:
            all_hand_masks.append(hand_masks)

        # 可以只处理部分批次以节省时间
        if len(all_generated) >= cfg.validation_data.get("max_eval_batches", 12):
            break

    # 合并所有批次
    all_generated = torch.cat(all_generated, dim=0)  # [N, C, H, W]
    all_real = torch.cat(all_real, dim=0)  # [N, C, H, W]
    all_hand_masks = (
        torch.cat(all_hand_masks, dim=0) if all_hand_masks else None
    )  # [N, 1, H, W] when available

    validation_metrics = {
        "ssim": 0.0,
        "hand_ssim": float("nan"),
        "nonhand_ssim": float("nan"),
        "score": 0.0,
    }
    if accelerator.is_main_process:
        # 计算 SSIM
        ssim = calc_ssim(
            all_real,  # [N, C, H, W]
            all_generated,  # [N, C, H, W]
            only_final=True,
        )
        hand_ssim = float("nan")
        nonhand_ssim = float("nan")
        if all_hand_masks is not None:
            hand_ssim, nonhand_ssim = calculate_hand_nonhand_ssim(
                all_real, all_generated, all_hand_masks
            )
        best_cfg = cfg.get("best_checkpoint", {})
        best_metric_name = str(best_cfg.get("metric", "global_ssim"))
        if best_metric_name == "balanced_hand_nonhand_ssim":
            if all_hand_masks is None:
                raise ValueError(
                    "balanced_hand_nonhand_ssim requires validation hand_mask data"
                )
            hand_weight = float(best_cfg.get("hand_weight", 0.5))
            nonhand_weight = float(best_cfg.get("nonhand_weight", 0.5))
            if not math.isclose(hand_weight + nonhand_weight, 1.0, abs_tol=1e-8):
                raise ValueError("best_checkpoint hand/nonhand weights must sum to 1")
            score = hand_weight * hand_ssim + nonhand_weight * nonhand_ssim
        elif best_metric_name == "global_ssim":
            score = ssim
        else:
            raise ValueError(
                "best_checkpoint.metric must be global_ssim or "
                "balanced_hand_nonhand_ssim"
            )
        validation_metrics = {
            "ssim": float(ssim),
            "hand_ssim": float(hand_ssim),
            "nonhand_ssim": float(nonhand_ssim),
            "score": float(score),
        }

        # 记录 SSIM 分数
        accelerator.log(
            {
                "validation/ssim": ssim,
                "validation/hand_ssim": hand_ssim,
                "validation/nonhand_ssim": nonhand_ssim,
                "validation/best_score": score,
            },
            step=global_step,
        )

        # 保存多个对比图片，并拼接真实图像、生成图像、骨架和HAMER
        to_pil = transforms.ToPILImage()

        # 确保保存多张图片
        num_samples = min(5, all_real.shape[0])  # 保存前5张图片或更少
        for i in range(num_samples):
            comparison_image = torch.cat(
                [
                    all_real[i : i + 1],  # 真实图像 (1,C,H,W)
                    all_generated[i : i + 1],  # 生成图像 (1,C,H,W)
                ],
                dim=3,  # 在宽度维度上拼接
            )

            # 转换为PIL图像并保存
            comparison_pil = to_pil(comparison_image[0])
            comparison_pil.save(out_path + f"/{global_step}_comparison_{i+1}.png")

    return validation_metrics


def main():
    cfg = parse_config()
    experiment_index = datetime.datetime.now().strftime("%Y%m%d-%H%M")
    workspace_root = os.environ.get("SIGNVIP_WORKSPACE_ROOT", "./workspace")
    workspace_dir = os.environ.get("SIGNVIP_RUN_DIR") or (
        str(cfg.resume_from_checkpoint)
        if cfg.resume_from_checkpoint
        else os.path.join(workspace_root, cfg.exp_name, experiment_index)
    )
    pathlib.Path(workspace_dir).mkdir(parents=True, exist_ok=True)
    config = accelerate.utils.ProjectConfiguration(
        project_dir=".", logging_dir=workspace_dir + "/log"
    )
    # save config under workspace_dir
    with open(os.path.join(workspace_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    accelerator = accelerate.Accelerator(
        log_with=cfg.report_to,
        project_config=config,
        gradient_accumulation_steps=cfg.solver.gradient_accumulation_steps,
    )
    accelerator.init_trackers("signvip")
    device = accelerator.device

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if cfg.seed is not None:
        seed_everything(cfg.seed)

    if cfg.weight_dtype == "fp16":
        weight_dtype = torch.float16
    elif cfg.weight_dtype == "fp32":
        weight_dtype = torch.float32
    elif cfg.weight_dtype == "bf16":
        weight_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Do not support weight dtype: {cfg.weight_dtype} during training!"
        )

    (
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        noisy_scheduler,
        val_noise_scheduler,
        empty_text_emb,
    ) = load_modules(cfg, device, weight_dtype)
    latent_refiner = load_latent_refiner(cfg, device, weight_dtype)

    train_full_vae = bool(cfg.grad.vae)
    train_vae_decoder = bool(cfg.grad.get("vae_decoder", False))
    if train_full_vae and train_vae_decoder:
        raise ValueError("grad.vae and grad.vae_decoder cannot both be enabled")
    vae.requires_grad_(train_full_vae)
    if train_vae_decoder:
        # Diffusers AutoencoderKL applies post_quant_conv immediately before
        # decoder, so it belongs to the decoder-side trainable path.
        vae.requires_grad_(False)
        vae.post_quant_conv.requires_grad_(True)
        vae.decoder.requires_grad_(True)
        logger.info("Training VAE decoder only: post_quant_conv + decoder")
    unet.requires_grad_(cfg.grad.unet_2d)
    appearance_encoder.requires_grad_(cfg.grad.appearance_encoder)
    condition_encoder.requires_grad_(cfg.grad.condition_encoder)
    if latent_refiner is not None:
        latent_refiner.requires_grad_(bool(cfg.grad.get("latent_refiner", True)))
    if unet.hamer_reinjector is not None:
        train_reinjector = bool(cfg.grad.get("hamer_reinjector", False))
        unet.hamer_reinjector.requires_grad_(train_reinjector)
        # Keep inactive ablation sites out of the optimizer entirely so each
        # variant changes only its requested injection location.
        for site in unet.hamer_reinjector.SITES:
            site_trainable = bool(
                train_reinjector
                and unet.hamer_reinjector.enabled
                and unet.hamer_reinjector.site_enabled[site]
            )
            unet.hamer_reinjector.adapters[site].requires_grad_(site_trainable)
            unet.hamer_reinjector.gamma(site).requires_grad_(site_trainable)
        unet.hamer_reinjector.set_geometry_trainable(
            encoder=bool(cfg.grad.get("hand_geometry_encoder", False)),
            fusion=bool(cfg.grad.get("hand_geometry_reinjector", False)),
        )
        unet.hamer_reinjector.set_hand_edge_trainable(
            encoder=bool(cfg.grad.get("hand_edge_encoder", False)),
            fusion=bool(cfg.grad.get("hand_edge_reinjector", False)),
        )
        unet.hamer_reinjector.set_pixel_geometry_trainable(
            encoder=bool(cfg.grad.get("pixel_geometry_encoder", False)),
            fusion=bool(cfg.grad.get("pixel_geometry_reinjector", False)),
        )
        unet.hamer_reinjector.set_wavelet_edge_trainable(
            encoder=bool(cfg.grad.get("wavelet_edge_encoder", False)),
            fusion=bool(cfg.grad.get("wavelet_edge_reinjector", False)),
        )
        logger.info(
            "HAMER reinjection location(s): %s",
            ", ".join(unet.hamer_reinjector.SITES),
        )
    selected_unet_modules = list(cfg.grad.get("selected_unet_modules", []) or [])
    for module_path in selected_unet_modules:
        try:
            unet.get_submodule(str(module_path)).requires_grad_(True)
        except AttributeError as exc:
            raise ValueError(f"Unknown selected U-Net module: {module_path}") from exc
    selected_unet_name_patterns = list(
        cfg.grad.get("selected_unet_name_patterns", []) or []
    )
    for name, parameter in unet.named_parameters():
        if any(pattern in name for pattern in selected_unet_name_patterns):
            parameter.requires_grad_(True)
    selected_condition_name_patterns = list(
        cfg.grad.get("selected_condition_encoder_name_patterns", []) or []
    )
    for name, parameter in condition_encoder.named_parameters():
        if any(pattern in name for pattern in selected_condition_name_patterns):
            parameter.requires_grad_(True)

    # Accelerate's FP16 GradScaler can only unscale FP32 gradients.  The U-Net
    # is loaded in weight_dtype to keep VRAM low, which also casts newly added
    # adapters to FP16.  Keep frozen weights in FP16, but maintain every
    # trainable master parameter in FP32 before the optimizer is constructed.
    # CUDA autocast still executes the eligible forward kernels in FP16.
    fp32_trainable_count = 0
    for parameter in itertools.chain(
        unet.parameters(),
        appearance_encoder.parameters(),
        condition_encoder.parameters(),
        vae.parameters(),
        latent_refiner.parameters() if latent_refiner is not None else (),
    ):
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()
            fp32_trainable_count += parameter.numel()
    logger.info(
        "FP32 trainable master parameters: %.3f M (frozen weights retain %s)",
        fp32_trainable_count / 1e6,
        cfg.weight_dtype,
    )

    perceptual_cfg = cfg.get("perceptual_loss", {})
    perceptual_enabled = bool(perceptual_cfg.get("enabled", False))
    region_cfg = cfg.get("region_x0_loss", {})
    region_enabled = bool(region_cfg.get("enabled", False))
    global_x0_cfg = cfg.get("global_x0_loss", {})
    global_x0_enabled = bool(global_x0_cfg.get("enabled", False))
    hand_latent_x0_cfg = cfg.get("hand_latent_x0_loss", {})
    hand_latent_x0_enabled = bool(hand_latent_x0_cfg.get("enabled", False))
    hand_frequency_cfg = cfg.get("hand_frequency_loss", {})
    hand_frequency_enabled = bool(hand_frequency_cfg.get("enabled", False))
    refiner_training_cfg = cfg.get("latent_refiner_training", {})
    direct_refiner_enabled = bool(
        refiner_training_cfg.get("direct_residual_enabled", False)
    )
    decoder_refiner_enabled = bool(
        refiner_training_cfg.get("decoder_hand_enabled", False)
    )
    clip_model = None
    dino_model = None
    if perceptual_enabled:
        clip_model = load_clip_model(
            str(perceptual_cfg.clip_model), device, weight_dtype
        )
        dino_model = load_dino_model(
            str(perceptual_cfg.dino_model), device, weight_dtype
        )
        logger.info(
            "Enabled perceptual loss: clip=%s (weight=%s), dino=%s (weight=%s)",
            perceptual_cfg.clip_model,
            perceptual_cfg.clip_weight,
            perceptual_cfg.dino_model,
            perceptual_cfg.dino_weight,
        )
    if region_enabled:
        local_perceptual_weights = (
            float(region_cfg.get("hand_clip_weight", 0.0)),
            float(region_cfg.get("hand_dino_weight", 0.0)),
            float(region_cfg.get("face_clip_weight", 0.0)),
            float(region_cfg.get("face_dino_weight", 0.0)),
        )
        if any(local_perceptual_weights) and not perceptual_enabled:
            raise ValueError(
                "region_x0_loss local CLIP/DINO weights require perceptual_loss.enabled=true"
            )
        logger.info(
            "Enabled region-aware x0 loss: reconstruction=%s, hand_weight=%s, "
            "face_weight=%s, local perceptual weights=%s",
            region_cfg.get("reconstruction", "charbonnier"),
            region_cfg.get("hand_reconstruction_weight", 0.0),
            region_cfg.get("face_reconstruction_weight", 0.0),
            local_perceptual_weights,
        )
    if global_x0_enabled:
        logger.info(
            "Enabled global x0 RGB loss: reconstruction=%s, weight=%s",
            global_x0_cfg.get("reconstruction", "mse"),
            global_x0_cfg.get("weight", 1.0),
        )
    if hand_latent_x0_enabled:
        logger.info(
            "Enabled normalized hand-latent x0 loss: weight=%s, dilation=%s, "
            "ring weights=(%s, %s)",
            hand_latent_x0_cfg.get("weight", 0.25),
            hand_latent_x0_cfg.get("dilation_radius", 2),
            hand_latent_x0_cfg.get("ring1_weight", 0.5),
            hand_latent_x0_cfg.get("ring2_weight", 0.25),
        )
    if hand_frequency_enabled:
        logger.info(
            "Enabled hand-frequency RGB loss: gradient_weight=%s, laplacian_weight=%s",
            hand_frequency_cfg.get("gradient_weight", 0.05),
            hand_frequency_cfg.get("laplacian_weight", 0.05),
        )

    model = Model(
        vae,
        unet,
        appearance_encoder,
        condition_encoder,
        cfg.dataloader.batch_size,
        latent_refiner=latent_refiner,
    ).to(device)

    if cfg.gradient_checkpointing:
        model.unet.enable_gradient_checkpointing()
        model.appearance_encoder.enable_gradient_checkpointing()

    if cfg.solver.optimizer.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            # C7 can reuse an existing bitsandbytes install without exposing
            # that environment's incompatible flash-attn/torch packages during
            # the earlier Diffusers and Transformers imports.
            import sys

            fallback_site = os.environ.get("C7_BNB_SITE_PACKAGES", "").strip()
            if not fallback_site:
                raise ImportError(
                    "Please install bitsandbytes or set C7_BNB_SITE_PACKAGES."
                )
            sys.path.append(fallback_site)
            try:
                import bitsandbytes as bnb
            finally:
                sys.path.remove(fallback_site)
        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    trainable_named_params = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable_named_params:
        raise RuntimeError("No trainable parameters are enabled by this configuration")
    trainable_params = [parameter for _, parameter in trainable_named_params]
    lr_new = float(
        cfg.solver.optimizer.get("lr_new", cfg.solver.optimizer.learning_rate)
    )
    lr_base = float(
        cfg.solver.optimizer.get("lr_base", cfg.solver.optimizer.learning_rate)
    )
    new_adapter_params = []
    new_gate_params = []
    base_params = []
    for name, parameter in trainable_named_params:
        if name.startswith("unet.hamer_reinjector."):
            if ".gamma_" in name:
                new_gate_params.append(parameter)
            else:
                new_adapter_params.append(parameter)
        else:
            base_params.append(parameter)
    optimizer_groups = []
    if new_adapter_params:
        optimizer_groups.append(
            {"params": new_adapter_params, "lr": lr_new, "name": "hamer_adapters"}
        )
    if new_gate_params:
        optimizer_groups.append(
            {
                "params": new_gate_params,
                "lr": lr_new,
                "weight_decay": 0.0,
                "name": "hamer_gates",
            }
        )
    if base_params:
        optimizer_groups.append(
            {"params": base_params, "lr": lr_base, "name": "selected_pretrained"}
        )
    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_params)
    new_count = (
        model.unet.hamer_reinjector.extra_parameter_count()
        if model.unet.hamer_reinjector is not None
        else 0
    )
    trainable_new_count = sum(
        parameter.numel() for parameter in new_adapter_params + new_gate_params
    )
    logger.info(f"New HAMER reinjection params: {new_count / 1e6:.3f} M")
    logger.info(
        f"Trainable HAMER reinjection params: {trainable_new_count / 1e6:.3f} M"
    )
    logger.info(f"Trainable params: {trainable_count / 1e6:.3f} M")
    logger.info(f"Frozen params: {(total_params - trainable_count) / 1e6:.3f} M")
    optimizer = optimizer_cls(
        optimizer_groups,
        lr=cfg.solver.optimizer.learning_rate,
        betas=(cfg.solver.optimizer.adam_beta1, cfg.solver.optimizer.adam_beta2),
        weight_decay=cfg.solver.optimizer.adam_weight_decay,
        eps=cfg.solver.optimizer.adam_epsilon,
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_scheduler.lr_warmup_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
        num_training_steps=cfg.solver.max_train_steps
        * cfg.solver.gradient_accumulation_steps
        * accelerator.num_processes,
    )

    dataset_kwargs = dict(
        frame_size=cfg.dataset.frame_size,
        frame_scale=cfg.dataset.frame_scale,
        frame_ratio=cfg.dataset.frame_ratio,
        sample_rate=cfg.dataset.sample_rate,
        num_frames=cfg.dataset.num_frames,
        ref_margin=cfg.dataset.ref_margin,
        uncond_ratio=cfg.dataset.uncond_ratio,
        mask_ratio=cfg.dataset.mask_ratio,
        mask_thershold=cfg.dataset.mask_thershold,
        skip_ratio=cfg.dataset.skip_ratio,
        sk_mask_ratio=cfg.dataset.sk_mask_ratio,
        hamer_mask_ratio=cfg.dataset.hamer_mask_ratio,
        both_mask_ratio=cfg.dataset.both_mask_ratio,
        hand_mask_roots=cfg.dataset.get("hand_mask_roots", None),
        hand_mask_strict=cfg.dataset.get("hand_mask_strict", True),
        fusion_keypoint_root=cfg.dataset.get("fusion_keypoint_root", None),
        fusion_keypoint_strict=cfg.dataset.get("fusion_keypoint_strict", True),
        face_score_threshold=cfg.dataset.get("face_score_threshold", 0.30),
        face_margin=cfg.dataset.get("face_margin", 0.08),
        detail_mask_cache=cfg.dataset.get("detail_mask_cache", None),
        detail_mask_split=cfg.dataset.get("detail_mask_split", "train"),
        detail_mask_strict=cfg.dataset.get("detail_mask_strict", True),
        pose_keypoint_cache=cfg.dataset.get("pose_keypoint_cache", None),
        pose_keypoint_split=cfg.dataset.get("pose_keypoint_split", "train"),
        pose_keypoint_strict=cfg.dataset.get("pose_keypoint_strict", True),
        hand_roi_score_threshold=cfg.dataset.get("hand_roi_score_threshold", 0.30),
        hand_roi_margin=cfg.dataset.get("hand_roi_margin", 0.20),
    )
    use_hdf5 = str(cfg.dataset.get("format", "video")).lower() == "hdf5"
    if use_hdf5:
        dataset = signdatasets.HDF5SignLangVideoDataset(
            roots=cfg.dataset.roots,
            hamer_roots=cfg.dataset.hamer_roots,
            **dataset_kwargs,
        )
    else:
        dataset = signdatasets.SignLangVideoDataset(
            roots=cfg.dataset.roots,
            sk_roots=cfg.dataset.sk_roots,
            hamer_roots=cfg.dataset.hamer_roots,
            meta_paths=cfg.dataset.meta_paths,
            **dataset_kwargs,
        )

    # 添加验证数据集
    valid_kwargs = dict(dataset_kwargs)
    valid_kwargs.update(
        sample_rate=4,
        num_frames=5,
        random_sample=False,
        mask_ratio=0,
        sk_mask_ratio=0,
        hamer_mask_ratio=0,
        both_mask_ratio=0,
        detail_mask_split="dev",
        hand_mask_roots=cfg.validation_data.get(
            "hand_mask_root", cfg.dataset.get("hand_mask_roots", None)
        ),
        pose_keypoint_split="dev",
    )
    if use_hdf5:
        valid_dataset = signdatasets.HDF5SignLangVideoDataset(
            roots=cfg.validation_data.root,
            hamer_roots=cfg.validation_data.hamer_root,
            **valid_kwargs,
        )
    else:
        valid_dataset = signdatasets.SignLangVideoDataset(
            roots=cfg.dataset.roots[0],  # 使用第一个数据根目录
            sk_roots=cfg.dataset.sk_roots[0],
            hamer_roots=cfg.dataset.hamer_roots[0],
            meta_paths=cfg.validation_data.meta_paths,  # 使用验证集的元数据路径
            **valid_kwargs,
        )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.dataloader.batch_size,
        shuffle=True,
        num_workers=cfg.dataloader.num_workers,
    )

    # 添加验证数据加载器
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=1,  # 验证时使用较小的批次大小
        shuffle=False,
        num_workers=1,
    )

    count_params(model)

    # 更新 accelerator.prepare 调用
    (model, optimizer, dataloader, valid_loader, lr_scheduler) = accelerator.prepare(
        model, optimizer, dataloader, valid_loader, lr_scheduler
    )

    num_steps_per_epoch = math.ceil(
        len(dataloader) / cfg.solver.gradient_accumulation_steps
    )
    num_epochs = math.ceil(cfg.solver.max_train_steps / num_steps_per_epoch)

    total_batch_size = (
        cfg.dataloader.batch_size
        * accelerator.num_processes
        * cfg.solver.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(dataset)}")
    logger.info(f"  Num Epochs = {num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.dataloader.batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(
        f"  Gradient Accumulation steps = {cfg.solver.gradient_accumulation_steps}"
    )
    logger.info(f"  Total optimization steps = {cfg.solver.max_train_steps}")

    global_step = 0
    start_epoch = 0

    if cfg.resume_from_checkpoint:
        if cfg.resume_from_checkpoint != "latest":
            resume_dir = cfg.resume_from_checkpoint
        else:
            resume_dir = workspace_dir

        dirs = os.listdir(resume_dir)
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        required_checkpoint_files = [
            "model.safetensors",
            "optimizer.bin",
            "scheduler.bin",
            "random_states_0.pkl",
        ]
        if accelerator.scaler is not None:
            required_checkpoint_files.append("scaler.pt")
        dirs = [
            directory
            for directory in dirs
            if all(
                os.path.isfile(os.path.join(resume_dir, directory, filename))
                and os.path.getsize(os.path.join(resume_dir, directory, filename)) > 0
                for filename in required_checkpoint_files
            )
        ]
        if not dirs:
            raise FileNotFoundError(
                f"No complete checkpoint found under {resume_dir}; required files: "
                f"{required_checkpoint_files}"
            )
        path = dirs[-1]
        # Accelerator forwards these kwargs to either Module.load_state_dict or
        # safetensors.torch.load_model.  The latter calls this option `strict`;
        # `load_module_strict` is not a supported argument in our installed
        # safetensors version.
        accelerator.load_state(os.path.join(resume_dir, path), strict=False)
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])

        start_epoch = global_step // num_steps_per_epoch

    accelerator.wait_for_everyone()

    best_model_path = os.path.join(workspace_dir, "best")
    best_metric_path = os.path.join(best_model_path, "metrics.json")
    early_cfg = cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_min_steps = int(early_cfg.get("min_steps", 0))
    early_patience = int(early_cfg.get("patience_validations", 0))
    early_patience_steps = int(early_cfg.get("patience_steps", 0))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    if early_enabled and early_patience_steps <= 0 and early_patience <= 0:
        raise ValueError(
            "early_stopping requires positive patience_steps or "
            "patience_validations"
        )
    early_state_path = os.path.join(workspace_dir, "early_stopping.json")
    best_score = float(cfg.get("best_score_at_resume", float("-inf")))
    best_record_step = 0
    if os.path.isfile(best_metric_path):
        with open(best_metric_path, "r", encoding="utf-8") as handle:
            best_record = json.load(handle)
        best_record_step = int(best_record.get("step", 0))
        best_metric_name = str(
            cfg.get("best_checkpoint", {}).get("metric", "global_ssim")
        )
        if best_metric_name == "balanced_hand_nonhand_ssim":
            hand_weight = float(cfg.best_checkpoint.get("hand_weight", 0.5))
            nonhand_weight = float(cfg.best_checkpoint.get("nonhand_weight", 0.5))
            if "hand_ssim" in best_record and "nonhand_ssim" in best_record:
                resumed_score = (
                    hand_weight * float(best_record["hand_ssim"])
                    + nonhand_weight * float(best_record["nonhand_ssim"])
                )
                best_score = max(best_score, resumed_score)
        elif "score" in best_record:
            best_score = max(best_score, float(best_record["score"]))
        elif best_metric_name == "global_ssim":
            best_score = max(best_score, float(best_record["ssim"]))
    logger.info(f"Best checkpoint score to beat: {best_score}")

    early_reference_score = best_score
    early_last_best_step = best_record_step
    early_bad_validations = 0
    early_last_validation_step = -1
    restored_early_state = False
    if early_enabled and os.path.isfile(early_state_path):
        with open(early_state_path, "r", encoding="utf-8") as handle:
            early_state = json.load(handle)
        state_mode = str(early_state.get("mode", "validations"))
        if early_patience_steps > 0 and state_mode == "steps_since_meaningful_best":
            early_reference_score = float(
                early_state.get("reference_score", early_reference_score)
            )
            early_last_best_step = int(
                early_state.get("last_best_step", early_last_best_step)
            )
            early_last_validation_step = int(
                early_state.get("last_validation_step", -1)
            )
            restored_early_state = True
        elif early_patience_steps <= 0:
            early_reference_score = float(
                early_state.get("reference_score", early_reference_score)
            )
            early_bad_validations = int(early_state.get("bad_validations", 0))
            early_last_validation_step = int(
                early_state.get("last_validation_step", -1)
            )
            restored_early_state = True
    if (
        early_enabled
        and not restored_early_state
        and global_step > 0
        and math.isfinite(best_score)
    ):
        if early_patience_steps > 0:
            # When switching an older validation-count run to step-based
            # patience, conservatively continue counting from its saved best.
            early_last_best_step = best_record_step
            early_last_validation_step = global_step
        else:
            # Backward-compatible resume for a run created before early-stopping
            # state existed. Validation is periodic, so reconstruct the conservative
            # no-improvement count from the saved best step without double-counting
            # the validation performed immediately after resume.
            early_bad_validations = max(
                (global_step - best_record_step) // int(cfg.valid_steps), 0
            )
            early_last_validation_step = global_step

    def update_early_stopping(step, score, new_best=False):
        nonlocal early_reference_score
        nonlocal early_last_best_step
        nonlocal early_bad_validations
        nonlocal early_last_validation_step
        if not early_enabled:
            return False
        if early_patience_steps > 0:
            meaningful_improvement = (
                not math.isfinite(early_reference_score)
                or float(score) >= early_reference_score + early_min_delta
            )
            if meaningful_improvement:
                early_reference_score = float(score)
                early_last_best_step = int(step)
            steps_without_best = max(int(step) - early_last_best_step, 0)
            early_last_validation_step = int(step)
            should_stop = (
                int(step) >= early_min_steps
                and steps_without_best >= early_patience_steps
            )
            if accelerator.is_main_process:
                with open(early_state_path, "w", encoding="utf-8") as handle:
                    json.dump(
                        {
                            "mode": "steps_since_meaningful_best",
                            "last_validation_step": early_last_validation_step,
                            "last_best_step": early_last_best_step,
                            "reference_score": early_reference_score,
                            "steps_without_best": steps_without_best,
                            "patience_steps": early_patience_steps,
                            "min_delta": early_min_delta,
                            "min_steps": early_min_steps,
                            "stopped": should_stop,
                        },
                        handle,
                        indent=2,
                    )
            logger.info(
                "Early stopping: step=%d score=%.9f reference=%.9f "
                "last_meaningful_best_step=%d gap=%d/%d min_delta=%g "
                "min_steps=%d stop=%s",
                int(step),
                float(score),
                early_reference_score,
                early_last_best_step,
                steps_without_best,
                early_patience_steps,
                early_min_delta,
                early_min_steps,
                should_stop,
            )
            return should_stop
        if int(step) == early_last_validation_step:
            return (
                int(step) >= early_min_steps
                and early_bad_validations >= early_patience
            )
        meaningful_improvement = (
            not math.isfinite(early_reference_score)
            or float(score) >= early_reference_score + early_min_delta
        )
        if meaningful_improvement:
            early_reference_score = float(score)
            early_bad_validations = 0
        else:
            early_bad_validations += 1
        early_last_validation_step = int(step)
        should_stop = (
            int(step) >= early_min_steps
            and early_bad_validations >= early_patience
        )
        if accelerator.is_main_process:
            with open(early_state_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "last_validation_step": early_last_validation_step,
                        "reference_score": early_reference_score,
                        "bad_validations": early_bad_validations,
                        "patience_validations": early_patience,
                        "min_delta": early_min_delta,
                        "min_steps": early_min_steps,
                        "stopped": should_stop,
                    },
                    handle,
                    indent=2,
                )
        logger.info(
            "Early stopping: step=%d score=%.9f reference=%.9f "
            "bad=%d/%d min_steps=%d stop=%s",
            int(step),
            float(score),
            float(early_reference_score),
            early_bad_validations,
            early_patience,
            early_min_steps,
            should_stop,
        )
        return should_stop

    validation_metrics = log_valid(
        cfg,
        accelerator,
        valid_loader,
        model,
        empty_text_emb,
        val_noise_scheduler,
        device,
        weight_dtype,
        global_step,
        workspace_dir + "/log",
    )

    initial_new_best = False
    if accelerator.is_main_process and validation_metrics["score"] > best_score:
        best_score = validation_metrics["score"]
        initial_new_best = True
        save_model(
            model, cfg, best_model_path, accelerator.is_main_process, weight_dtype
        )
        with open(best_metric_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "step": global_step,
                    **validation_metrics,
                    "selection_metric": str(
                        cfg.get("best_checkpoint", {}).get("metric", "global_ssim")
                    ),
                },
                handle,
                indent=2,
            )
        logger.info(f"Saved best model with selection score: {best_score}")
        notify_best_checkpoint(
            workspace_dir,
            best_metric_path,
            os.environ.get("SIGNVIP_EXPERIMENT_NAME", cfg.exp_name),
        )
    early_stop_requested = update_early_stopping(
        global_step, validation_metrics["score"], new_best=initial_new_best
    )

    progress_bar = tqdm(
        range(global_step, cfg.solver.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description(f"Exp: {cfg.exp_name}")

    hamer_log_interval = int(
        cfg.get("hamer_reinject", {}).get("log_interval", 100)
    )
    hamer_shapes_logged = False
    hamer_target_shapes_logged = False

    for epoch in range(start_epoch, num_epochs):
        train_loss = 0.0
        train_l1_loss = 0.0
        t_data_start = time.time()
        for step, batch in enumerate(dataloader):
            t_data = time.time() - t_data_start
            model.train()
            with accelerator.accumulate(model):
                ref_frame = batch["ref_frame"].to(device, weight_dtype)  # [bs, c, h, w]
                tgt_frames = batch["tgt_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]
                tgt_sk_frames = batch["tgt_sk_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]

                tgt_hamer_frames = batch["tgt_hamer_frames"].to(
                    device, weight_dtype
                )  # [bs, c, f, h, w]
                left_hand_roi_mask = batch.get("left_hand_roi_mask")
                right_hand_roi_mask = batch.get("right_hand_roi_mask")

                with torch.no_grad(), accelerator.autocast():
                    frames = torch.cat(
                        [ref_frame.unsqueeze(2), tgt_frames], dim=2
                    )  # [bs, c, f+1, h, w]
                    _, _, num_frames, _, _ = frames.shape

                    frames = rearrange(frames, "b c f h w -> (b f) c h w")
                    latents = vae.encode(frames).latent_dist.sample()
                    latents = rearrange(
                        latents, "(b f) c h w -> b c f h w", f=num_frames
                    )
                    latents = latents * 0.18215
                    tgt_latents = latents[:, :, 1:, ...]  # [bs, c, f, h, w]
                    ref_latents = latents[:, :, 0, ...]  # [bs, c, h, w]

                    num_visible_frames = (
                        0
                        if cfg.stage_init
                        else random.randint(0, cfg.max_visible_frames)
                    )
                    visible_tgt_latents = tgt_latents[:, :, :num_visible_frames, ...]
                    non_visible_tgt_latents = tgt_latents[
                        :, :, num_visible_frames:, ...
                    ]

                bs, nc, _, _, _ = non_visible_tgt_latents.shape

                text_embeds = empty_text_emb.repeat(bs, 1, 1)

                noise = torch.randn_like(non_visible_tgt_latents)
                if cfg.noise_offset > 0:
                    noise += cfg.noise_offset * torch.randn(
                        (bs, nc, 1, 1, 1), device=latents.device
                    )
                noise = (
                    torch.cat(
                        [
                            torch.zeros_like(visible_tgt_latents),
                            noise,
                        ],
                        dim=2,
                    )
                    if not cfg.stage_init
                    else noise
                )

                timesteps = torch.randint(
                    0, noisy_scheduler.num_train_timesteps, (bs,), device=latents.device
                ).long()
                noisy_latents = noisy_scheduler.add_noise(tgt_latents, noise, timesteps)
                tgt_sk_frames = rearrange(tgt_sk_frames, "b c f h w -> (b f) c h w")
                tgt_hamer_frames = rearrange(
                    tgt_hamer_frames, "b c f h w -> (b f) c h w"
                )
                if left_hand_roi_mask is not None:
                    left_hand_roi_mask = rearrange(
                        left_hand_roi_mask.to(device=device, dtype=torch.float32),
                        "b c f h w -> (b f) c h w",
                    )
                    right_hand_roi_mask = rearrange(
                        right_hand_roi_mask.to(device=device, dtype=torch.float32),
                        "b c f h w -> (b f) c h w",
                    )

                with accelerator.autocast():
                    reinjector = getattr(model.unet, "hamer_reinjector", None)
                    hamer_reinject_active = bool(
                        reinjector is not None
                        and reinjector.enabled
                        and any(reinjector.site_enabled.values())
                    )
                    hamer_pre_gate = None
                    collect_hamer_stats = False
                    if hamer_reinject_active:
                        sk_features, mid_cond, pre_gate_features = model.condition_encoder(
                            tgt_sk_frames,
                            tgt_hamer_frames,
                            video_length=num_frames - 1,
                            return_cond=True,
                            return_pre_gate=True,
                        )
                        hamer_pre_gate = pre_gate_features["hand"]
                        if not hamer_shapes_logged and accelerator.is_local_main_process:
                            logger.info(
                                "C5.4-A pre-Gate runtime shapes: pose=%s hand=%s",
                                tuple(pre_gate_features["pose"].shape),
                                tuple(hamer_pre_gate.shape),
                            )
                            hamer_shapes_logged = True
                        collect_hamer_stats = (
                            global_step < 3
                            or (global_step + 1) % hamer_log_interval == 0
                        )
                    else:
                        sk_features, mid_cond = model.condition_encoder(
                            tgt_sk_frames,
                            tgt_hamer_frames,
                            video_length=num_frames - 1,
                            return_cond=True,
                        )
                    sk_features = [
                        rearrange(
                            sk_feature,
                            "(b f) c h w -> b c f h w",
                            f=num_frames - 1,
                        )
                        for sk_feature in sk_features
                    ]

                    model.appearance_encoder(ref_latents, timesteps, text_embeds)
                    model.reference_control_reader.update(
                        model.reference_control_writer, dtype=noisy_latents.dtype
                    )
                    pred = model.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=text_embeds,
                        sk_features=sk_features,
                        hamer_pre_gate=hamer_pre_gate,
                        hamer_geometry_image=tgt_hamer_frames,
                        left_hand_roi_mask=left_hand_roi_mask,
                        right_hand_roi_mask=right_hand_roi_mask,
                        collect_hamer_reinject_stats=collect_hamer_stats,
                    ).sample

                    refiner_details = {}
                    if model.latent_refiner is not None:
                        unrefined_x0 = predict_x0(
                            noisy_latents, pred, timesteps, noisy_scheduler
                        )
                        refined_x0, refiner_details = model.latent_refiner(
                            unrefined_x0,
                            timesteps,
                            hamer_image=tgt_hamer_frames,
                        )
                        pred = prediction_from_x0(
                            noisy_latents, refined_x0, timesteps, noisy_scheduler
                        )

                if noisy_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noisy_scheduler.prediction_type == "v_prediction":
                    target = noisy_scheduler.get_velocity(tgt_latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noisy_scheduler.prediction_type}"
                    )

                loss = func.mse_loss(pred.float(), target.float(), reduction="none")

                roi_loss_mode = str(cfg.get("roi_loss_mode", "none")).lower()
                if roi_loss_mode not in {"none", "stableanimator_weighted"}:
                    raise ValueError(
                        "roi_loss_mode must be none or stableanimator_weighted, "
                        f"got {roi_loss_mode!r}"
                    )
                if roi_loss_mode == "stableanimator_weighted":
                    roi_weight = torch.ones_like(loss[:, :1])
                    for mask_name, mask_weight in (
                        ("hand_mask", float(cfg.get("hand_loss_weight", 1.0))),
                        ("face_mask", float(cfg.get("face_loss_weight", 1.0))),
                    ):
                        if mask_name not in batch:
                            raise KeyError(
                                f"{mask_name} is required for StableAnimator loss"
                            )
                        mask = batch[mask_name].to(
                            device=device, dtype=torch.float32, non_blocking=True
                        )
                        if mask.ndim != 5 or mask.shape[1] != 1:
                            raise ValueError(
                                f"Expected {mask_name} [B,1,F,H,W], got {tuple(mask.shape)}"
                            )
                        latent_mask = rearrange(mask, "b c f h w -> (b f) c h w")
                        latent_mask = func.interpolate(
                            latent_mask,
                            size=loss.shape[-2:],
                            mode="area",
                        ).clamp_(0, 1)
                        region = mask_name.removesuffix("_mask")
                        mask_hard = bool(
                            cfg.get(f"roi_{region}_mask_hard", False)
                        )
                        mask_dilation = int(
                            cfg.get(f"roi_{region}_mask_dilation_radius", 0)
                        )
                        if mask_dilation not in (0, 1, 2):
                            raise ValueError(
                                f"roi_{region}_mask_dilation_radius must be "
                                f"0, 1, or 2, got {mask_dilation}"
                            )
                        if mask_hard or mask_dilation > 0:
                            threshold = float(
                                cfg.get(f"roi_{region}_mask_threshold", 0.05)
                            )
                            latent_mask = (latent_mask > threshold).float()
                            if mask_dilation > 0:
                                kernel_size = 2 * mask_dilation + 1
                                latent_mask = func.max_pool2d(
                                    latent_mask,
                                    kernel_size=kernel_size,
                                    stride=1,
                                    padding=mask_dilation,
                                )
                        latent_mask = rearrange(
                            latent_mask,
                            "(b f) c h w -> b c f h w",
                            f=loss.shape[2],
                        )
                        roi_weight = roi_weight + mask_weight * latent_mask
                    # StableAnimator: global mean of epsilon error weighted by
                    # 1 + hand_mask + face_mask, without area normalization.
                    loss = loss * roi_weight

                if cfg.snr_gamma != 0:
                    snr = compute_snr(noisy_scheduler, timesteps)
                    if noisy_scheduler.prediction_type == "v_prediction":
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack(
                            [
                                snr,
                                cfg.snr_gamma * torch.ones_like(timesteps),
                            ],
                            dim=1,
                        ).min(dim=1)[0]
                        / snr
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )

                diffusion_loss = loss.mean()
                clip_loss_value = diffusion_loss.new_zeros(())
                dino_loss_value = diffusion_loss.new_zeros(())
                hand_recon_loss = diffusion_loss.new_zeros(())
                face_recon_loss = diffusion_loss.new_zeros(())
                hand_local_clip_loss = diffusion_loss.new_zeros(())
                hand_local_dino_loss = diffusion_loss.new_zeros(())
                face_local_clip_loss = diffusion_loss.new_zeros(())
                face_local_dino_loss = diffusion_loss.new_zeros(())
                global_x0_recon_loss = diffusion_loss.new_zeros(())
                hand_latent_x0_loss = diffusion_loss.new_zeros(())
                hand_gradient_loss = diffusion_loss.new_zeros(())
                hand_laplacian_loss = diffusion_loss.new_zeros(())
                direct_refiner_loss = diffusion_loss.new_zeros(())
                decoder_refiner_loss = diffusion_loss.new_zeros(())
                if direct_refiner_enabled:
                    if model.latent_refiner is None:
                        raise RuntimeError("Direct residual loss requires latent_refiner")
                    if "hand_mask" not in batch:
                        raise KeyError("hand_mask is required for direct refiner supervision")
                    direct_mask = build_soft_hand_latent_mask(
                        batch["hand_mask"].to(device=device, dtype=torch.float32),
                        tgt_latents.shape[-2:],
                        threshold=float(refiner_training_cfg.get("mask_threshold", 0.05)),
                        dilation_radius=int(refiner_training_cfg.get("dilation_radius", 2)),
                        ring1_weight=1.0,
                        ring2_weight=1.0,
                    )
                    target_delta = (tgt_latents.float() - unrefined_x0.float()).detach()
                    delta_error = (refiner_details["delta"].float() - target_delta).square()
                    direct_refiner_loss = (delta_error * direct_mask).sum() / (
                        direct_mask.sum().clamp_min(1.0) * delta_error.shape[1]
                    )
                perceptual_gradient_checkpointing = bool(
                    perceptual_cfg.get("gradient_checkpointing", False)
                )
                if (
                    perceptual_enabled
                    or region_enabled
                    or global_x0_enabled
                    or hand_latent_x0_enabled
                    or hand_frequency_enabled
                    or decoder_refiner_enabled
                ):
                    predicted_x0 = predict_x0(
                        noisy_latents, pred, timesteps, noisy_scheduler
                    )
                    if hand_latent_x0_enabled:
                        if "hand_mask" not in batch:
                            raise KeyError("hand_mask is required for hand_latent_x0_loss")
                        hand_latent_mask = build_soft_hand_latent_mask(
                            batch["hand_mask"].to(
                                device=device,
                                dtype=torch.float32,
                                non_blocking=True,
                            ),
                            predicted_x0.shape[-2:],
                            threshold=float(
                                hand_latent_x0_cfg.get("mask_threshold", 0.05)
                            ),
                            dilation_radius=int(
                                hand_latent_x0_cfg.get("dilation_radius", 2)
                            ),
                            ring1_weight=float(
                                hand_latent_x0_cfg.get("ring1_weight", 0.5)
                            ),
                            ring2_weight=float(
                                hand_latent_x0_cfg.get("ring2_weight", 0.25)
                            ),
                        ).to(dtype=torch.float32)
                        latent_error = (
                            predicted_x0.float() - tgt_latents.float()
                        ).square()
                        numerator = (latent_error * hand_latent_mask).sum(
                            dim=(1, 2, 3, 4)
                        )
                        denominator = hand_latent_mask.sum(
                            dim=(1, 2, 3, 4)
                        ) * latent_error.shape[1]
                        valid = denominator > 0
                        if valid.any():
                            per_sample_hand_latent_loss = (
                                numerator[valid]
                                / denominator[valid].clamp_min(1e-6)
                            )
                            timestep_weighting = str(
                                hand_latent_x0_cfg.get(
                                    "timestep_weighting", "alpha_bar"
                                )
                            )
                            if timestep_weighting == "alpha_bar":
                                hand_snr = compute_snr(
                                    noisy_scheduler, timesteps
                                ).float()
                                timestep_weights = hand_snr / (hand_snr + 1.0)
                                per_sample_hand_latent_loss = (
                                    per_sample_hand_latent_loss
                                    * timestep_weights[valid]
                                )
                            elif timestep_weighting != "none":
                                raise ValueError(
                                    "hand_latent_x0_loss.timestep_weighting must "
                                    "be alpha_bar or none"
                                )
                            hand_latent_x0_loss = (
                                per_sample_hand_latent_loss.mean()
                            )
                    with accelerator.autocast():
                        predicted_images = decode_latents(
                            vae,
                            predicted_x0,
                            chunk_size=int(perceptual_cfg.get("decode_chunk_size", 2)),
                        )
                        target_images = flatten_target_frames(tgt_frames)
                    if global_x0_enabled:
                        global_x0_recon_loss = global_reconstruction_loss(
                            predicted_images,
                            target_images,
                            loss_type=str(global_x0_cfg.get("reconstruction", "mse")),
                            charbonnier_eps=float(
                                global_x0_cfg.get("charbonnier_eps", 1e-3)
                            ),
                        )
                    if hand_frequency_enabled:
                        if "hand_mask" not in batch:
                            raise KeyError("hand_mask is required for hand_frequency_loss")
                        frequency_latent_mask = build_soft_hand_latent_mask(
                            batch["hand_mask"].to(device=device, dtype=torch.float32),
                            predicted_x0.shape[-2:],
                            threshold=float(hand_frequency_cfg.get("mask_threshold", 0.05)),
                            dilation_radius=int(hand_frequency_cfg.get("dilation_radius", 2)),
                            ring1_weight=1.0,
                            ring2_weight=1.0,
                        )
                        frequency_mask = rearrange(
                            frequency_latent_mask, "b 1 f h w -> (b f) 1 h w"
                        )
                        hand_gradient_loss, hand_laplacian_loss = hand_frequency_losses(
                            predicted_images, target_images, frequency_mask
                        )
                    if decoder_refiner_enabled:
                        decoder_latent_mask = build_soft_hand_latent_mask(
                            batch["hand_mask"].to(device=device, dtype=torch.float32),
                            predicted_x0.shape[-2:],
                            threshold=float(refiner_training_cfg.get("mask_threshold", 0.05)),
                            dilation_radius=int(refiner_training_cfg.get("dilation_radius", 2)),
                            ring1_weight=1.0,
                            ring2_weight=1.0,
                        )
                        decoder_mask = rearrange(
                            decoder_latent_mask, "b 1 f h w -> (b f) 1 h w"
                        )
                        decoder_refiner_loss = normalized_masked_reconstruction_loss(
                            predicted_images,
                            target_images,
                            decoder_mask,
                            loss_type=str(refiner_training_cfg.get("decoder_loss_type", "charbonnier")),
                            charbonnier_eps=float(refiner_training_cfg.get("charbonnier_eps", 1e-3)),
                            min_mask_pixels=1.0,
                        )
                    if region_enabled:
                        region_masks = {}
                        for region_name, batch_name in (
                            ("hand", "hand_mask"),
                            ("face", "face_mask"),
                        ):
                            if batch_name not in batch:
                                raise KeyError(f"{batch_name} is required for region_x0_loss")
                            region_masks[region_name] = flatten_region_masks(
                                batch[batch_name].to(
                                    device=device, dtype=torch.float32, non_blocking=True
                                )
                            )
                        recon_kwargs = dict(
                            loss_type=str(region_cfg.get("reconstruction", "charbonnier")),
                            charbonnier_eps=float(region_cfg.get("charbonnier_eps", 1e-3)),
                            min_mask_pixels=float(region_cfg.get("min_mask_pixels", 4.0)),
                        )
                        hand_recon_loss = normalized_masked_reconstruction_loss(
                            predicted_images, target_images, region_masks["hand"], **recon_kwargs
                        )
                        face_recon_loss = normalized_masked_reconstruction_loss(
                            predicted_images, target_images, region_masks["face"], **recon_kwargs
                        )
                        crop_kwargs = dict(
                            output_size=int(region_cfg.get("crop_size", 128)),
                            padding=float(region_cfg.get("crop_padding", 0.2)),
                            threshold=float(region_cfg.get("mask_threshold", 0.5)),
                            min_mask_pixels=float(region_cfg.get("min_mask_pixels", 4.0)),
                        )
                        for region_name in ("hand", "face"):
                            local_prediction, local_target = crop_resize_region_pairs(
                                predicted_images,
                                target_images,
                                region_masks[region_name],
                                **crop_kwargs,
                            )
                            if local_prediction.shape[0] > 0:
                                if float(region_cfg.get(f"{region_name}_clip_weight", 0.0)) != 0:
                                    value = clip_loss(
                                        clip_model,
                                        local_prediction,
                                        local_target,
                                        gradient_checkpointing=perceptual_gradient_checkpointing,
                                    )
                                    if region_name == "hand":
                                        hand_local_clip_loss = value
                                    else:
                                        face_local_clip_loss = value
                                if float(region_cfg.get(f"{region_name}_dino_weight", 0.0)) != 0:
                                    value = dino_loss(
                                        dino_model,
                                        local_prediction,
                                        local_target,
                                        gradient_checkpointing=perceptual_gradient_checkpointing,
                                    )
                                    if region_name == "hand":
                                        hand_local_dino_loss = value
                                    else:
                                        face_local_dino_loss = value
                if perceptual_enabled:
                    with accelerator.autocast():
                        clip_loss_value = clip_loss(
                            clip_model,
                            predicted_images,
                            target_images,
                            gradient_checkpointing=perceptual_gradient_checkpointing,
                        )
                        dino_loss_value = dino_loss(
                            dino_model,
                            predicted_images,
                            target_images,
                            gradient_checkpointing=perceptual_gradient_checkpointing,
                        )
                diffusion_weight = float(
                    perceptual_cfg.get("diffusion_weight", 1.0)
                )
                loss = (
                    float(perceptual_cfg.get("clip_weight", 0.0)) * clip_loss_value
                    + float(perceptual_cfg.get("dino_weight", 0.0)) * dino_loss_value
                )
                if diffusion_weight != 0:
                    loss = loss + diffusion_weight * diffusion_loss
                if region_enabled:
                    loss = (
                        loss
                        + float(region_cfg.get("hand_reconstruction_weight", 0.0))
                        * hand_recon_loss
                        + float(region_cfg.get("face_reconstruction_weight", 0.0))
                        * face_recon_loss
                        + float(region_cfg.get("hand_clip_weight", 0.0))
                        * hand_local_clip_loss
                        + float(region_cfg.get("hand_dino_weight", 0.0))
                        * hand_local_dino_loss
                        + float(region_cfg.get("face_clip_weight", 0.0))
                        * face_local_clip_loss
                        + float(region_cfg.get("face_dino_weight", 0.0))
                        * face_local_dino_loss
                    )
                if global_x0_enabled:
                    loss = (
                        loss
                        + float(global_x0_cfg.get("weight", 1.0))
                        * global_x0_recon_loss
                    )
                if hand_latent_x0_enabled:
                    loss = (
                        loss
                        + float(hand_latent_x0_cfg.get("weight", 0.25))
                        * hand_latent_x0_loss
                    )
                if hand_frequency_enabled:
                    loss = (
                        loss
                        + float(hand_frequency_cfg.get("gradient_weight", 0.05))
                        * hand_gradient_loss
                        + float(hand_frequency_cfg.get("laplacian_weight", 0.05))
                        * hand_laplacian_loss
                    )
                if direct_refiner_enabled:
                    loss = loss + float(
                        refiner_training_cfg.get("direct_residual_weight", 0.25)
                    ) * direct_refiner_loss
                if decoder_refiner_enabled:
                    loss = loss + float(
                        refiner_training_cfg.get("decoder_hand_weight", 0.25)
                    ) * decoder_refiner_loss

                avg_loss = accelerator.gather(
                    loss.repeat(cfg.dataloader.batch_size)
                ).mean()

                train_loss += avg_loss.item() / cfg.solver.gradient_accumulation_steps
                accelerator.backward(loss)

                hamer_grad_stats = {}
                if accelerator.sync_gradients and collect_hamer_stats and getattr(
                    model.unet, "hamer_reinjector", None
                ) is not None:
                    active_reinjector = model.unet.hamer_reinjector
                    for site in active_reinjector.SITES:
                        gamma = active_reinjector.gamma(site)
                        if gamma.grad is not None:
                            hamer_grad_stats[f"hamer_reinject/grad_gamma_{site}"] = float(
                                gamma.grad.detach().float().norm().item()
                            )
                        adapter_grads = [
                            parameter.grad.detach().float().norm().square()
                            for parameter in active_reinjector.adapters[site].parameters()
                            if parameter.grad is not None
                        ]
                        if adapter_grads:
                            hamer_grad_stats[f"hamer_reinject/grad_adapter_{site}"] = float(
                                torch.stack(adapter_grads).sum().sqrt().item()
                            )
                    if active_reinjector.geometry_enabled:
                        if active_reinjector.gamma_geometry.grad is not None:
                            hamer_grad_stats["hamer_reinject/grad_gamma_geometry"] = float(
                                active_reinjector.gamma_geometry.grad.detach().float().norm().item()
                            )
                        geometry_grads = [
                            parameter.grad.detach().float().norm().square()
                            for name, parameter in active_reinjector.named_parameters()
                            if name.startswith("geometry_")
                            and name != "gamma_geometry"
                            and parameter.grad is not None
                        ]
                        if geometry_grads:
                            hamer_grad_stats["hamer_reinject/grad_geometry_modules"] = float(
                                torch.stack(geometry_grads).sum().sqrt().item()
                            )
                    if active_reinjector.hand_edge_enabled:
                        if active_reinjector.gamma_hand_edge.grad is not None:
                            hamer_grad_stats["hamer_reinject/grad_gamma_hand_edge"] = float(
                                active_reinjector.gamma_hand_edge.grad.detach().float().norm().item()
                            )
                        edge_grads = [
                            parameter.grad.detach().float().norm().square()
                            for name, parameter in active_reinjector.named_parameters()
                            if name.startswith("hand_edge_")
                            and name != "gamma_hand_edge"
                            and parameter.grad is not None
                        ]
                        if edge_grads:
                            hamer_grad_stats["hamer_reinject/grad_hand_edge_modules"] = float(
                                torch.stack(edge_grads).sum().sqrt().item()
                            )
                    if active_reinjector.pixel_geometry_enabled:
                        if active_reinjector.gamma_pixel_geometry.grad is not None:
                            hamer_grad_stats[
                                "hamer_reinject/grad_gamma_pixel_geometry"
                            ] = float(
                                active_reinjector.gamma_pixel_geometry.grad.detach()
                                .float()
                                .norm()
                                .item()
                            )
                        pixel_grads = [
                            parameter.grad.detach().float().norm().square()
                            for name, parameter in active_reinjector.named_parameters()
                            if name.startswith("pixel_geometry_")
                            and name != "gamma_pixel_geometry"
                            and parameter.grad is not None
                        ]
                        if pixel_grads:
                            hamer_grad_stats[
                                "hamer_reinject/grad_pixel_geometry_modules"
                            ] = float(torch.stack(pixel_grads).sum().sqrt().item())
                    if active_reinjector.wavelet_edge_enabled:
                        if active_reinjector.gamma_wavelet_edge.grad is not None:
                            hamer_grad_stats[
                                "hamer_reinject/grad_gamma_wavelet_edge"
                            ] = float(
                                active_reinjector.gamma_wavelet_edge.grad.detach()
                                .float()
                                .norm()
                                .item()
                            )
                        wavelet_grads = [
                            parameter.grad.detach().float().norm().square()
                            for name, parameter in active_reinjector.named_parameters()
                            if name.startswith("wavelet_edge_")
                            and name != "gamma_wavelet_edge"
                            and parameter.grad is not None
                        ]
                        if wavelet_grads:
                            hamer_grad_stats[
                                "hamer_reinject/grad_wavelet_edge_modules"
                            ] = float(torch.stack(wavelet_grads).sum().sqrt().item())

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable_params, cfg.solver.max_grad_norm
                    )

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # ReferenceAttentionControl stores feature tensors from the
                # current forward pass. With gradient accumulation those
                # tensors must be released after every micro-batch; retaining
                # them until sync_gradients reuses an already-freed autograd
                # graph on the next backward call.
                model.reference_control_reader.clear()
                model.reference_control_writer.clear()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if accelerator.is_main_process:
                    notify_training_progress(
                        workspace_dir,
                        cfg.exp_name,
                        global_step,
                        int(cfg.solver.max_train_steps),
                    )
                step_metrics = {
                        "train_loss": train_loss,
                        "diffusion_loss": diffusion_loss.detach().item(),
                        "clip_loss": clip_loss_value.detach().item(),
                        "dino_loss": dino_loss_value.detach().item(),
                        "hand_recon_loss": hand_recon_loss.detach().item(),
                        "face_recon_loss": face_recon_loss.detach().item(),
                        "hand_local_clip_loss": hand_local_clip_loss.detach().item(),
                        "hand_local_dino_loss": hand_local_dino_loss.detach().item(),
                        "face_local_clip_loss": face_local_clip_loss.detach().item(),
                        "face_local_dino_loss": face_local_dino_loss.detach().item(),
                        "global_x0_recon_loss": global_x0_recon_loss.detach().item(),
                        "hand_latent_x0_loss": hand_latent_x0_loss.detach().item(),
                        "hand_gradient_loss": hand_gradient_loss.detach().item(),
                        "hand_laplacian_loss": hand_laplacian_loss.detach().item(),
                        "direct_refiner_loss": direct_refiner_loss.detach().item(),
                        "decoder_refiner_loss": decoder_refiner_loss.detach().item(),
                    }
                step_metrics.update(hamer_grad_stats)
                active_reinjector = getattr(model.unet, "hamer_reinjector", None)
                if active_reinjector is not None and collect_hamer_stats:
                    for site, stats in active_reinjector.last_stats.items():
                        for key in (
                            "gamma",
                            "hidden_norm",
                            "residual_norm",
                            "delta_norm",
                            "ratio",
                        ):
                            step_metrics[f"hamer_reinject/{site}/{key}"] = stats[key]
                        shape = stats["hidden_shape"]
                        residual_shape = stats["residual_shape"]
                        step_metrics[f"hamer_reinject/{site}/batch"] = shape[0]
                        step_metrics[f"hamer_reinject/{site}/channels"] = shape[1]
                        step_metrics[f"hamer_reinject/{site}/frames"] = shape[2]
                        step_metrics[f"hamer_reinject/{site}/height"] = shape[-2]
                        step_metrics[f"hamer_reinject/{site}/width"] = shape[-1]
                        step_metrics[f"hamer_reinject/{site}/residual_channels"] = (
                            residual_shape[1]
                        )
                        if not hamer_target_shapes_logged and accelerator.is_local_main_process:
                            logger.info(
                                "HAMER reinjection location=%s target_shape=%s "
                                "residual_shape=%s gamma=%.8g ratio=%.8g",
                                site,
                                tuple(shape),
                                tuple(residual_shape),
                                stats["gamma"],
                                stats["ratio"],
                            )
                    if active_reinjector.last_stats:
                        hamer_target_shapes_logged = True
                accelerator.log(step_metrics, step=global_step)
                train_loss = 0.0
                if global_step % cfg.valid_steps == 0 and global_step > 0:
                    validation_metrics = log_valid(
                        cfg,
                        accelerator,
                        valid_loader,
                        model,
                        empty_text_emb,
                        val_noise_scheduler,
                        device,
                        weight_dtype,
                        global_step,
                        workspace_dir + "/log",
                    )
                    # Save using the configured validation selection score.
                    validation_new_best = False
                    if (
                        accelerator.is_main_process
                        and validation_metrics["score"] > best_score
                    ):
                        best_score = validation_metrics["score"]
                        validation_new_best = True
                        save_model(
                            model,
                            cfg,
                            best_model_path,
                            accelerator.is_main_process,
                            weight_dtype,
                        )
                        with open(best_metric_path, "w", encoding="utf-8") as handle:
                            json.dump(
                                {
                                    "step": global_step,
                                    **validation_metrics,
                                    "selection_metric": str(
                                        cfg.get("best_checkpoint", {}).get(
                                            "metric", "global_ssim"
                                        )
                                    ),
                                },
                                handle,
                                indent=2,
                            )
                        logger.info(
                            "Saved best model with selection score: %s",
                            best_score,
                        )
                        notify_best_checkpoint(
                            workspace_dir,
                            best_metric_path,
                            os.environ.get("SIGNVIP_EXPERIMENT_NAME", cfg.exp_name),
                        )
                    early_stop_requested = update_early_stopping(
                        global_step,
                        validation_metrics["score"],
                        new_best=validation_new_best,
                    )

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "td": f"{t_data:.2f}s",
            }
            t_data_start = time.time()
            progress_bar.set_postfix(**logs)

            if global_step >= cfg.solver.max_train_steps or early_stop_requested:
                break

            if (
                accelerator.sync_gradients
                and global_step > 0
                and global_step % cfg.checkpointing_steps == 0
            ):
                if accelerator.is_main_process:
                    if cfg.max_model is not None:
                        models = os.listdir(workspace_dir)
                        models = [d for d in models if d.startswith("model")]
                        models = sorted(models, key=lambda x: int(x.split("-")[1]))

                        # before we save the new model, we need to have at most `max_model - 1` models
                        if len(models) >= cfg.max_model:
                            num_to_remove = len(models) - cfg.max_model + 1
                            removing_models = models[0:num_to_remove]

                            logger.info(
                                f"{len(models)} models already exist, removing {len(removing_models)} models"
                            )
                            logger.info(
                                f"removing models: {', '.join(removing_models)}"
                            )

                            for removing_model in removing_models:
                                removing_model_path = os.path.join(
                                    workspace_dir, removing_model
                                )
                                shutil.rmtree(removing_model_path)
                ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
                model_path = os.path.join(workspace_dir, f"model-{global_step}")

                save_resumable_checkpoint(
                    accelerator, workspace_dir, global_step, cfg.max_ckpt
                )

        # Breaking only the dataloader loop lets the next epoch perform one
        # additional optimizer step. Stop the epoch loop as well.
        if global_step >= cfg.solver.max_train_steps or early_stop_requested:
            break

    ckpt_path = os.path.join(workspace_dir, f"checkpoint-{global_step}")
    model_path = os.path.join(workspace_dir, f"model-{global_step}")

    save_resumable_checkpoint(
        accelerator, workspace_dir, global_step, cfg.max_ckpt
    )

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
