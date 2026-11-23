"""Frozen CLIP/DINO image losses adapted from ./code.py."""

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel, CLIPVisionModelWithProjection


_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _freeze(model, device, dtype):
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device, dtype=dtype)


def load_clip_model(model_id, device, dtype):
    model = CLIPVisionModelWithProjection.from_pretrained(
        model_id, torch_dtype=dtype
    )
    return _freeze(model, device, dtype)


def load_dino_model(model_id, device, dtype):
    model = AutoModel.from_pretrained(model_id, torch_dtype=dtype)
    return _freeze(model, device, dtype)


def _normalize(x, mean, std, size):
    if size is not None and x.shape[-2:] != (size, size):
        x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    mean = x.new_tensor(mean).view(1, 3, 1, 1)
    std = x.new_tensor(std).view(1, 3, 1, 1)
    return (x - mean) / std


def _cosine_distance(pred_embedding, target_embedding):
    pred_embedding = F.normalize(pred_embedding.float(), dim=-1)
    target_embedding = F.normalize(target_embedding.float(), dim=-1)
    return (1 - (pred_embedding * target_embedding).sum(dim=-1)).mean()


def clip_loss(model, prediction, target, gradient_checkpointing=False):
    size = int(model.config.image_size)
    prediction = _normalize(prediction, _CLIP_MEAN, _CLIP_STD, size)
    target = _normalize(target, _CLIP_MEAN, _CLIP_STD, size)
    if gradient_checkpointing and prediction.requires_grad:
        pred_embedding = checkpoint(
            lambda pixels: model(pixel_values=pixels).image_embeds,
            prediction,
            use_reentrant=False,
        )
    else:
        pred_embedding = model(pixel_values=prediction).image_embeds
    with torch.no_grad():
        target_embedding = model(pixel_values=target).image_embeds
    return _cosine_distance(pred_embedding, target_embedding)


def dino_loss(model, prediction, target, gradient_checkpointing=False):
    size = getattr(model.config, "image_size", None)
    if isinstance(size, (tuple, list)):
        if len(set(size)) != 1:
            raise ValueError(f"DINO requires a square image_size, got {size}")
        size = size[0]
    size = int(size) if size is not None else None
    prediction = _normalize(prediction, _IMAGENET_MEAN, _IMAGENET_STD, size)
    target = _normalize(target, _IMAGENET_MEAN, _IMAGENET_STD, size)
    if gradient_checkpointing and prediction.requires_grad:
        pred_embedding = checkpoint(
            lambda pixels: model(pixel_values=pixels).last_hidden_state[:, 0],
            prediction,
            use_reentrant=False,
        )
    else:
        pred_embedding = model(pixel_values=prediction).last_hidden_state[:, 0]
    with torch.no_grad():
        target_embedding = model(pixel_values=target).last_hidden_state[:, 0]
    return _cosine_distance(pred_embedding, target_embedding)


def predict_x0(noisy_latents, model_prediction, timesteps, scheduler):
    """Convert epsilon/v-prediction to differentiable clean latent x0."""
    alphas = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha = alphas[timesteps].view(-1, 1, 1, 1, 1)
    sigma = (1 - alpha).sqrt()
    alpha = alpha.sqrt()
    if scheduler.prediction_type == "epsilon":
        return (noisy_latents - sigma * model_prediction) / alpha.clamp_min(1e-6)
    if scheduler.prediction_type == "v_prediction":
        return alpha * noisy_latents - sigma * model_prediction
    raise ValueError(f"Unsupported prediction type: {scheduler.prediction_type}")


def decode_latents(vae, latents, chunk_size=2):
    flat = latents.permute(0, 2, 1, 3, 4).reshape(-1, *latents.shape[1:2], *latents.shape[3:])
    scaling_factor = float(getattr(vae.config, "scaling_factor", 0.18215))
    decoded = []
    for start in range(0, len(flat), chunk_size):
        decoded.append(vae.decode(flat[start : start + chunk_size] / scaling_factor).sample)
    return (torch.cat(decoded, dim=0) / 2 + 0.5).clamp(0, 1)


def flatten_target_frames(target_frames):
    return (
        target_frames.permute(0, 2, 1, 3, 4)
        .reshape(-1, target_frames.shape[1], target_frames.shape[3], target_frames.shape[4])
        .add(1)
        .div(2)
        .clamp(0, 1)
    )


def flatten_region_masks(masks):
    """Convert [B,1,F,H,W] masks to the [B*F,1,H,W] RGB-frame order."""
    if masks.ndim != 5 or masks.shape[1] != 1:
        raise ValueError(f"Expected masks [B,1,F,H,W], got {tuple(masks.shape)}")
    return masks.permute(0, 2, 1, 3, 4).reshape(
        -1, 1, masks.shape[3], masks.shape[4]
    )


def global_reconstruction_loss(
    prediction,
    target,
    loss_type="mse",
    charbonnier_eps=1e-3,
):
    """Global, area-unweighted RGB reconstruction objective."""
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    difference = prediction.float() - target.float()
    loss_type = str(loss_type).lower()
    if loss_type == "mse":
        error = difference.square()
    elif loss_type == "l1":
        error = difference.abs()
    elif loss_type == "charbonnier":
        epsilon = float(charbonnier_eps)
        error = torch.sqrt(difference.square() + epsilon * epsilon) - epsilon
    else:
        raise ValueError(
            "global reconstruction loss must be mse, l1, or charbonnier, "
            f"got {loss_type!r}"
        )
    return error.mean()


def normalized_masked_reconstruction_loss(
    prediction,
    target,
    mask,
    loss_type="charbonnier",
    charbonnier_eps=1e-3,
    min_mask_pixels=4.0,
):
    """Area-normalized x0 RGB loss, averaged only over frames with a region.

    Normalizing each frame by its own mask area prevents large face/hand masks
    from dominating small ones. The subtraction in Charbonnier makes an exact
    match contribute zero while retaining its smooth-L1 behavior.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"Prediction/target mismatch: {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if mask.shape[0] != prediction.shape[0] or mask.shape[1] != 1:
        raise ValueError(
            f"Mask {tuple(mask.shape)} does not align with images {tuple(prediction.shape)}"
        )
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask, size=prediction.shape[-2:], mode="area")

    prediction = prediction.float()
    target = target.float()
    mask = mask.to(device=prediction.device, dtype=prediction.dtype).clamp(0, 1)
    difference = prediction - target
    loss_type = str(loss_type).lower()
    if loss_type == "charbonnier":
        epsilon = float(charbonnier_eps)
        error = torch.sqrt(difference.square() + epsilon * epsilon) - epsilon
    elif loss_type == "l1":
        error = difference.abs()
    else:
        raise ValueError(f"region reconstruction loss must be charbonnier or l1, got {loss_type!r}")

    areas = mask.sum(dim=(1, 2, 3))
    denominator = (areas * prediction.shape[1]).clamp_min(1e-6)
    per_frame = (error * mask).sum(dim=(1, 2, 3)) / denominator
    valid = areas >= float(min_mask_pixels)
    if valid.any():
        return per_frame[valid].mean()
    # Preserve a differentiable zero when a batch contains no valid region.
    return prediction.sum() * 0.0


def crop_resize_region_pairs(
    prediction,
    target,
    mask,
    output_size=128,
    padding=0.2,
    threshold=0.5,
    min_mask_pixels=4.0,
):
    """Crop prediction/target pairs to mask boxes and resize for local losses."""
    if mask.shape[-2:] != prediction.shape[-2:]:
        mask = F.interpolate(mask.float(), size=prediction.shape[-2:], mode="area")
    masks = mask.detach().float()
    predicted_crops = []
    target_crops = []
    height, width = prediction.shape[-2:]
    for index in range(prediction.shape[0]):
        region = masks[index, 0] >= float(threshold)
        if region.sum().item() < float(min_mask_pixels):
            continue
        coordinates = region.nonzero(as_tuple=False)
        y0, x0 = coordinates.amin(dim=0).tolist()
        y1, x1 = coordinates.amax(dim=0).tolist()
        box_height, box_width = y1 - y0 + 1, x1 - x0 + 1
        pad_y = max(1, int(round(box_height * float(padding))))
        pad_x = max(1, int(round(box_width * float(padding))))
        y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y + 1)
        x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x + 1)
        predicted_crops.append(prediction[index : index + 1, :, y0:y1, x0:x1])
        target_crops.append(target[index : index + 1, :, y0:y1, x0:x1])

    if not predicted_crops:
        empty = prediction[:0]
        return empty, target[:0]
    size = (int(output_size), int(output_size))
    predicted_crops = torch.cat(
        [F.interpolate(crop, size=size, mode="bilinear", align_corners=False) for crop in predicted_crops]
    )
    target_crops = torch.cat(
        [F.interpolate(crop, size=size, mode="bilinear", align_corners=False) for crop in target_crops]
    )
    return predicted_crops, target_crops
