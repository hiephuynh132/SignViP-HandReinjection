"""Post-U-Net hand-latent residual refiners for the C5.19 ablations."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class HAMERLatentConditioner(nn.Module):
    """Encode a 128x128 HAMER render to a 16x16 spatial condition."""

    def __init__(self, channels: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels // 2, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels // 2),
            nn.SiLU(),
            nn.Conv2d(channels // 2, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
        )

    def forward(self, image, target_size):
        image = image.float()
        if image.detach().amin() < -0.01:
            image = image.add(1).mul(0.5)
        feature = self.net(image.clamp(0, 1))
        return F.interpolate(feature, size=target_size, mode="bilinear", align_corners=False)


class CorrectionStage(nn.Module):
    def __init__(self, latent_channels: int, condition_channels: int, hidden_channels: int):
        super().__init__()
        in_channels = latent_channels + condition_channels + 1
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(ResidualBlock(hidden_channels), ResidualBlock(hidden_channels))
        self.out_conv = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, latent, condition, timestep_map):
        inputs = [latent, timestep_map]
        if condition is not None:
            inputs.insert(1, condition)
        hidden = self.in_conv(torch.cat(inputs, dim=1))
        return self.out_conv(self.blocks(hidden))


class HighResolutionCorrection(nn.Module):
    """Temporarily refine a 16x16 latent at 32x32, then return to 16x16."""

    def __init__(self, latent_channels: int, condition_channels: int, hidden_channels: int):
        super().__init__()
        in_channels = latent_channels + condition_channels + 1
        self.in_conv = nn.Conv2d(in_channels, hidden_channels, 3, padding=1)
        self.blocks = nn.Sequential(
            ResidualBlock(hidden_channels), ResidualBlock(hidden_channels),
            ResidualBlock(hidden_channels),
        )
        self.out_conv = nn.Conv2d(hidden_channels, latent_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, latent, condition, timestep_map):
        size_32 = (latent.shape[-2] * 2, latent.shape[-1] * 2)
        latent_32 = F.interpolate(latent, size=size_32, mode="bilinear", align_corners=False)
        time_32 = F.interpolate(timestep_map, size=size_32, mode="nearest")
        inputs = [latent_32, time_32]
        if condition is not None:
            inputs.insert(1, F.interpolate(condition, size=size_32, mode="bilinear", align_corners=False))
        delta_32 = self.out_conv(self.blocks(self.in_conv(torch.cat(inputs, dim=1))))
        return F.interpolate(delta_32, size=latent.shape[-2:], mode="area")


class HandLatentResidualRefiner(nn.Module):
    """Correct predicted clean latents after the frozen denoising U-Net."""

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_channels: int = 64,
        condition_channels: int = 32,
        hamer_conditioned: bool = False,
        spatial_gate: bool = False,
        two_stage: bool = False,
        high_resolution: bool = False,
        mask_threshold: float = 0.05,
        mask_dilation: int = 2,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.hamer_conditioned = bool(hamer_conditioned)
        self.spatial_gate_enabled = bool(spatial_gate)
        self.two_stage = bool(two_stage)
        self.high_resolution = bool(high_resolution)
        self.mask_threshold = float(mask_threshold)
        self.mask_dilation = int(mask_dilation)
        self.num_train_timesteps = int(num_train_timesteps)
        if self.mask_dilation not in (0, 1, 2):
            raise ValueError("mask_dilation must be 0, 1, or 2")
        effective_condition_channels = int(condition_channels) if self.hamer_conditioned else 0
        self.hamer_conditioner = (
            HAMERLatentConditioner(int(condition_channels)) if self.hamer_conditioned else None
        )
        self.stage1 = CorrectionStage(
            int(latent_channels), effective_condition_channels, int(hidden_channels)
        )
        self.stage2 = (
            CorrectionStage(int(latent_channels), effective_condition_channels, int(hidden_channels))
            if self.two_stage else None
        )
        self.highres_stage = (
            HighResolutionCorrection(
                int(latent_channels), effective_condition_channels, int(hidden_channels)
            ) if self.high_resolution else None
        )
        if self.spatial_gate_enabled:
            self.spatial_gate = nn.Conv2d(
                int(latent_channels) + effective_condition_channels, 1, 3, padding=1
            )
            nn.init.zeros_(self.spatial_gate.weight)
            nn.init.zeros_(self.spatial_gate.bias)

    def _hand_mask(self, hamer_image, target_size):
        image = hamer_image.float()
        if image.detach().amin() < -0.01:
            image = image.add(1).mul(0.5)
        mask = image.amax(dim=1, keepdim=True) > self.mask_threshold
        mask = F.interpolate(mask.float(), size=target_size, mode="nearest")
        if self.mask_dilation >= 1:
            mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        if self.mask_dilation >= 2:
            mask = F.max_pool2d(mask, 3, stride=1, padding=1)
        return mask

    def forward(
        self,
        predicted_x0: torch.Tensor,
        timesteps: torch.Tensor,
        hamer_image: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if predicted_x0.ndim != 5:
            raise ValueError(f"Expected predicted x0 [B,C,F,H,W], got {tuple(predicted_x0.shape)}")
        batch, _, frames, height, width = predicted_x0.shape
        flat = rearrange(predicted_x0, "b c f h w -> (b f) c h w")
        if self.hamer_conditioned or self.spatial_gate_enabled:
            if hamer_image is None or hamer_image.shape[0] != batch * frames:
                raise ValueError("HAMER image [B*F,3,H,W] is required by this C5.19 variant")
        condition = (
            self.hamer_conditioner(hamer_image, (height, width))
            if self.hamer_conditioned else None
        )
        timesteps = timesteps.reshape(batch, 1).expand(batch, frames).reshape(-1)
        timestep_map = (
            timesteps.to(device=flat.device, dtype=flat.dtype)
            / max(self.num_train_timesteps - 1, 1)
        ).view(-1, 1, 1, 1).expand(-1, 1, height, width)
        gate = flat.new_ones((batch * frames, 1, height, width))
        hand_mask = None
        if self.spatial_gate_enabled:
            hand_mask = self._hand_mask(hamer_image, (height, width)).to(flat.dtype)
            gate_input = torch.cat((flat, condition), dim=1) if condition is not None else flat
            gate = torch.sigmoid(self.spatial_gate(gate_input)) * hand_mask

        delta1 = self.stage1(flat, condition, timestep_map) * gate
        refined = flat + delta1
        delta2 = torch.zeros_like(delta1)
        if self.stage2 is not None:
            delta2 = self.stage2(refined, condition, timestep_map) * gate
            refined = refined + delta2
        delta_highres = torch.zeros_like(delta1)
        if self.highres_stage is not None:
            delta_highres = self.highres_stage(refined, condition, timestep_map) * gate
            refined = refined + delta_highres
        refined_5d = rearrange(refined, "(b f) c h w -> b c f h w", b=batch, f=frames)
        details = {
            "delta": rearrange(refined - flat, "(b f) c h w -> b c f h w", b=batch, f=frames),
            "delta_stage1": delta1,
            "delta_stage2": delta2,
            "delta_highres": delta_highres,
            "gate": gate,
        }
        if hand_mask is not None:
            details["hand_mask"] = hand_mask
        return refined_5d, details
