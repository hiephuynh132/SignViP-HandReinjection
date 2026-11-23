"""HAMER reinjection modules used by the C5/C6 architecture ablations."""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision.ops import roi_align


class HAMERResidualAdapter(nn.Module):
    """Project one 2D pre-Gate HAMER stream to a 5D U-Net feature tensor."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
        zero_init_output: bool = False,
    ):
        super().__init__()
        hidden_channels = min(int(hidden_channels), int(out_channels))
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.net = nn.Sequential(
            nn.Conv2d(self.in_channels, hidden_channels, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, self.out_channels, kernel_size=1),
        )
        if zero_init_output:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        hand_feature: torch.Tensor,
        target_shape: Tuple[int, ...],
    ) -> torch.Tensor:
        if hand_feature.ndim != 4:
            raise ValueError(
                f"Expected pre-Gate HAMER [B*F,C,H,W], got {tuple(hand_feature.shape)}"
            )
        if len(target_shape) != 5:
            raise ValueError(f"Expected target [B,C,F,H,W], got {tuple(target_shape)}")
        batch, channels, frames, height, width = map(int, target_shape)
        if channels != self.out_channels:
            raise ValueError(
                f"Adapter outputs {self.out_channels} channels but target has {channels}"
            )
        if hand_feature.shape[0] != batch * frames:
            raise ValueError(
                "Pre-Gate HAMER batch/frame mismatch: "
                f"BF={hand_feature.shape[0]}, B={batch}, F={frames}"
            )
        if hand_feature.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} HAMER channels, got {hand_feature.shape[1]}"
            )
        if hand_feature.shape[-2:] != (height, width):
            hand_feature = F.interpolate(
                hand_feature,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )
        residual = self.net(hand_feature)
        return rearrange(residual, "(b f) c h w -> b c f h w", b=batch, f=frames)


class HandGeometryEncoder(nn.Module):
    """Encode the raw HAMER render into a spatial, position-aware hand latent.

    The output stays at 1/4 input resolution (32x32 for the 128x128 setup),
    rather than passing through the main VAE's 8x bottleneck.  Explicit XY
    channels make the geometry tokens position-aware for local cross-attention.
    """

    def __init__(
        self,
        out_channels: int = 128,
        temporal_attention: bool = False,
        attention_heads: int = 4,
        mask_threshold: float = 0.05,
    ):
        super().__init__()
        out_channels = int(out_channels)
        if out_channels % int(attention_heads) != 0:
            raise ValueError("geometry_channels must be divisible by attention_heads")
        self.out_channels = out_channels
        self.temporal_attention_enabled = bool(temporal_attention)
        self.mask_threshold = float(mask_threshold)
        self.net = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        self.temporal_norm = nn.LayerNorm(out_channels)
        self.temporal_attention = nn.MultiheadAttention(
            out_channels, int(attention_heads), batch_first=True
        )

    @staticmethod
    def _to_zero_one(image: torch.Tensor) -> torch.Tensor:
        # SignViP dataset tensors are normally [-1, 1], while standalone
        # callers may already provide [0, 1].
        if image.detach().amin() < -0.01:
            return (image + 1.0) * 0.5
        return image

    def forward(
        self, image: torch.Tensor, *, batch: int, frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected raw HAMER [B*F,3,H,W], got {tuple(image.shape)}")
        if image.shape[0] != int(batch) * int(frames):
            raise ValueError(
                f"HAMER geometry BF={image.shape[0]} does not match B={batch}, F={frames}"
            )
        image_01 = self._to_zero_one(image).clamp(0.0, 1.0)
        height, width = image_01.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, width, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(image.shape[0], -1, -1, -1)
        geometry = self.net(torch.cat((image_01, coords), dim=1))
        mask = image_01.amax(dim=1, keepdim=True) > self.mask_threshold
        mask = F.interpolate(mask.float(), size=geometry.shape[-2:], mode="nearest")

        if self.temporal_attention_enabled and int(frames) > 1:
            _, channels, geo_h, geo_w = geometry.shape
            tokens = rearrange(
                geometry,
                "(b f) c h w -> (b h w) f c",
                b=int(batch),
                f=int(frames),
            )
            normalized = self.temporal_norm(tokens)
            attended, _ = self.temporal_attention(
                normalized, normalized, normalized, need_weights=False
            )
            tokens = tokens + attended
            geometry = rearrange(
                tokens,
                "(b h w) f c -> (b f) c h w",
                b=int(batch),
                h=geo_h,
                w=geo_w,
            )
        return geometry, mask


class HandEdgeEncoder(nn.Module):
    """Encode fixed spatial gradients of the HAMER render at 1/4 resolution."""

    def __init__(self, out_channels: int = 64, mask_threshold: float = 0.05):
        super().__init__()
        out_channels = int(out_channels)
        self.mask_threshold = float(mask_threshold)
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, image: torch.Tensor, *, batch: int, frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected raw HAMER [B*F,3,H,W], got {tuple(image.shape)}")
        if image.shape[0] != int(batch) * int(frames):
            raise ValueError(
                f"HAMER edge BF={image.shape[0]} does not match B={batch}, F={frames}"
            )
        image_01 = HandGeometryEncoder._to_zero_one(image).clamp(0.0, 1.0)
        # Channel-wise central differences preserve both the silhouette and
        # internal colour/depth discontinuities of the HAMER render.
        padded = F.pad(image_01, (1, 1, 1, 1), mode="replicate")
        dx = padded[:, :, 1:-1, 2:] - padded[:, :, 1:-1, :-2]
        dy = padded[:, :, 2:, 1:-1] - padded[:, :, :-2, 1:-1]
        edge = torch.sqrt(dx.square() + dy.square() + 1e-8).amax(dim=1, keepdim=True)
        edge = edge.clamp(0.0, 1.0)
        feature = self.net(edge)
        mask = image_01.amax(dim=1, keepdim=True) > self.mask_threshold
        mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        return feature, mask


class PixelUnshuffleGeometryEncoder(nn.Module):
    """Preserve raw HAMER pixels in channels before learning geometry features.

    For the 128x128 setup, PixelUnshuffle(4) maps RGB exactly to 48 channels
    at 32x32.  Two explicit XY channels are appended after the rearrangement,
    so the learned projection remains aware of absolute hand position.
    """

    def __init__(
        self,
        out_channels: int = 128,
        downscale_factor: int = 4,
        mask_threshold: float = 0.05,
    ):
        super().__init__()
        out_channels = int(out_channels)
        downscale_factor = int(downscale_factor)
        if downscale_factor < 1:
            raise ValueError("pixel_geometry_downscale_factor must be positive")
        self.out_channels = out_channels
        self.downscale_factor = downscale_factor
        self.mask_threshold = float(mask_threshold)
        packed_channels = 3 * downscale_factor * downscale_factor + 2
        self.net = nn.Sequential(
            nn.Conv2d(packed_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(
        self, image: torch.Tensor, *, batch: int, frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected raw HAMER [B*F,3,H,W], got {tuple(image.shape)}")
        if image.shape[0] != int(batch) * int(frames):
            raise ValueError(
                f"Pixel geometry BF={image.shape[0]} does not match B={batch}, F={frames}"
            )
        image_01 = HandGeometryEncoder._to_zero_one(image).clamp(0.0, 1.0)
        height, width = image_01.shape[-2:]
        factor = self.downscale_factor
        if height % factor or width % factor:
            raise ValueError(
                f"HAMER spatial size {(height, width)} must be divisible by "
                f"PixelUnshuffle factor {factor}"
            )
        packed = F.pixel_unshuffle(image_01, factor)
        packed_h, packed_w = packed.shape[-2:]
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, packed_h, device=image.device, dtype=image.dtype),
            torch.linspace(-1.0, 1.0, packed_w, device=image.device, dtype=image.dtype),
            indexing="ij",
        )
        coords = torch.stack((xx, yy), dim=0).unsqueeze(0).expand(
            image.shape[0], -1, -1, -1
        )
        feature = self.net(torch.cat((packed, coords), dim=1))
        mask = image_01.amax(dim=1, keepdim=True) > self.mask_threshold
        mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        return feature, mask


class HaarDWTHandEdgeEncoder(nn.Module):
    """Encode fixed two-level Haar high-pass bands from the HAMER render.

    Level-one LH/HL/HH bands are pixel-unshuffled from 64x64 to 32x32 and
    concatenated with level-two LH/HL/HH bands.  This keeps every level-one
    coefficient while presenting a single 32x32 tensor to the learned CNN.
    """

    def __init__(self, out_channels: int = 64, mask_threshold: float = 0.05):
        super().__init__()
        out_channels = int(out_channels)
        self.out_channels = out_channels
        self.mask_threshold = float(mask_threshold)
        # 3 RGB * 3 high-pass bands * 4 packed pixels + 3 RGB * 3 level-2 bands.
        wavelet_channels = 3 * 3 * 4 + 3 * 3
        self.net = nn.Sequential(
            nn.Conv2d(wavelet_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1),
        )

    @staticmethod
    def _haar_level(image: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if image.shape[-2] % 2 or image.shape[-1] % 2:
            raise ValueError(
                f"Haar DWT requires even spatial dimensions, got {tuple(image.shape[-2:])}"
            )
        x00 = image[..., 0::2, 0::2]
        x01 = image[..., 0::2, 1::2]
        x10 = image[..., 1::2, 0::2]
        x11 = image[..., 1::2, 1::2]
        # Orthonormal 2D Haar filters have a coefficient of 1/2.
        ll = (x00 + x01 + x10 + x11) * 0.5
        lh = (-x00 - x01 + x10 + x11) * 0.5
        hl = (-x00 + x01 - x10 + x11) * 0.5
        hh = (x00 - x01 - x10 + x11) * 0.5
        return ll, lh, hl, hh

    def forward(
        self, image: torch.Tensor, *, batch: int, frames: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"Expected raw HAMER [B*F,3,H,W], got {tuple(image.shape)}")
        if image.shape[0] != int(batch) * int(frames):
            raise ValueError(
                f"Wavelet edge BF={image.shape[0]} does not match B={batch}, F={frames}"
            )
        image_01 = HandGeometryEncoder._to_zero_one(image).clamp(0.0, 1.0)
        ll1, lh1, hl1, hh1 = self._haar_level(image_01)
        _, lh2, hl2, hh2 = self._haar_level(ll1)
        high1 = F.pixel_unshuffle(torch.cat((lh1, hl1, hh1), dim=1), 2)
        high2 = torch.cat((lh2, hl2, hh2), dim=1)
        if high1.shape[-2:] != high2.shape[-2:]:
            raise RuntimeError(
                f"Haar levels do not align: {tuple(high1.shape)} vs {tuple(high2.shape)}"
            )
        feature = self.net(torch.cat((high1, high2), dim=1))
        mask = image_01.amax(dim=1, keepdim=True) > self.mask_threshold
        mask = F.interpolate(mask.float(), size=feature.shape[-2:], mode="nearest")
        return feature, mask


class HandGeometryCrossAttention(nn.Module):
    """Cross-attend U-Net spatial tokens to masked geometry tokens."""

    def __init__(self, query_channels: int, geometry_channels: int, heads: int = 4):
        super().__init__()
        geometry_channels = int(geometry_channels)
        self.query_norm = nn.LayerNorm(int(query_channels))
        self.query_projection = nn.Linear(int(query_channels), geometry_channels)
        self.geometry_norm = nn.LayerNorm(geometry_channels)
        self.attention = nn.MultiheadAttention(
            geometry_channels, int(heads), batch_first=True
        )
        self.output_projection = nn.Linear(geometry_channels, int(query_channels))
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        geometry: torch.Tensor,
        geometry_mask: torch.Tensor,
        output_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch, _, frames, height, width = hidden_states.shape
        query = rearrange(hidden_states, "b c f h w -> (b f) (h w) c")
        key_value = rearrange(geometry, "(b f) c h w -> (b f) (h w) c", b=batch, f=frames)
        key_value = self.geometry_norm(key_value)
        # True entries are ignored by MultiheadAttention. Guarantee at least
        # one valid token for blank HAMER renders to avoid NaNs.
        valid = rearrange(geometry_mask, "(b f) 1 h w -> (b f) (h w)", b=batch, f=frames).bool()
        blank = ~valid.any(dim=1)
        if blank.any():
            valid = valid.clone()
            valid[blank, 0] = True
        attended, _ = self.attention(
            self.query_projection(self.query_norm(query)),
            key_value,
            key_value,
            key_padding_mask=~valid,
            need_weights=False,
        )
        attended = self.output_projection(attended)
        attended = rearrange(
            attended, "(b f) (h w) c -> b c f h w", b=batch, f=frames, h=height, w=width
        )
        target_mask = output_mask
        if target_mask is None:
            target_mask = F.interpolate(
                geometry_mask.float(), size=(height, width), mode="nearest"
            )
        target_mask = rearrange(target_mask, "(b f) 1 h w -> b 1 f h w", b=batch, f=frames)
        return attended * target_mask


class HAMERReinjectionModule(nn.Module):
    """Zero-gated C5.4 legacy sites or one configurable C6 Up16 site."""

    LEGACY_SITES = ("mid", "up_second_highest", "up_highest")
    GEOMETRY_SITES = LEGACY_SITES + ("pre_output",)
    UP16_LOCATIONS = (
        "baseline",
        "pre_resnet",
        "post_resnet",
        "post_spatial_attn",
        "post_temporal_attn",
        "pre_output",
    )

    def __init__(
        self,
        hand_channels: int,
        mid_channels: int,
        up_second_highest_channels: int,
        up_highest_channels: int,
        pre_output_channels: Optional[int] = None,
        hidden_channels: int = 128,
        enabled: bool = False,
        inject_mid: bool = False,
        inject_up_second_highest: bool = False,
        inject_up_highest: bool = False,
        hamer_up16_injection_location: Optional[str] = None,
        up16_channels: Optional[int] = None,
        up16_concat_channels: Optional[int] = None,
        log_shapes: bool = True,
        ratio_epsilon: float = 1e-6,
        geometry_enabled: bool = False,
        geometry_site: str = "up_highest",
        geometry_channels: int = 128,
        geometry_cross_attention: bool = False,
        geometry_timestep_gate: bool = False,
        geometry_temporal_attention: bool = False,
        geometry_attention_heads: int = 4,
        geometry_mask_threshold: float = 0.05,
        geometry_roi_enabled: bool = False,
        geometry_roi_dilation: int = 2,
        geometry_roi_ring1_weight: float = 0.5,
        geometry_roi_ring2_weight: float = 0.25,
        hand_edge_enabled: bool = False,
        hand_edge_site: str = "pre_output",
        hand_edge_channels: int = 64,
        hand_edge_mask_threshold: float = 0.05,
        hand_edge_roi_dilation: int = 2,
        hand_edge_roi_ring1_weight: float = 1.0,
        hand_edge_roi_ring2_weight: float = 1.0,
        hand_edge_residual_scale: float = 1.0,
        pixel_geometry_enabled: bool = False,
        pixel_geometry_site: str = "up_highest",
        pixel_geometry_channels: int = 128,
        pixel_geometry_downscale_factor: int = 4,
        pixel_geometry_mask_threshold: float = 0.05,
        pixel_geometry_roi_dilation: int = 2,
        pixel_geometry_roi_ring1_weight: float = 1.0,
        pixel_geometry_roi_ring2_weight: float = 1.0,
        wavelet_edge_enabled: bool = False,
        wavelet_edge_site: str = "up_highest",
        wavelet_edge_channels: int = 64,
        wavelet_edge_mask_threshold: float = 0.05,
        wavelet_edge_roi_dilation: int = 2,
        wavelet_edge_roi_ring1_weight: float = 1.0,
        wavelet_edge_roi_ring2_weight: float = 1.0,
        c518_multiscale_enabled: bool = False,
        c518_separate_hand_roi: bool = False,
        c518_local_cross_attention: bool = False,
        c518_attention_heads: int = 4,
        num_train_timesteps: int = 1000,
    ):
        super().__init__()
        self.enabled = bool(enabled)
        self.hamer_up16_injection_location = hamer_up16_injection_location
        self.log_shapes = bool(log_shapes)
        self.ratio_epsilon = float(ratio_epsilon)
        self.geometry_enabled = bool(geometry_enabled)
        self.geometry_site = str(geometry_site)
        self.geometry_cross_attention_enabled = bool(geometry_cross_attention)
        self.geometry_timestep_gate_enabled = bool(geometry_timestep_gate)
        self.geometry_roi_enabled = bool(geometry_roi_enabled)
        self.geometry_roi_dilation = int(geometry_roi_dilation)
        self.geometry_roi_ring1_weight = float(geometry_roi_ring1_weight)
        self.geometry_roi_ring2_weight = float(geometry_roi_ring2_weight)
        if self.geometry_roi_dilation not in (0, 1, 2):
            raise ValueError("geometry_roi_dilation must be 0, 1, or 2")
        self.num_train_timesteps = int(num_train_timesteps)
        self.hand_edge_enabled = bool(hand_edge_enabled)
        self.hand_edge_site = str(hand_edge_site)
        self.hand_edge_roi_dilation = int(hand_edge_roi_dilation)
        self.hand_edge_roi_ring1_weight = float(hand_edge_roi_ring1_weight)
        self.hand_edge_roi_ring2_weight = float(hand_edge_roi_ring2_weight)
        self.hand_edge_residual_scale = float(hand_edge_residual_scale)
        if self.hand_edge_residual_scale <= 0:
            raise ValueError("hand_edge_residual_scale must be positive")
        if self.hand_edge_site not in self.GEOMETRY_SITES:
            raise ValueError(
                f"Unknown hand_edge_site {self.hand_edge_site!r}; expected {self.GEOMETRY_SITES}"
            )
        if self.hand_edge_roi_dilation not in (0, 1, 2):
            raise ValueError("hand_edge_roi_dilation must be 0, 1, or 2")
        self.pixel_geometry_enabled = bool(pixel_geometry_enabled)
        self.pixel_geometry_site = str(pixel_geometry_site)
        self.pixel_geometry_roi_dilation = int(pixel_geometry_roi_dilation)
        self.pixel_geometry_roi_ring1_weight = float(pixel_geometry_roi_ring1_weight)
        self.pixel_geometry_roi_ring2_weight = float(pixel_geometry_roi_ring2_weight)
        self.wavelet_edge_enabled = bool(wavelet_edge_enabled)
        self.wavelet_edge_site = str(wavelet_edge_site)
        self.wavelet_edge_roi_dilation = int(wavelet_edge_roi_dilation)
        self.wavelet_edge_roi_ring1_weight = float(wavelet_edge_roi_ring1_weight)
        self.wavelet_edge_roi_ring2_weight = float(wavelet_edge_roi_ring2_weight)
        self.c518_multiscale_enabled = bool(c518_multiscale_enabled)
        self.c518_separate_hand_roi = bool(c518_separate_hand_roi)
        self.c518_local_cross_attention = bool(c518_local_cross_attention)
        for branch, site, dilation in (
            ("pixel geometry", self.pixel_geometry_site, self.pixel_geometry_roi_dilation),
            ("wavelet edge", self.wavelet_edge_site, self.wavelet_edge_roi_dilation),
        ):
            if site not in self.GEOMETRY_SITES:
                raise ValueError(f"Unknown {branch} site {site!r}; expected {self.GEOMETRY_SITES}")
            if dilation not in (0, 1, 2):
                raise ValueError(f"{branch} ROI dilation must be 0, 1, or 2")

        if hamer_up16_injection_location is None:
            # Preserve the C5.4-A module/state-dict layout exactly.
            self.SITES = self.LEGACY_SITES
            self.site_enabled = {
                "mid": bool(inject_mid),
                "up_second_highest": bool(inject_up_second_highest),
                "up_highest": bool(inject_up_highest),
            }
            adapters = {
                "mid": HAMERResidualAdapter(hand_channels, mid_channels, hidden_channels),
                "up_second_highest": HAMERResidualAdapter(
                    hand_channels, up_second_highest_channels, hidden_channels
                ),
                "up_highest": HAMERResidualAdapter(
                    hand_channels, up_highest_channels, hidden_channels
                ),
            }
            self.adapters = nn.ModuleDict(adapters)
            # Raw scalars are required for exact identity: sigmoid(0) would be 0.5.
            self.gamma_mid = nn.Parameter(torch.zeros(1))
            self.gamma_up_second_highest = nn.Parameter(torch.zeros(1))
            self.gamma_up_highest = nn.Parameter(torch.zeros(1))
        else:
            location = str(hamer_up16_injection_location)
            if location not in self.UP16_LOCATIONS:
                raise ValueError(
                    f"Unknown Up16 HAMER injection location {location!r}; "
                    f"expected one of {self.UP16_LOCATIONS}"
                )
            if any((inject_mid, inject_up_second_highest, inject_up_highest)):
                raise ValueError(
                    "C6 hamer_up16_injection_location cannot be combined with legacy "
                    "C5.4-A injection flags"
                )
            regular_channels = int(up16_channels or up_highest_channels)
            concat_channels = int(up16_concat_channels or (2 * regular_channels))
            if location == "baseline":
                target_channels = int(up_highest_channels)
            elif location in ("pre_resnet", "pre_output"):
                target_channels = concat_channels
            else:
                target_channels = regular_channels
            self.SITES = (location,)
            self.site_enabled = {location: self.enabled}
            self.adapters = nn.ModuleDict(
                {
                    location: HAMERResidualAdapter(
                        hand_channels, target_channels, hidden_channels
                    )
                }
            )
            self.register_parameter(f"gamma_{location}", nn.Parameter(torch.zeros(1)))

        if self.geometry_enabled:
            if self.hamer_up16_injection_location is not None:
                raise ValueError("C5.17 geometry cannot be combined with a C6 Up16 location")
            if self.geometry_site not in self.GEOMETRY_SITES:
                raise ValueError(
                    f"Unknown geometry_site {self.geometry_site!r}; expected {self.GEOMETRY_SITES}"
                )
            if self.geometry_site == "pre_output" and pre_output_channels is None:
                raise ValueError(
                    "pre_output_channels is required when geometry_site='pre_output'"
                )
            target_channels = {
                "mid": int(mid_channels),
                "up_second_highest": int(up_second_highest_channels),
                "up_highest": int(up_highest_channels),
                "pre_output": int(pre_output_channels or up_highest_channels),
            }[self.geometry_site]
            self.geometry_encoder = HandGeometryEncoder(
                out_channels=int(geometry_channels),
                temporal_attention=bool(geometry_temporal_attention),
                attention_heads=int(geometry_attention_heads),
                mask_threshold=float(geometry_mask_threshold),
            )
            self.geometry_adapter = HAMERResidualAdapter(
                int(geometry_channels),
                target_channels,
                hidden_channels,
                zero_init_output=True,
            )
            self.geometry_attention = (
                HandGeometryCrossAttention(
                    target_channels, int(geometry_channels), int(geometry_attention_heads)
                )
                if self.geometry_cross_attention_enabled
                else None
            )
            # The residual and attention output projections are zero-convs.
            # Start the multiplier at one so those projections receive a
            # gradient immediately while initial output remains exact C5.16.
            self.gamma_geometry = nn.Parameter(torch.ones(1))
        if self.hand_edge_enabled:
            if self.hamer_up16_injection_location is not None:
                raise ValueError("C5.17 hand edge cannot be combined with a C6 Up16 location")
            if self.hand_edge_site == "pre_output" and pre_output_channels is None:
                raise ValueError(
                    "pre_output_channels is required when hand_edge_site='pre_output'"
                )
            edge_target_channels = {
                "mid": int(mid_channels),
                "up_second_highest": int(up_second_highest_channels),
                "up_highest": int(up_highest_channels),
                "pre_output": int(pre_output_channels or up_highest_channels),
            }[self.hand_edge_site]
            self.hand_edge_encoder = HandEdgeEncoder(
                out_channels=int(hand_edge_channels),
                mask_threshold=float(hand_edge_mask_threshold),
            )
            self.hand_edge_adapter = HAMERResidualAdapter(
                int(hand_edge_channels),
                edge_target_channels,
                hidden_channels,
                zero_init_output=True,
            )
            self.gamma_hand_edge = nn.Parameter(torch.ones(1))
        if self.pixel_geometry_enabled:
            if self.hamer_up16_injection_location is not None:
                raise ValueError("C5.18 pixel geometry cannot be combined with a C6 Up16 location")
            if self.pixel_geometry_site == "pre_output" and pre_output_channels is None:
                raise ValueError(
                    "pre_output_channels is required when pixel_geometry_site='pre_output'"
                )
            pixel_target_channels = {
                "mid": int(mid_channels),
                "up_second_highest": int(up_second_highest_channels),
                "up_highest": int(up_highest_channels),
                "pre_output": int(pre_output_channels or up_highest_channels),
            }[self.pixel_geometry_site]
            self.pixel_geometry_encoder = PixelUnshuffleGeometryEncoder(
                out_channels=int(pixel_geometry_channels),
                downscale_factor=int(pixel_geometry_downscale_factor),
                mask_threshold=float(pixel_geometry_mask_threshold),
            )
            self.pixel_geometry_adapter = HAMERResidualAdapter(
                int(pixel_geometry_channels),
                pixel_target_channels,
                hidden_channels,
                zero_init_output=True,
            )
            self.gamma_pixel_geometry = nn.Parameter(torch.ones(1))
            if self.c518_multiscale_enabled:
                self.pixel_geometry_multiscale_adapters = nn.ModuleDict({
                    "mid": HAMERResidualAdapter(int(pixel_geometry_channels), int(mid_channels), hidden_channels, True),
                    "up_second_highest": HAMERResidualAdapter(int(pixel_geometry_channels), int(up_second_highest_channels), hidden_channels, True),
                    "up_highest": HAMERResidualAdapter(int(pixel_geometry_channels), int(up_highest_channels), hidden_channels, True),
                })
                self.gamma_pixel_geometry_multiscale = nn.ParameterDict({
                    site: nn.Parameter(torch.ones(1)) for site in self.LEGACY_SITES
                })
        if self.wavelet_edge_enabled:
            if self.hamer_up16_injection_location is not None:
                raise ValueError("C5.18 wavelet edge cannot be combined with a C6 Up16 location")
            if self.wavelet_edge_site == "pre_output" and pre_output_channels is None:
                raise ValueError(
                    "pre_output_channels is required when wavelet_edge_site='pre_output'"
                )
            wavelet_target_channels = {
                "mid": int(mid_channels),
                "up_second_highest": int(up_second_highest_channels),
                "up_highest": int(up_highest_channels),
                "pre_output": int(pre_output_channels or up_highest_channels),
            }[self.wavelet_edge_site]
            self.wavelet_edge_encoder = HaarDWTHandEdgeEncoder(
                out_channels=int(wavelet_edge_channels),
                mask_threshold=float(wavelet_edge_mask_threshold),
            )
            self.wavelet_edge_adapter = HAMERResidualAdapter(
                int(wavelet_edge_channels),
                wavelet_target_channels,
                hidden_channels,
                zero_init_output=True,
            )
            self.gamma_wavelet_edge = nn.Parameter(torch.ones(1))
            if self.c518_multiscale_enabled:
                self.wavelet_edge_multiscale_adapters = nn.ModuleDict({
                    "mid": HAMERResidualAdapter(int(wavelet_edge_channels), int(mid_channels), hidden_channels, True),
                    "up_second_highest": HAMERResidualAdapter(int(wavelet_edge_channels), int(up_second_highest_channels), hidden_channels, True),
                    "up_highest": HAMERResidualAdapter(int(wavelet_edge_channels), int(up_highest_channels), hidden_channels, True),
                })
                self.gamma_wavelet_edge_multiscale = nn.ParameterDict({
                    site: nn.Parameter(torch.ones(1)) for site in self.LEGACY_SITES
                })
        if self.c518_local_cross_attention:
            if not (self.pixel_geometry_enabled and self.c518_multiscale_enabled):
                raise ValueError("C5.18 local attention requires multiscale pixel geometry")
            self.c518_roi_attention = nn.ModuleDict({
                "mid": HandGeometryCrossAttention(int(mid_channels), int(pixel_geometry_channels), int(c518_attention_heads)),
                "up_second_highest": HandGeometryCrossAttention(int(up_second_highest_channels), int(pixel_geometry_channels), int(c518_attention_heads)),
                "up_highest": HandGeometryCrossAttention(int(up_highest_channels), int(pixel_geometry_channels), int(c518_attention_heads)),
            })
        self.last_stats: Dict[str, object] = {}

    def gamma(self, site: str) -> nn.Parameter:
        return getattr(self, f"gamma_{site}")

    def forward_site(
        self,
        site: str,
        hidden_states: torch.Tensor,
        hand_feature: Optional[torch.Tensor],
        geometry_feature: Optional[torch.Tensor] = None,
        geometry_mask: Optional[torch.Tensor] = None,
        hand_edge_feature: Optional[torch.Tensor] = None,
        hand_edge_mask: Optional[torch.Tensor] = None,
        pixel_geometry_feature: Optional[torch.Tensor] = None,
        pixel_geometry_mask: Optional[torch.Tensor] = None,
        wavelet_edge_feature: Optional[torch.Tensor] = None,
        wavelet_edge_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        legacy_site = site in self.SITES
        geometry_site = self.geometry_enabled and site == self.geometry_site
        edge_site = self.hand_edge_enabled and site == self.hand_edge_site
        pixel_site = self.pixel_geometry_enabled and (
            site == self.pixel_geometry_site
            or (self.c518_multiscale_enabled and site in self.LEGACY_SITES)
        )
        wavelet_site = self.wavelet_edge_enabled and (
            site == self.wavelet_edge_site
            or (self.c518_multiscale_enabled and site in self.LEGACY_SITES)
        )
        if not any((legacy_site, geometry_site, edge_site, pixel_site, wavelet_site)):
            raise KeyError(f"Unknown HAMER reinjection site: {site}")
        output = hidden_states
        if legacy_site and self.enabled and self.site_enabled[site]:
            if hand_feature is None:
                raise ValueError(f"HAMER pre-Gate feature is required for enabled site {site}")
            hand_feature = hand_feature.to(device=hidden_states.device, dtype=hidden_states.dtype)
            residual = self.adapters[site](hand_feature, tuple(hidden_states.shape))
            if residual.shape != hidden_states.shape:
                raise RuntimeError(
                    f"HAMER {site} residual {tuple(residual.shape)} does not match "
                    f"hidden state {tuple(hidden_states.shape)}"
                )
            gamma = self.gamma(site).to(dtype=hidden_states.dtype)
            delta = gamma * residual
            output = output + delta
            if collect_stats:
                self._record_stats(site, hidden_states, residual, delta, self.gamma(site))

        if self.geometry_enabled and site == self.geometry_site:
            if geometry_feature is None or geometry_mask is None:
                raise ValueError("Encoded HAMER geometry and mask are required for C5.17")
            geometry_feature = geometry_feature.to(device=hidden_states.device, dtype=hidden_states.dtype)
            geometry_mask = geometry_mask.to(device=hidden_states.device)
            residual = self.geometry_adapter(geometry_feature, tuple(hidden_states.shape))
            roi_mask = None
            if self.geometry_roi_enabled:
                roi_mask = self._geometry_roi_mask(
                    geometry_mask, hidden_states.shape[-2:]
                ).to(dtype=hidden_states.dtype)
                roi_mask_5d = rearrange(
                    roi_mask,
                    "(b f) 1 h w -> b 1 f h w",
                    b=int(hidden_states.shape[0]),
                    f=int(hidden_states.shape[2]),
                )
                residual = residual * roi_mask_5d
            if self.geometry_attention is not None:
                residual = residual + self.geometry_attention(
                    hidden_states,
                    geometry_feature,
                    geometry_mask,
                    output_mask=roi_mask,
                )
            gate = self.gamma_geometry.to(dtype=hidden_states.dtype)
            if self.geometry_timestep_gate_enabled:
                if timesteps is None:
                    raise ValueError("timesteps are required by the C5.17 timestep gate")
                schedule = 1.0 - timesteps.to(hidden_states.dtype) / max(
                    self.num_train_timesteps - 1, 1
                )
                gate = gate * schedule.clamp(0.0, 1.0).view(-1, 1, 1, 1, 1)
            delta = gate * residual
            output = output + delta
            if collect_stats:
                self._record_stats(
                    f"geometry_{site}", hidden_states, residual, delta, self.gamma_geometry
                )
        if edge_site:
            if hand_edge_feature is None or hand_edge_mask is None:
                raise ValueError("Encoded HAMER edge and mask are required for hand-edge injection")
            hand_edge_feature = hand_edge_feature.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            hand_edge_mask = hand_edge_mask.to(device=hidden_states.device)
            edge_residual = self.hand_edge_adapter(
                hand_edge_feature, tuple(hidden_states.shape)
            )
            edge_roi = self._hand_edge_roi_mask(
                hand_edge_mask, hidden_states.shape[-2:]
            ).to(dtype=hidden_states.dtype)
            edge_roi_5d = rearrange(
                edge_roi,
                "(b f) 1 h w -> b 1 f h w",
                b=int(hidden_states.shape[0]),
                f=int(hidden_states.shape[2]),
            )
            edge_residual = edge_residual * edge_roi_5d
            edge_delta = (
                self.hand_edge_residual_scale
                * self.gamma_hand_edge.to(dtype=hidden_states.dtype)
                * edge_residual
            )
            output = output + edge_delta
            if collect_stats:
                self._record_stats(
                    f"hand_edge_{site}",
                    hidden_states,
                    edge_residual,
                    edge_delta,
                    self.gamma_hand_edge,
                )
        if pixel_site:
            if pixel_geometry_feature is None or pixel_geometry_mask is None:
                raise ValueError("Encoded PixelUnshuffle geometry and mask are required")
            pixel_geometry_feature = pixel_geometry_feature.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            pixel_geometry_mask = pixel_geometry_mask.to(device=hidden_states.device)
            pixel_source = pixel_geometry_feature
            if self.c518_multiscale_enabled and site in self.LEGACY_SITES:
                source_size = {"mid": 8, "up_second_highest": 16, "up_highest": 32}[site]
                pixel_source = F.adaptive_avg_pool2d(pixel_source, (source_size, source_size))
            pixel_adapter = (
                self.pixel_geometry_multiscale_adapters[site]
                if self.c518_multiscale_enabled and site in self.LEGACY_SITES
                else self.pixel_geometry_adapter
            )
            pixel_residual = pixel_adapter(
                pixel_source, tuple(hidden_states.shape)
            )
            pixel_roi = self._expanded_roi_mask(
                pixel_geometry_mask,
                hidden_states.shape[-2:],
                dilation=self.pixel_geometry_roi_dilation,
                ring1_weight=self.pixel_geometry_roi_ring1_weight,
                ring2_weight=self.pixel_geometry_roi_ring2_weight,
            ).to(dtype=hidden_states.dtype)
            pixel_residual = pixel_residual * rearrange(
                pixel_roi,
                "(b f) 1 h w -> b 1 f h w",
                b=int(hidden_states.shape[0]),
                f=int(hidden_states.shape[2]),
            )
            pixel_gamma = (
                self.gamma_pixel_geometry_multiscale[site]
                if self.c518_multiscale_enabled and site in self.LEGACY_SITES
                else self.gamma_pixel_geometry
            )
            if self.c518_local_cross_attention and site in self.LEGACY_SITES:
                pixel_residual = pixel_residual + self.c518_roi_attention[site](
                    hidden_states, pixel_source,
                    F.interpolate(pixel_geometry_mask.float(), size=pixel_source.shape[-2:], mode="nearest"),
                    output_mask=pixel_roi,
                )
            pixel_delta = pixel_gamma.to(dtype=hidden_states.dtype) * pixel_residual
            output = output + pixel_delta
            if collect_stats:
                self._record_stats(
                    f"pixel_geometry_{site}",
                    hidden_states,
                    pixel_residual,
                    pixel_delta,
                    pixel_gamma,
                )
        if wavelet_site:
            if wavelet_edge_feature is None or wavelet_edge_mask is None:
                raise ValueError("Encoded Haar-DWT edge and mask are required")
            wavelet_edge_feature = wavelet_edge_feature.to(
                device=hidden_states.device, dtype=hidden_states.dtype
            )
            wavelet_edge_mask = wavelet_edge_mask.to(device=hidden_states.device)
            wavelet_source = wavelet_edge_feature
            if self.c518_multiscale_enabled and site in self.LEGACY_SITES:
                source_size = {"mid": 8, "up_second_highest": 16, "up_highest": 32}[site]
                wavelet_source = F.adaptive_avg_pool2d(wavelet_source, (source_size, source_size))
            wavelet_adapter = (
                self.wavelet_edge_multiscale_adapters[site]
                if self.c518_multiscale_enabled and site in self.LEGACY_SITES
                else self.wavelet_edge_adapter
            )
            wavelet_residual = wavelet_adapter(
                wavelet_source, tuple(hidden_states.shape)
            )
            wavelet_roi = self._expanded_roi_mask(
                wavelet_edge_mask,
                hidden_states.shape[-2:],
                dilation=self.wavelet_edge_roi_dilation,
                ring1_weight=self.wavelet_edge_roi_ring1_weight,
                ring2_weight=self.wavelet_edge_roi_ring2_weight,
            ).to(dtype=hidden_states.dtype)
            wavelet_residual = wavelet_residual * rearrange(
                wavelet_roi,
                "(b f) 1 h w -> b 1 f h w",
                b=int(hidden_states.shape[0]),
                f=int(hidden_states.shape[2]),
            )
            wavelet_gamma = (
                self.gamma_wavelet_edge_multiscale[site]
                if self.c518_multiscale_enabled and site in self.LEGACY_SITES
                else self.gamma_wavelet_edge
            )
            wavelet_delta = wavelet_gamma.to(dtype=hidden_states.dtype) * wavelet_residual
            output = output + wavelet_delta
            if collect_stats:
                self._record_stats(
                    f"wavelet_edge_{site}",
                    hidden_states,
                    wavelet_residual,
                    wavelet_delta,
                    wavelet_gamma,
                )
        return output

    @staticmethod
    def _expanded_roi_mask(
        hand_mask,
        target_size,
        *,
        dilation: int,
        ring1_weight: float,
        ring2_weight: float,
    ):
        core = F.interpolate(
            hand_mask.float(), size=tuple(target_size), mode="nearest"
        ).clamp(0.0, 1.0)
        if int(dilation) == 0:
            return core
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        roi = core + float(ring1_weight) * (dilated_1 - core)
        if int(dilation) == 1:
            return roi
        dilated_2 = F.max_pool2d(core, kernel_size=5, stride=1, padding=2)
        return roi + float(ring2_weight) * (dilated_2 - dilated_1)

    def _geometry_roi_mask(self, geometry_mask, target_size):
        """Build a soft hand core + two-ring ROI at the U-Net resolution."""
        core = F.interpolate(
            geometry_mask.float(), size=tuple(target_size), mode="nearest"
        ).clamp(0.0, 1.0)
        if self.geometry_roi_dilation == 0:
            return core
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        roi = core + self.geometry_roi_ring1_weight * (dilated_1 - core)
        if self.geometry_roi_dilation == 1:
            return roi
        dilated_2 = F.max_pool2d(core, kernel_size=5, stride=1, padding=2)
        return roi + self.geometry_roi_ring2_weight * (dilated_2 - dilated_1)

    def _hand_edge_roi_mask(self, hand_mask, target_size):
        """Build the hard expanded hand ROI used by C5.17-I."""
        core = F.interpolate(
            hand_mask.float(), size=tuple(target_size), mode="nearest"
        ).clamp(0.0, 1.0)
        if self.hand_edge_roi_dilation == 0:
            return core
        dilated_1 = F.max_pool2d(core, kernel_size=3, stride=1, padding=1)
        roi = core + self.hand_edge_roi_ring1_weight * (dilated_1 - core)
        if self.hand_edge_roi_dilation == 1:
            return roi
        dilated_2 = F.max_pool2d(core, kernel_size=5, stride=1, padding=2)
        return roi + self.hand_edge_roi_ring2_weight * (dilated_2 - dilated_1)

    def _record_stats(self, name, hidden_states, residual, delta, gamma):
        with torch.no_grad():
            hidden_norm = hidden_states.detach().float().norm()
            residual_norm = residual.detach().float().norm()
            delta_norm = delta.detach().float().norm()
            self.last_stats[name] = {
                "hidden_shape": tuple(hidden_states.shape),
                "residual_shape": tuple(residual.shape),
                "gamma": float(gamma.detach().float().item()),
                "hidden_norm": float(hidden_norm.item()),
                "residual_norm": float(residual_norm.item()),
                "delta_norm": float(delta_norm.item()),
                "ratio": float((delta_norm / (hidden_norm + self.ratio_epsilon)).item()),
            }

    def encode_geometry(
        self, image: Optional[torch.Tensor], *, batch: int, frames: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.geometry_enabled:
            return None, None
        if image is None:
            raise ValueError("Raw HAMER image is required when hand geometry is enabled")
        return self.geometry_encoder(image, batch=int(batch), frames=int(frames))

    def encode_hand_edge(
        self, image: Optional[torch.Tensor], *, batch: int, frames: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.hand_edge_enabled:
            return None, None
        if image is None:
            raise ValueError("Raw HAMER image is required when hand edge is enabled")
        return self.hand_edge_encoder(image, batch=int(batch), frames=int(frames))

    def encode_pixel_geometry(
        self,
        image: Optional[torch.Tensor],
        *,
        batch: int,
        frames: int,
        left_roi_mask: Optional[torch.Tensor] = None,
        right_roi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.pixel_geometry_enabled:
            return None, None
        if image is None:
            raise ValueError("Raw HAMER image is required for PixelUnshuffle geometry")
        if self.c518_separate_hand_roi:
            return self._encode_separate_rois(
                self.pixel_geometry_encoder, image, left_roi_mask, right_roi_mask,
                batch=int(batch), frames=int(frames), output_size=128,
            )
        return self.pixel_geometry_encoder(image, batch=int(batch), frames=int(frames))

    def encode_wavelet_edge(
        self,
        image: Optional[torch.Tensor],
        *,
        batch: int,
        frames: int,
        left_roi_mask: Optional[torch.Tensor] = None,
        right_roi_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.wavelet_edge_enabled:
            return None, None
        if image is None:
            raise ValueError("Raw HAMER image is required for Haar-DWT edge")
        if self.c518_separate_hand_roi:
            return self._encode_separate_rois(
                self.wavelet_edge_encoder, image, left_roi_mask, right_roi_mask,
                batch=int(batch), frames=int(frames), output_size=128,
            )
        return self.wavelet_edge_encoder(image, batch=int(batch), frames=int(frames))

    @staticmethod
    def _roi_boxes(mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return torchvision ROIAlign boxes and a presence flag for BF masks."""
        if mask.ndim != 4 or mask.shape[1] != 1:
            raise ValueError(f"Expected ROI mask [B*F,1,H,W], got {tuple(mask.shape)}")
        boxes = []
        present = []
        height, width = mask.shape[-2:]
        for index, item in enumerate(mask[:, 0]):
            coordinates = torch.nonzero(item > 0.5, as_tuple=False)
            if coordinates.numel():
                y0, x0 = coordinates.amin(dim=0)
                y1, x1 = coordinates.amax(dim=0)
                # roi_align uses inclusive floating endpoints.
                boxes.append(torch.stack((
                    item.new_tensor(float(index)), x0.float(), y0.float(),
                    (x1 + 1).float().clamp_max(float(width)),
                    (y1 + 1).float().clamp_max(float(height)),
                )))
                present.append(True)
            else:
                boxes.append(item.new_tensor((float(index), 0.0, 0.0, 1.0, 1.0)))
                present.append(False)
        return torch.stack(boxes), torch.tensor(present, device=mask.device, dtype=torch.bool)

    def _encode_separate_rois(
        self, encoder, image, left_roi_mask, right_roi_mask, *, batch, frames, output_size
    ):
        """ROIAlign left/right hands, encode independently, then paste spatially."""
        if left_roi_mask is None or right_roi_mask is None:
            raise ValueError("C5.18-E/F/G require left and right DWPose ROI masks")
        expected = int(batch) * int(frames)
        masks = []
        crops = []
        boxes_by_side = []
        present_by_side = []
        for roi_mask in (left_roi_mask, right_roi_mask):
            roi_mask = roi_mask.to(device=image.device)
            if roi_mask.shape[0] != expected:
                raise ValueError(f"ROI BF={roi_mask.shape[0]} does not match {expected}")
            boxes, present = self._roi_boxes(roi_mask)
            crops.append(roi_align(image, boxes, output_size=(output_size, output_size), aligned=True))
            masks.append(roi_mask)
            boxes_by_side.append(boxes)
            present_by_side.append(present)

        encoded = []
        for crop, present in zip(crops, present_by_side):
            feature, _ = encoder(crop, batch=int(batch), frames=int(frames))
            encoded.append(feature * present[:, None, None, None].to(feature.dtype))

        feature_h, feature_w = encoded[0].shape[-2:]
        canvas = encoded[0].new_zeros((expected, encoded[0].shape[1], feature_h, feature_w))
        image_h, image_w = image.shape[-2:]
        for side_feature, boxes, present in zip(encoded, boxes_by_side, present_by_side):
            for index in range(expected):
                if not bool(present[index]):
                    continue
                x0, y0, x1, y1 = boxes[index, 1:]
                fx0 = max(0, int(torch.floor(x0 * feature_w / image_w).item()))
                fy0 = max(0, int(torch.floor(y0 * feature_h / image_h).item()))
                fx1 = min(feature_w, int(torch.ceil(x1 * feature_w / image_w).item()))
                fy1 = min(feature_h, int(torch.ceil(y1 * feature_h / image_h).item()))
                if fx1 > fx0 and fy1 > fy0:
                    pasted = F.interpolate(
                        side_feature[index:index + 1], size=(fy1 - fy0, fx1 - fx0),
                        mode="bilinear", align_corners=False,
                    )[0]
                    canvas[index, :, fy0:fy1, fx0:fx1] = (
                        canvas[index, :, fy0:fy1, fx0:fx1] + pasted
                    )
        combined_mask = torch.maximum(masks[0].float(), masks[1].float())
        combined_mask = F.interpolate(combined_mask, size=(feature_h, feature_w), mode="nearest")
        return canvas, combined_mask

    def set_geometry_trainable(self, encoder: bool, fusion: bool) -> None:
        if not self.geometry_enabled:
            if encoder or fusion:
                raise ValueError("Geometry training requested but geometry_enabled is false")
            return
        self.geometry_encoder.requires_grad_(bool(encoder))
        if not self.geometry_encoder.temporal_attention_enabled:
            self.geometry_encoder.temporal_norm.requires_grad_(False)
            self.geometry_encoder.temporal_attention.requires_grad_(False)
        self.geometry_adapter.requires_grad_(bool(fusion))
        if self.geometry_attention is not None:
            self.geometry_attention.requires_grad_(bool(fusion))
        self.gamma_geometry.requires_grad_(bool(fusion))

    def set_hand_edge_trainable(self, encoder: bool, fusion: bool) -> None:
        if not self.hand_edge_enabled:
            if encoder or fusion:
                raise ValueError("Hand-edge training requested but hand_edge_enabled is false")
            return
        self.hand_edge_encoder.requires_grad_(bool(encoder))
        self.hand_edge_adapter.requires_grad_(bool(fusion))
        self.gamma_hand_edge.requires_grad_(bool(fusion))

    def set_pixel_geometry_trainable(self, encoder: bool, fusion: bool) -> None:
        if not self.pixel_geometry_enabled:
            if encoder or fusion:
                raise ValueError("Pixel geometry training requested but branch is disabled")
            return
        self.pixel_geometry_encoder.requires_grad_(bool(encoder))
        single_scale_fusion = bool(fusion) and not self.c518_multiscale_enabled
        self.pixel_geometry_adapter.requires_grad_(single_scale_fusion)
        self.gamma_pixel_geometry.requires_grad_(single_scale_fusion)
        if self.c518_multiscale_enabled:
            self.pixel_geometry_multiscale_adapters.requires_grad_(bool(fusion))
            self.gamma_pixel_geometry_multiscale.requires_grad_(bool(fusion))
        if self.c518_local_cross_attention:
            self.c518_roi_attention.requires_grad_(bool(fusion))

    def set_wavelet_edge_trainable(self, encoder: bool, fusion: bool) -> None:
        if not self.wavelet_edge_enabled:
            if encoder or fusion:
                raise ValueError("Wavelet edge training requested but branch is disabled")
            return
        self.wavelet_edge_encoder.requires_grad_(bool(encoder))
        single_scale_fusion = bool(fusion) and not self.c518_multiscale_enabled
        self.wavelet_edge_adapter.requires_grad_(single_scale_fusion)
        self.gamma_wavelet_edge.requires_grad_(single_scale_fusion)
        if self.c518_multiscale_enabled:
            self.wavelet_edge_multiscale_adapters.requires_grad_(bool(fusion))
            self.gamma_wavelet_edge_multiscale.requires_grad_(bool(fusion))

    def forward_token_site(
        self,
        site: str,
        hidden_states: torch.Tensor,
        hand_feature: Optional[torch.Tensor],
        *,
        batch: int,
        frames: int,
        height: int,
        width: int,
        collect_stats: bool = False,
    ) -> torch.Tensor:
        """Inject at an attention-token boundary without changing adapter design."""
        if hidden_states.ndim != 3:
            raise ValueError(
                f"Expected attention tokens [B*F,H*W,C], got {tuple(hidden_states.shape)}"
            )
        channels = int(hidden_states.shape[-1])
        feature_5d = rearrange(
            hidden_states,
            "(b f) (h w) c -> b c f h w",
            b=int(batch),
            f=int(frames),
            h=int(height),
            w=int(width),
        )
        feature_5d = self.forward_site(
            site, feature_5d, hand_feature, collect_stats=collect_stats
        )
        if int(feature_5d.shape[1]) != channels:
            raise RuntimeError("HAMER token reinjection changed the channel count")
        return rearrange(feature_5d, "b c f h w -> (b f) (h w) c")

    def extra_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
