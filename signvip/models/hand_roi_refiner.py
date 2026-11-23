"""Small post-decoder RGB hand refiner used by the C5.21 experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = min(8, channels)
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.net(inputs)


class HandROIRefiner(nn.Module):
    """Predict a bounded RGB residual and confidence alpha for a hand crop.

    The final convolution is zero initialized, so a new model is exactly an
    identity mapping.  The same module is shared by left and right hands.
    """

    def __init__(
        self,
        keypoint_channels: int = 21,
        geometry_channels: int = 1,
        base_channels: int = 32,
        residual_scale: float = 0.15,
    ):
        super().__init__()
        self.input_channels = 3 + 1 + int(keypoint_channels) + int(geometry_channels)
        self.residual_scale = float(residual_scale)
        width = int(base_channels)
        self.stem = nn.Conv2d(self.input_channels, width, 3, padding=1)
        self.encoder1 = nn.Sequential(ResidualBlock(width), ResidualBlock(width))
        self.down1 = nn.Conv2d(width, width * 2, 4, stride=2, padding=1)
        self.encoder2 = nn.Sequential(ResidualBlock(width * 2), ResidualBlock(width * 2))
        self.down2 = nn.Conv2d(width * 2, width * 4, 4, stride=2, padding=1)
        self.middle = nn.Sequential(ResidualBlock(width * 4), ResidualBlock(width * 4))
        self.up2 = nn.Conv2d(width * 4 + width * 2, width * 2, 3, padding=1)
        self.decoder2 = ResidualBlock(width * 2)
        self.up1 = nn.Conv2d(width * 2 + width, width, 3, padding=1)
        self.decoder1 = ResidualBlock(width)
        self.output = nn.Conv2d(width, 4, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def feather(mask: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
        if kernel_size <= 1:
            return mask.clamp(0, 1)
        if kernel_size % 2 == 0:
            raise ValueError("feather kernel_size must be odd")
        padding = kernel_size // 2
        expanded = F.max_pool2d(mask.float(), 3, stride=1, padding=1)
        return F.avg_pool2d(expanded, kernel_size, stride=1, padding=padding).clamp(0, 1)

    def forward(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
        keypoint_heatmaps: torch.Tensor,
        geometry: torch.Tensor,
    ):
        inputs = torch.cat((rgb, mask, keypoint_heatmaps, geometry), dim=1)
        if inputs.shape[1] != self.input_channels:
            raise ValueError(
                f"Expected {self.input_channels} input channels, got {inputs.shape[1]}"
            )
        skip1 = self.encoder1(self.stem(inputs))
        skip2 = self.encoder2(self.down1(skip1))
        hidden = self.middle(self.down2(skip2))
        hidden = F.interpolate(hidden, size=skip2.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decoder2(self.up2(torch.cat((hidden, skip2), dim=1)))
        hidden = F.interpolate(hidden, size=skip1.shape[-2:], mode="bilinear", align_corners=False)
        hidden = self.decoder1(self.up1(torch.cat((hidden, skip1), dim=1)))
        raw = self.output(hidden)
        delta = self.residual_scale * torch.tanh(raw[:, :3])
        alpha = torch.sigmoid(raw[:, 3:4])
        blend_mask = self.feather(mask)
        refined = (rgb + delta * alpha * blend_mask).clamp(0, 1)
        return refined, {"delta": delta, "alpha": alpha, "blend_mask": blend_mask}
