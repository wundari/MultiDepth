# %%
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.cfg import DepthAnythingConfig

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from jaxtyping import Float


# %%


@dataclass(frozen=True)
class DepthAnythingOutput:
    inverse_depth: torch.Tensor
    penultimate: torch.Tensor


class DepthBackbone(Protocol):
    def __call__(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1
            ),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MockDepthAnythingBackbone(nn.Module):
    """
    MockDepthAnythingBackbone is a lightweight test/stub implementation of the DepthBackbone Protocol.
    Rather than using a real, heavyweight depth model, it provides a structurally identical but
    computationally cheap stand-in — useful for unit testing, debugging pipelines, or CI without
    needing real model weights.

    Input [B, 3, H, W]
        │
        ▼
    Conv2d (3→32,  stride=2)  + ReLU    →  [B, 32,  H/2,  W/2]
    Conv2d (32→64, stride=2)  + ReLU    →  [B, 64,  H/4,  W/4]
    Conv2d (64→feature_dim, stride=2) + ReLU  →  [B, feature_dim, H/8, W/8]   ← penultimate features
        │
        ▼
    depth_head Conv2d (feature_dim→1)
        │
        ▼
    softplus   →  depth [B, 1, H/8, W/8]   (guaranteed positive)

    softplus is used rather than ReLU to ensure depth values are strictly positive and smooth
    — physically sensible for a depth map.
    """

    def __init__(self, penultimate_dim: int = 128) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(32),
            nn.GELU(),
        )
        self.stage2 = ConvBlock(32, 64, stride=2)
        self.stage3 = ConvBlock(64, penultimate_dim, stride=2)
        self.refine = ConvBlock(penultimate_dim, penultimate_dim, stride=1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(penultimate_dim, penultimate_dim // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(penultimate_dim // 2, 1, kernel_size=1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        x = self.stage2(x)
        penultimate = self.refine(self.stage3(x))
        inverse_depth = self.depth_head(penultimate)
        return inverse_depth, penultimate


class FrozenDepthAnythingV2(nn.Module):
    """
    FrozenDepthAnythingV2 is a production wrapper around the real Depth Anything V2 model.
    It handles weight loading, input preprocessing, optional freezing, and output resizing —
    producing the same DepthAnythingBackboneOutput interface as the MockDepthAnythingBackbone
    seen earlier.

    """

    MODEL_CONFIGS = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    def __init__(
        self,
        backbone: nn.Module | None = None,
        config: DepthAnythingConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or DepthAnythingConfig()
        self.config = cfg
        self.backbone = backbone or MockDepthAnythingBackbone(
            penultimate_dim=cfg.penultimate_dim
        )
        if cfg.freeze_backbone:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("imagenet_mean", mean, persistent=False)
        self.register_buffer("imagenet_std", std, persistent=False)

    def forward(self, image_uint8: torch.Tensor) -> DepthAnythingOutput:
        if image_uint8.ndim != 4 or image_uint8.shape[1] != 3:
            raise ValueError(
                f"Expected raw RGB input [B,3,H,W], got {image_uint8.shape}"
            )

        batch, _, height, width = image_uint8.shape
        x = image_uint8 / 255.0
        if self.config.use_imagenet_normalization:
            x = (x - self.imagenet_mean.to(x.device, x.dtype)) / self.imagenet_std.to(
                x.device, x.dtype
            )

        resized_size = None
        if self.config.resize_long_side is not None:
            resized_size = self._compute_resized_hw(height, width)
            x = F.interpolate(x, size=resized_size, mode="bicubic", align_corners=False)

        inverse_depth, penultimate = self.backbone(x)

        target_size = (
            height // self.config.output_stride,
            width // self.config.output_stride,
        )
        if inverse_depth.shape[-2:] != target_size:
            inverse_depth = F.interpolate(
                inverse_depth, size=target_size, mode="bilinear", align_corners=False
            )
        if penultimate.shape[-2:] != target_size:
            penultimate = F.interpolate(
                penultimate, size=target_size, mode="bilinear", align_corners=False
            )

        return DepthAnythingOutput(inverse_depth=inverse_depth, penultimate=penultimate)

    def _compute_resized_hw(self, height: int, width: int) -> tuple[int, int]:
        target = self.config.resize_long_side
        if target is None:
            return height, width
        if height >= width:
            new_h = target
            new_w = int(round(width * (target / max(height, 1))))
        else:
            new_w = target
            new_h = int(round(height * (target / max(width, 1))))
        mult = max(1, self.config.resize_multiple_of)
        new_h = ((new_h + mult - 1) // mult) * mult
        new_w = ((new_w + mult - 1) // mult) * mult
        return new_h, new_w
