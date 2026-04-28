from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from utils.geometry import bilinear_sample_2d


def all_pairs_correlation_1d(
    fmap_left: torch.Tensor, fmap_right: torch.Tensor
) -> torch.Tensor:
    if fmap_left.shape != fmap_right.shape:
        raise ValueError(
            f"Feature shapes must match, got {fmap_left.shape} vs {fmap_right.shape}"
        )
    batch, channels, height, width = fmap_left.shape
    corr = torch.einsum("bchw,bchv->bhwv", fmap_left, fmap_right)
    corr = corr / math.sqrt(float(channels))
    return corr.reshape(batch, height, width, width)


class CorrelationPyramid1D:
    def __init__(
        self,
        fmap_left: torch.Tensor,
        fmap_right: torch.Tensor,
        *,
        num_levels: int,
        radius: int,
    ) -> None:
        if num_levels <= 0:
            raise ValueError(f"num_levels must be positive, got {num_levels}")
        if radius < 0:
            raise ValueError(f"radius must be non-negative, got {radius}")
        corr = all_pairs_correlation_1d(fmap_left, fmap_right)
        batch, height, width_left, width_right = corr.shape
        corr = corr.reshape(batch * height * width_left, 1, 1, width_right)
        self.batch = batch
        self.height = height
        self.width_left = width_left
        self.num_levels = num_levels
        self.radius = radius
        self.levels: list[torch.Tensor] = [corr]
        for _ in range(1, num_levels):
            corr = F.avg_pool2d(corr, kernel_size=(1, 2), stride=(1, 2))
            self.levels.append(corr)

    def sample(self, coords: torch.Tensor) -> torch.Tensor:
        if coords.ndim != 4 or coords.shape[1] != 2:
            raise ValueError(f"coords must have shape [B, 2, H, W], got {coords.shape}")
        if (
            coords.shape[0] != self.batch
            or coords.shape[2] != self.height
            or coords.shape[3] != self.width_left
        ):
            raise ValueError(
                "coords shape does not match the stored correlation pyramid: "
                f"got {coords.shape}, expected [B={self.batch}, 2, H={self.height}, W={self.width_left}]"
            )
        coords_x = coords[:, :1]
        outputs = []
        for level_index, corr_level in enumerate(self.levels):
            outputs.append(
                self._sample_single_level(corr_level, coords_x / (2**level_index))
            )
        return torch.cat(outputs, dim=1)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        return self.sample(coords)

    def _sample_single_level(
        self, corr_level: torch.Tensor, coords_x: torch.Tensor
    ) -> torch.Tensor:
        batch, _, height, width_left = coords_x.shape
        if (
            batch != self.batch
            or height != self.height
            or width_left != self.width_left
        ):
            raise ValueError("coords_x shape mismatch inside correlation sampling")
        num_samples = 2 * self.radius + 1
        offsets = torch.linspace(
            -self.radius,
            self.radius,
            steps=num_samples,
            device=coords_x.device,
            dtype=coords_x.dtype,
        )
        base_x = coords_x.permute(0, 2, 3, 1).reshape(
            batch * height * width_left, 1, 1, 1
        )
        sample_x = base_x + offsets.view(1, 1, num_samples, 1)
        sample_y = torch.zeros_like(sample_x)
        grid = torch.cat([sample_x, sample_y], dim=-1)
        sampled = bilinear_sample_2d(corr_level, grid)
        sampled = sampled.reshape(batch, height, width_left, num_samples)
        return sampled.permute(0, 3, 1, 2).contiguous().float()


CorrBlock1D = CorrelationPyramid1D
