from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_LBP_OFFSETS: tuple[tuple[int, int], ...] = ((-1, -1), (1, 1), (1, -1), (-1, 1))


@dataclass(frozen=True)
class LBPConfig:
    neighbor_offsets: tuple[tuple[int, int], ...] = DEFAULT_LBP_OFFSETS
    compare_mode: str = "ge"


class LocalBinaryPattern(nn.Module):
    """LBP encoder for single-channel disparity-like maps.

    The released project exposes diagonal offsets by default in its inference
    arguments. This module keeps the offsets configurable and returns one binary
    channel per neighbor.
    """

    def __init__(self, config: LBPConfig | None = None) -> None:
        super().__init__()
        cfg = config or LBPConfig()
        self.config = cfg
        self.neighbor_offsets = tuple(cfg.neighbor_offsets)
        self.num_neighbors = len(self.neighbor_offsets)
        if self.num_neighbors == 0:
            raise ValueError("LocalBinaryPattern requires at least one neighbor offset")
        if cfg.compare_mode not in {"ge", "gt"}:
            raise ValueError(f"Unsupported compare_mode={cfg.compare_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"LBP expects [B,1,H,W], got {x.shape}")

        codes = []
        for dy, dx in self.neighbor_offsets:
            neighbor = shift_with_replicate_pad(x, dy=dy, dx=dx)
            if self.config.compare_mode == "ge":
                code = (neighbor >= x).to(x.dtype)
            else:
                code = (neighbor > x).to(x.dtype)
            codes.append(code)
        return torch.cat(codes, dim=1)


def shift_with_replicate_pad(x: torch.Tensor, *, dy: int, dx: int) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"Expected 4D tensor, got {x.shape}")
    pad_left = max(dx, 0)
    pad_right = max(-dx, 0)
    pad_top = max(dy, 0)
    pad_bottom = max(-dy, 0)
    padded = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")

    height, width = x.shape[-2:]
    y0 = pad_bottom
    x0 = pad_right
    return padded[:, :, y0 : y0 + height, x0 : x0 + width]
