from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.geometry import resize_flow


def convex_upsample_flow(
    flow: torch.Tensor, mask_logits: torch.Tensor, *, factor: int
) -> torch.Tensor:
    """Convex upsampling from RAFT-style optical flow / stereo models.

    Parameters
    ----------
    flow:
        Tensor[B, D, H, W]
    mask_logits:
        Tensor[B, 9 * factor^2, H, W]
    factor:
        Spatial upsampling factor.
    """
    if flow.ndim != 4:
        raise ValueError(f"flow must have shape [B, D, H, W], got {flow.shape}")
    if mask_logits.ndim != 4:
        raise ValueError(
            f"mask_logits must have shape [B, 9*factor^2, H, W], got {mask_logits.shape}"
        )

    batch, channels, height, width = flow.shape
    expected = 9 * (factor**2)
    if mask_logits.shape[1] != expected:
        raise ValueError(
            f"mask channel mismatch: expected {expected} for factor={factor}, got {mask_logits.shape[1]}"
        )

    mask = mask_logits.view(batch, 1, 9, factor, factor, height, width)
    mask = torch.softmax(mask, dim=2)

    unfolded = F.unfold(factor * flow, kernel_size=3, padding=1)
    unfolded = unfolded.view(batch, channels, 9, 1, 1, height, width)

    upsampled = torch.sum(mask * unfolded, dim=2)
    upsampled = upsampled.permute(0, 1, 4, 2, 5, 3)
    return upsampled.reshape(batch, channels, factor * height, factor * width)


def upsample_flow(
    flow: torch.Tensor, *, factor: int, mask_logits: torch.Tensor | None = None
) -> torch.Tensor:
    if mask_logits is None:
        return resize_flow(flow, factor)
    return convex_upsample_flow(flow, mask_logits, factor=factor)
