from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_uint8_image(image: torch.Tensor) -> torch.Tensor:
    """Map raw 0..255 RGB tensors to [-1, 1].

    Parameters
    ----------
    image:
        Tensor[B, 3, H, W] in raw image range.
    """
    return (2.0 * (image / 255.0) - 1.0).contiguous()


def make_coords_grid(batch: int, height: int, width: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Create an (x, y) pixel-coordinate grid.

    Returns
    -------
    Tensor[B, 2, H, W]
        Channel 0 stores x coordinates, channel 1 stores y coordinates.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    grid = torch.stack([xs, ys], dim=0)
    return grid.unsqueeze(0).repeat(batch, 1, 1, 1)


def bilinear_sample_2d(image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    """Sample `image` at pixel coordinates.

    Parameters
    ----------
    image:
        Tensor[N, C, H, W]
    coords:
        Tensor[N, H_out, W_out, 2] with pixel coordinates in (x, y) order.
    """
    height, width = image.shape[-2:]
    x, y = coords.split([1, 1], dim=-1)
    x = 2.0 * x / max(width - 1, 1) - 1.0
    if height > 1:
        y = 2.0 * y / max(height - 1, 1) - 1.0
    else:
        y = torch.zeros_like(y)
    grid = torch.cat([x, y], dim=-1)
    return F.grid_sample(image, grid, mode="bilinear", padding_mode="zeros", align_corners=True)


def resize_flow(flow: torch.Tensor, factor: int) -> torch.Tensor:
    """Upsample a low-resolution flow/disparity field by a constant factor."""
    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")
    new_size = (flow.shape[-2] * factor, flow.shape[-1] * factor)
    return factor * F.interpolate(flow, size=new_size, mode="bilinear", align_corners=True)
