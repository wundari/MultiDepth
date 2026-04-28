from __future__ import annotations

import torch

from models.stereo.corr import CorrelationPyramid1D, all_pairs_correlation_1d
from utils.geometry import make_coords_grid


def _one_hot_row_features(width: int) -> torch.Tensor:
    # Shape: [1, C=width, H=1, W=width]
    return torch.eye(width, dtype=torch.float32).unsqueeze(1).unsqueeze(0)


def test_all_pairs_correlation_identity_matrix() -> None:
    fmap = _one_hot_row_features(width=5)
    corr = all_pairs_correlation_1d(fmap, fmap)
    assert corr.shape == (1, 1, 5, 5)

    diagonal = torch.diagonal(corr[0, 0], dim1=0, dim2=1)
    off_diagonal_sum = corr[0, 0].sum() - diagonal.sum()
    assert torch.all(diagonal > 0)
    assert torch.isclose(off_diagonal_sum, torch.tensor(0.0))


def test_local_sampler_prefers_zero_offset_for_identical_features() -> None:
    fmap = _one_hot_row_features(width=5)
    corr_fn = CorrelationPyramid1D(fmap, fmap, num_levels=1, radius=1)
    coords = make_coords_grid(
        batch=1, height=1, width=5, device=fmap.device, dtype=fmap.dtype
    )
    sampled = corr_fn.sample(coords)
    assert sampled.shape == (1, 3, 1, 5)

    # For the center pixel x=2, the best match should be offset 0.
    center_pixel_scores = sampled[0, :, 0, 2]
    best_offset_index = int(center_pixel_scores.argmax().item())
    assert best_offset_index == 1  # offsets [-1, 0, +1]


def test_local_sampler_detects_negative_flow_for_shifted_match() -> None:
    left = _one_hot_row_features(width=5)
    right = torch.zeros_like(left)
    right[:, :, :, :-1] = left[:, :, :, 1:]  # match for left x appears at right x-1

    corr_fn = CorrelationPyramid1D(left, right, num_levels=1, radius=1)
    coords = make_coords_grid(
        batch=1, height=1, width=5, device=left.device, dtype=left.dtype
    )
    sampled = corr_fn.sample(coords)

    # For left pixel x=2, the best right candidate is x=1 -> flow_x = -1.
    center_pixel_scores = sampled[0, :, 0, 2]
    best_offset_index = int(center_pixel_scores.argmax().item())
    assert best_offset_index == 0  # offsets [-1, 0, +1]
