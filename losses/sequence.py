from __future__ import annotations

import torch


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.float()
    denom = mask_f.sum().clamp(min=1.0)
    return (values * mask_f).sum() / denom


def sequence_l1_loss(
    predictions: list[torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    *,
    gamma: float = 0.9,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Weighted L1 loss over a sequence of disparity predictions.

    Parameters
    ----------
    predictions:
        List of tensors, each of shape [B, 1, H, W].
    target:
        Tensor[B, 1, H, W]
    valid:
        Bool or float tensor[B, 1, H, W]
    gamma:
        Later predictions get larger weight.
    """
    if not predictions:
        raise ValueError("sequence_l1_loss received an empty predictions list")
    if target.ndim != 4 or target.shape[1] != 1:
        raise ValueError(f"target must have shape [B, 1, H, W], got {target.shape}")
    if valid.shape != target.shape:
        raise ValueError(
            f"valid shape must match target shape, got {valid.shape} vs {target.shape}"
        )

    loss = target.new_zeros(())
    num_predictions = len(predictions)
    for index, prediction in enumerate(predictions):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction {index} shape mismatch: expected {target.shape}, got {prediction.shape}"
            )
        weight = gamma ** (num_predictions - index - 1)
        loss = loss + weight * masked_mean((prediction - target).abs(), valid)

    final_abs_error = (predictions[-1] - target).abs()
    metrics = {
        "loss": float(loss.detach().cpu()),
        "final_l1": float(masked_mean(final_abs_error, valid).detach().cpu()),
        "valid_fraction": float(valid.float().mean().detach().cpu()),
    }
    return loss, metrics
