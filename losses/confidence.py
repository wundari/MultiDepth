from __future__ import annotations

import torch
import torch.nn.functional as F

from losses.sequence import masked_mean


def focal_confidence_loss(
    confidence_logits: torch.Tensor,
    disp_before_refine: torch.Tensor,
    target_flow: torch.Tensor,
    *,
    threshold: float = 1.25,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Supervise stereo confidence from pre-refinement disparity error.

    This follows the released code closely: confidence is trained from whether the
    absolute pre-refinement disparity error stays below 5/4 pixels after resizing to
    the confidence-map resolution.
    """

    if disp_before_refine.shape != target_flow.shape:
        raise ValueError(
            f"disp_before_refine and target_flow must match, got {disp_before_refine.shape} vs {target_flow.shape}"
        )

    with torch.no_grad():
        error = (disp_before_refine - target_flow).abs()
        if error.shape[-2:] != confidence_logits.shape[-2:]:
            error = F.interpolate(
                error,
                size=confidence_logits.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        conf_gt = (error < threshold).float()
    bce = F.binary_cross_entropy_with_logits(
        confidence_logits, conf_gt, reduction="none"
    )
    p_t = torch.exp(-bce)
    return (alpha * ((1.0 - p_t) ** gamma) * bce).mean()


def sequence_l1_with_confidence_loss(
    predictions: list[torch.Tensor],
    target: torch.Tensor,
    valid: torch.Tensor,
    confidence_logits: torch.Tensor,
    *,
    gamma: float = 0.9,
    max_flow: float = 700.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if not predictions:
        raise ValueError(
            "sequence_l1_with_confidence_loss received an empty predictions list"
        )
    if valid.shape != target.shape:
        raise ValueError(
            f"valid shape must match target shape, got {valid.shape} vs {target.shape}"
        )

    magnitude = torch.sum(target**2, dim=1).sqrt()
    valid_mask = (valid >= 0.5) & (magnitude.unsqueeze(1) < max_flow)

    flow_loss = target.new_zeros(())
    num_predictions = len(predictions)
    for index, prediction in enumerate(predictions):
        if prediction.shape != target.shape:
            raise ValueError(
                f"prediction {index} shape mismatch: expected {target.shape}, got {prediction.shape}"
            )
        adjusted_gamma = gamma ** (15 / max(num_predictions - 1, 1))
        weight = adjusted_gamma ** (num_predictions - index - 1)
        flow_loss = flow_loss + weight * masked_mean(
            (prediction - target).abs(), valid_mask
        )

    disp_before_refine = predictions[-3] if len(predictions) >= 3 else predictions[-1]
    conf_loss = focal_confidence_loss(confidence_logits, disp_before_refine, target)
    total_loss = flow_loss + conf_loss

    final_epe = torch.sum((predictions[-1] - target) ** 2, dim=1).sqrt()
    final_epe = final_epe[valid_mask.squeeze(1)]
    metrics = {
        "loss": float(total_loss.detach().cpu()),
        "flow_loss": float(flow_loss.detach().cpu()),
        "confidence_loss": float(conf_loss.detach().cpu()),
        "epe": float(final_epe.mean().detach().cpu()) if final_epe.numel() > 0 else 0.0,
        "1px": (
            float((final_epe < 1).float().mean().detach().cpu())
            if final_epe.numel() > 0
            else 0.0
        ),
        "3px": (
            float((final_epe < 3).float().mean().detach().cpu())
            if final_epe.numel() > 0
            else 0.0
        ),
        "5px": (
            float((final_epe < 5).float().mean().detach().cpu())
            if final_epe.numel() > 0
            else 0.0
        ),
    }
    return total_loss, metrics
