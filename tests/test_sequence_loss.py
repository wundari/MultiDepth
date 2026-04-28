from __future__ import annotations

import torch

from losses.sequence import sequence_l1_loss


def test_sequence_l1_loss_weights_late_predictions_more() -> None:
    target = torch.zeros(1, 1, 2, 2)
    valid = torch.ones_like(target, dtype=torch.bool)
    predictions = [
        torch.full_like(target, 2.0),
        torch.full_like(target, 1.0),
    ]
    loss, metrics = sequence_l1_loss(predictions, target, valid, gamma=0.5)

    # 0.5 * mean(|2|) + 1.0 * mean(|1|) = 2.0
    assert torch.isclose(loss, torch.tensor(2.0))
    assert metrics["final_l1"] == 1.0
    assert metrics["valid_fraction"] == 1.0


def test_sequence_l1_loss_respects_valid_mask() -> None:
    target = torch.zeros(1, 1, 2, 2)
    valid = torch.tensor([[[[1, 0], [1, 0]]]], dtype=torch.bool)
    prediction = torch.tensor([[[[2.0, 100.0], [4.0, 100.0]]]])

    loss, metrics = sequence_l1_loss([prediction], target, valid)
    assert torch.isclose(loss, torch.tensor(3.0))
    assert metrics["final_l1"] == 3.0
