from __future__ import annotations

import torch

from illusion_depth.models.stereo.raft_stereo_tiny import RAFTStereoTiny, RAFTStereoTinyConfig


def test_stereo_tiny_forward_shapes() -> None:
    config = RAFTStereoTinyConfig(iters=3)
    model = RAFTStereoTiny(config)

    left = torch.randint(0, 256, (2, 3, 64, 80)).float()
    right = torch.randint(0, 256, (2, 3, 64, 80)).float()

    predictions, lowres = model(left, right, return_lowres=True)
    assert len(predictions) == 3
    for prediction in predictions:
        assert prediction.shape == (2, 1, 64, 80)
        assert torch.isfinite(prediction).all()
    assert lowres.shape == (2, 1, 8, 10)


def test_stereo_tiny_rejects_non_divisible_sizes() -> None:
    model = RAFTStereoTiny()
    left = torch.randint(0, 256, (1, 3, 65, 80)).float()
    right = torch.randint(0, 256, (1, 3, 65, 80)).float()
    try:
        model(left, right)
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("Expected non-divisible input size to raise")
