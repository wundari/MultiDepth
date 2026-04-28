# %%
from __future__ import annotations

import torch

from models.stereo.encoder import MultiScaleContextConfig, MultiScaleContextEncoder
from models.stereo.raft_stereo_core import RAFTStereoCore, RAFTStereoCoreConfig
from models.stereo.update_block import (
    MultiScaleStereoUpdateBlock,
    MultiScaleUpdateConfig,
)

# %%


def test_multiscale_context_encoder_shapes() -> None:
    encoder = MultiScaleContextEncoder(
        MultiScaleContextConfig(
            hidden_dims=(32, 24, 16), context_dims=(32, 24, 16), norm="instance"
        )
    )
    image = torch.randn(2, 3, 64, 96)
    out8, out16, out32 = encoder(image)
    assert out8.shape == (2, 64, 8, 12)
    assert out16.shape == (2, 48, 4, 6)
    assert out32.shape == (2, 32, 2, 3)


def test_multiscale_update_block_shapes() -> None:
    cfg = MultiScaleUpdateConfig(
        hidden_dims=(32, 24, 16),
        context_dims=(32, 24, 16),
        corr_levels=2,
        corr_radius=2,
    )
    block = MultiScaleStereoUpdateBlock(cfg)
    h8 = torch.randn(1, 32, 8, 12)
    h16 = torch.randn(1, 24, 4, 6)
    h32 = torch.randn(1, 16, 2, 3)
    c8 = torch.randn(1, 32, 8, 12)
    c16 = torch.randn(1, 24, 4, 6)
    c32 = torch.randn(1, 16, 2, 3)
    corr = torch.randn(1, cfg.corr_channels, 8, 12)
    flow = torch.randn(1, 2, 8, 12)
    prepared = block.prepare_contexts((c8, c16, c32))
    hidden_states, up_mask, delta_flow = block((h8, h16, h32), prepared, corr, flow)
    out8, out16, out32 = hidden_states
    assert out8.shape == h8.shape
    assert out16.shape == h16.shape
    assert out32.shape == h32.shape
    assert up_mask.shape == (1, 9 * (cfg.upsample_factor**2), 8, 12)
    assert delta_flow.shape == (1, 2, 8, 12)
    assert torch.allclose(delta_flow[:, 1], torch.zeros_like(delta_flow[:, 1]))


def test_raft_stereo_core_forward_shapes() -> None:
    cfg = RAFTStereoCoreConfig(
        feature_dim=64,
        hidden_dims=(32, 24, 16),
        context_dims=(32, 24, 16),
        corr_levels=2,
        corr_radius=2,
        iters=4,
        update_16_every=2,
        update_32_every=3,
    )
    model = RAFTStereoCore(cfg)
    left = torch.randint(0, 256, (2, 3, 64, 96)).float()
    right = torch.randint(0, 256, (2, 3, 64, 96)).float()
    predictions, lowres = model(left, right, return_lowres=True)
    assert len(predictions) == 4
    for prediction in predictions:
        assert prediction.shape == (2, 1, 64, 96)
        assert torch.isfinite(prediction).all()
    assert lowres.shape == (2, 1, 8, 12)


def test_raft_stereo_core_rejects_non_multiple_of_32() -> None:
    model = RAFTStereoCore()
    left = torch.randint(0, 256, (1, 3, 64, 72)).float()
    right = torch.randint(0, 256, (1, 3, 64, 72)).float()
    try:
        model(left, right)
    except ValueError as exc:
        assert "required factor=32" in str(exc) or "factor=32" in str(exc)
    else:
        raise AssertionError("Expected non-multiple-of-32 input size to raise")
