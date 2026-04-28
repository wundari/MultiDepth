from __future__ import annotations

import torch

from illusion_depth.models.mono.context_adapter import MonoContextAdapter, MonoContextAdapterConfig
from illusion_depth.models.mono.depthanything_wrapper import DepthAnythingBackboneOutput, MockDepthAnythingBackbone
from illusion_depth.models.stereo.raft_stereo_mono_core import RAFTStereoMonoCore, RAFTStereoMonoCoreConfig


def test_mock_depth_backbone_contract():
    model = MockDepthAnythingBackbone(feature_dim=128)
    image = torch.randint(0, 255, (2, 3, 128, 160), dtype=torch.float32)
    output = model(image)
    assert isinstance(output, DepthAnythingBackboneOutput)
    assert output.inverse_depth.shape == (2, 1, 16, 20)
    assert output.penultimate.shape == (2, 128, 16, 20)
    assert torch.all(output.inverse_depth >= 0)


def test_mono_context_adapter_shapes():
    adapter = MonoContextAdapter(MonoContextAdapterConfig(in_dim=128, hidden_dims=(128, 96, 64), context_dims=(128, 96, 64)))
    penultimate = torch.randn(2, 128, 16, 20)
    hidden_states, contexts = adapter(penultimate)
    assert hidden_states[0].shape == (2, 128, 16, 20)
    assert hidden_states[1].shape == (2, 96, 8, 10)
    assert hidden_states[2].shape == (2, 64, 4, 5)
    assert contexts[0].shape == (2, 128, 16, 20)
    assert contexts[1].shape == (2, 96, 8, 10)
    assert contexts[2].shape == (2, 64, 4, 5)


def test_stereo_mono_core_outputs_predictions_and_mono_depth():
    cfg = RAFTStereoMonoCoreConfig(iters=3)
    model = RAFTStereoMonoCore(cfg)
    left = torch.randint(0, 255, (1, 3, 128, 160), dtype=torch.float32)
    right = torch.randint(0, 255, (1, 3, 128, 160), dtype=torch.float32)
    predictions, flow_lowres, mono_depth = model(left, right, return_lowres=True, return_mono=True)
    assert len(predictions) == 3
    for pred in predictions:
        assert pred.shape == (1, 1, 128, 160)
    assert flow_lowres.shape == (1, 1, 16, 20)
    assert mono_depth.shape == (1, 1, 16, 20)


def test_stereo_mono_core_rejects_non_divisible_size():
    cfg = RAFTStereoMonoCoreConfig(iters=1)
    model = RAFTStereoMonoCore(cfg)
    left = torch.randint(0, 255, (1, 3, 130, 160), dtype=torch.float32)
    right = torch.randint(0, 255, (1, 3, 130, 160), dtype=torch.float32)
    try:
        model(left, right)
    except ValueError as exc:
        assert 'divisible' in str(exc)
    else:
        raise AssertionError('Expected ValueError for incompatible image size')
