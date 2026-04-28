# %%
from __future__ import annotations

import torch
from torch.distributions import Beta

from models.fusion.refine_vlm import (
    StereoMonoVLMRefinement,
    StereoMonoVLMRefinementConfig,
)
from models.stereo.raft_stereo_mono_beta_vlm_flux_core import (
    RAFTStereoMonoBetaVLMFluxConfig,
    RAFTStereoMonoBetaVLMFluxCore,
)
from models.vlm.flux_confidence import FluxConfidenceConfig, QwenFluxConfidenceBranch
from models.vlm.prompts import ReflectiveObjectPromptBuilder
from models.vlm.qwen_encoder import QwenVisionBackbone, QwenVisionConfig


# %%
def test_prompt_mentions_reflective_objects() -> None:
    prompt = ReflectiveObjectPromptBuilder().build().full_text.lower()
    assert "reflective" in prompt
    assert "glass" in prompt
    assert "mirror" in prompt


def test_qwen_backbone_shapes() -> None:
    model = QwenVisionBackbone(
        QwenVisionConfig(backend="mock", qwen_dim=64, flux_dim=64)
    )
    images = torch.rand(2, 3, 64, 96)
    out = model(images)
    assert out.tokens.shape[0] == 2
    assert out.tokens.shape[-1] == 64
    assert out.spatial.shape == (2, 64, 8, 12)
    assert out.prompt_embedding.shape == (2, 64)


def test_flux_confidence_branch_shapes() -> None:
    cfg = FluxConfidenceConfig(
        qwen=QwenVisionConfig(backend="mock", qwen_dim=64, flux_dim=64),
        hidden_dim=64,
        latent_dim=64,
        num_blocks=2,
        num_heads=4,
        use_hidden_features=True,
        use_beta_statistics=True,
    )
    model = QwenFluxConfidenceBranch(cfg)
    left = torch.rand(2, 3, 64, 96)
    cost = torch.rand(2, cfg.corr_channels, 8, 12)
    disp = torch.rand(2, 1, 8, 12)
    depth = torch.rand(2, 1, 8, 12)
    hidden = torch.rand(2, 64, 8, 12)
    beta = Beta(torch.ones_like(disp) * 2.0, torch.ones_like(disp) * 3.0)
    logits, conf, aux = model(
        left, cost, disp, depth, hidden=hidden, beta_distribution=beta
    )
    assert logits.shape == (2, 1, 8, 12)
    assert conf.shape == (2, 1, 8, 12)
    assert aux["qwen_spatial"].shape[-2:] == (8, 12)


def test_vlm_refinement_shapes() -> None:
    cfg = StereoMonoVLMRefinementConfig(
        hidden_dim=64,
        upsample_factor=8,
        flux_confidence=FluxConfidenceConfig(
            qwen=QwenVisionConfig(backend="mock", qwen_dim=64, flux_dim=64),
            hidden_dim=64,
            latent_dim=64,
            num_blocks=2,
            num_heads=4,
            use_hidden_features=True,
            use_beta_statistics=True,
        ),
    )
    model = StereoMonoVLMRefinement(cfg)
    left = torch.rand(1, 3, 64, 96)
    disp = torch.rand(1, 1, 8, 12)
    depth = torch.rand(1, 1, 8, 12)
    hidden = torch.rand(1, 64, 8, 12)
    cost = torch.rand(1, cfg.flux_confidence.corr_channels, 8, 12)
    beta = Beta(torch.ones_like(disp) * 2.0, torch.ones_like(disp) * 5.0)
    out = model(left, disp, depth, hidden, cost, beta_distribution=beta)
    assert out.confidence_logits.shape == (1, 1, 8, 12)
    assert out.fused_disp.shape == (1, 1, 8, 12)
    assert out.up_mask.shape == (1, 9 * 64, 8, 12)


def test_stereo_mono_beta_vlm_flux_core_end_to_end() -> None:
    cfg = RAFTStereoMonoBetaVLMFluxConfig(
        feature_dim=64,
        hidden_dims=(64, 48, 32),
        context_dims=(64, 48, 32),
        corr_levels=2,
        corr_radius=2,
        iters=2,
        mono_penultimate_dim=64,
        vlm_refinement=StereoMonoVLMRefinementConfig(
            corr_levels=2,
            corr_radius=2,
            hidden_dim=64,
            upsample_factor=8,
            flux_confidence=FluxConfidenceConfig(
                qwen=QwenVisionConfig(backend="mock", qwen_dim=64, flux_dim=64),
                corr_levels=2,
                corr_radius=2,
                hidden_dim=64,
                latent_dim=64,
                num_blocks=1,
                num_heads=4,
                use_hidden_features=True,
                use_beta_statistics=True,
            ),
        ),
    )
    model = RAFTStereoMonoBetaVLMFluxCore(cfg)
    left = torch.randint(0, 255, (1, 3, 64, 96), dtype=torch.uint8).float()
    right = torch.randint(0, 255, (1, 3, 64, 96), dtype=torch.uint8).float()
    out = model(left, right, prompt="Highlight reflective and transparent regions.")
    assert len(out["disp_predictions"]) == cfg.iters + 2
    assert out["conf"].shape[-2:] == (8, 12)
    assert out["disp_predictions"][-1].shape[-2:] == (64, 96)
    assert "qwen_tokens_pooled" in out["vlm_aux"]
