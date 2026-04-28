# %%
from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Beta
from torch import Tensor

from models.fusion.affine_align import AffineAlignmentHead
from models.vlm.flux_confidence import QwenFluxConfidenceBranch
from config.cfg import FluxConfidenceConfig
from config.cfg import AffineAlignmentConfig
from config.cfg import StereoMonoVLMRefinementConfig, StereoMonoVLMRefinementOutput
from jaxtyping import Float


# %%
class StereoMonoVLMRefinement(nn.Module):
    """
    What This Class Does
    This is the top-level fusion and refinement module of a stereo+monocular depth
    system with VLM guidance. It combines three distinct sub-systems:

    1. VLM-guided Confidence Estimation (confidence_head)
    QwenFluxConfidenceBranch is a multimodal branch that fuses stereo cost
    volume features, disparity, monocular depth, optional hidden encoder features,
    and an optional text prompt to produce a per-pixel confidence map ∈ (0, 1).
    High confidence means the stereo disparity is trustworthy at that pixel;
    low confidence means the monocular estimate should dominate.
    The VLM (Qwen) allows semantic priors (e.g., "this is a textureless wall
    where stereo fails") to influence confidence via the prompt.

    2. Affine Alignment (affine_alignment)
    As covered previously, this warps monocular depth into the disparity coordinate
    space with a learned per-pixel (or global) affine transform depth * a + b,
    producing registered_depth.

    3. Confidence-Weighted Fusion + Upsampling Mask
    fused_disp = disp × confidence + (1 - confidence) × registered_depth

    This is a soft selection between stereo and monocular-registered depth —
    wherever stereo is confident, use stereo; wherever it isn't, fall back to
    the affine-corrected monocular estimate. The result is then concatenated with
    the hidden features and passed through mask_head, which predicts a RAFT-style
    convex upsampling kernel — a 9 × upsample_factor**2 channel map used downstream
    to upsample fused_disp to full resolution by taking a learned weighted
    combination of the 3×3 neighborhood at each coarse pixel.

    """

    def __init__(self, config: StereoMonoVLMRefinementConfig) -> None:

        super().__init__()
        self.config = config

        self.confidence_head = QwenFluxConfidenceBranch(config.flux_confidence)
        self.affine_alignment = AffineAlignmentHead(
            AffineAlignmentConfig(hidden_dim=32, global_pool=config.global_pool_affine)
        )
        self.mask_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_dim + 1,
                out_channels=256,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=256,
                out_channels=9 * (config.upsample_factor**2),
                kernel_size=1,
            ),
        )

        self._init_weights()

    def _init_weights(self) -> None:

        for module in self.mask_head.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        left_rgb: Float[Tensor, "B C H W"],
        disp: Float[Tensor, "B 1 H W"],
        depth: Float[Tensor, "B 1 H W"],
        hidden: Float[Tensor, "B C H W"],
        cost_volume: Float[Tensor, "B Cc H W"],
        beta_distribution: Beta | None = None,
        prompt: str | None = None,
    ) -> StereoMonoVLMRefinementOutput:

        # validate spatial dimensions, [H, W]
        spatial_dims = disp.shape[-2:]  # [H, W]

        if disp.shape != depth.shape:
            raise ValueError(
                f"disp/depth mismatch in refinement: {disp.shape} vs. {depth.shape}"
            )

        if hidden.shape[-2:] != spatial_dims or cost_volume.shape[-2:] != spatial_dims:
            raise ValueError(
                "hidden, cost_volume, and disparity must share the same spatial size"
            )

        # if left_rgb.shape[-2:] != spatial_dims:
        #     raise ValueError(
        #         f"left_rgb spatial size {left_rgb.shape[-2:]} must match "
        #         f"disparity spatial size {spatial_dims}"
        #     )

        confidence_logits, confidence, vlm_aux = self.confidence_head(
            left_rgb,
            cost_volume,
            disp,
            depth,
            hidden=hidden if self.config.use_hidden_features else None,
            beta_distribution=(
                beta_distribution if self.config.use_beta_statistics else None
            ),
            prompt=prompt,
        )

        # validate confidence spatial alignment before blending
        if confidence.shape[-2:] != spatial_dims:
            raise ValueError(
                f"confidence map spatial size {confidence.shape[-2:]} does not match "
                f"disparity spatial size {spatial_dims}. confidence_head output must be "
                f"upsampled to full resolution before blending."
            )

        affine = self.affine_alignment(disp, depth)
        fused_disp = disp * confidence + (1.0 - confidence) * affine.registered_depth
        up_mask = self.mask_head(torch.cat([hidden, fused_disp], dim=1))

        return StereoMonoVLMRefinementOutput(
            fused_disp=fused_disp,
            up_mask=up_mask,
            depth_registered=affine.registered_depth,  # monocular depth
            confidence_logits=confidence_logits,
            confidence=confidence,
            a=affine.a,
            b=affine.b,
            vlm_aux=vlm_aux,
        )
