# %%
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from config.cfg import AffineAlignmentConfig, AffineAlignmentOutput
from jaxtyping import Float

# %%


class AffineAlignmentHead(nn.Module):
    """
    Estimate affine registration parameters to align monocular depth with
    stereo disparity.

    What This Class Does
    This module learns to align monocular depth predictions with stereo
    disparity maps using a learnable per-pixel (or global) affine transformation.

    The Core Problem
    Monocular depth estimators and stereo disparity estimators produce depth
    maps that are consistent in structure but differ in scale and shift —
    they operate in different units and may have different biases.
    This head bridges that gap by learning the corrective transform.

    Step-by-Step Logic
    1. Input stacking (forward)
    [disp (B,1,H,W), depth (B,1,H,W)] → cat → (B,2,H,W)
    Disparity and depth are concatenated along the channel dimension so the
    network sees both simultaneously and can reason about their relationship.

    2. Parameter prediction (self.net)
    A lightweight 3-layer conv network maps the 2-channel input to a 2-channel
    output — one channel for scale a and one for shift b.
    By default (global_pool=False) this produces spatially varying per-pixel
    parameters.

    3. Optional global pooling (self.pool)
    If config.global_pool=True, the spatial maps are collapsed to
    scalars (B,2,1,1) via average pooling, then broadcast back to full resolution.
    This enforces a single global scale and shift for the whole image —
    a stronger, simpler prior.

    4. Affine registration
    registered_depth = depth × a + b
    This is a standard affine transform. The network learns a (scale correction)
    and b (offset correction) to warp the monocular depth into the coordinate
    space of the stereo disparity.

    Weight Initialization
    Kaiming normal init is used for all conv layers
    (appropriate for ReLU activations) with zero-init biases —
    this is why _init_weights matters and the bug of not calling it is significant.

    """

    def __init__(self, config: AffineAlignmentConfig) -> None:

        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=2, out_channels=config.hidden_dim, kernel_size=3, padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=config.hidden_dim,
                out_channels=config.hidden_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=config.hidden_dim, out_channels=2, kernel_size=1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1)) if config.global_pool else None
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, disp: Float[Tensor, "B 1 H W"], depth: Float[Tensor, "B 1 H W"]
    ) -> AffineAlignmentOutput:

        if disp.shape != depth.shape:
            raise ValueError(
                f"disp/depth must match for affine alignment, got {disp.shape} vs. {depth.shape}"
            )

        params = self.net(torch.cat([disp, depth], dim=1))
        if self.pool is not None:
            params = self.pool(params)
            params = params.expand(-1, -1, disp.shape[-2], disp.shape[-1])
        a, b = torch.split(params, 1, dim=1)
        registered = depth * a + b

        return AffineAlignmentOutput(a=a, b=b, registered_depth=registered)
