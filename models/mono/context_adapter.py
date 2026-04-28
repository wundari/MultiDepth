# %%
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from config.cfg import MonoContextAdapterConfig
from models.stereo.encoder import ResidualBlock

from jaxtyping import Float


# %%
class MonoContextAdapter(nn.Module):
    """
    MonoContextAdapter is a multi-scale feature adapter —
    it takes a single feature map (likely the penultimate layer of an encoder)
    and produces two sets of feature maps at three different spatial scales,
    intended to inject information into a decoder or diffusion-style UNet.

    Input [B, C, H, W]
        │
        ▼
    self.pre         →  feat8  [B, 128, H/8,  W/8]   (two 3×3 convs)
        │
        ▼
    self.down16      →  feat16 [B, 128, H/16, W/16]  (stride-2 residual blocks)
        │
        ▼
    self.down32      →  feat32 [B, 128, H/32, W/32]  (stride-2 residual blocks)

    Each of the three feature maps is then passed through a projection head
    (out8, out16, out32) which maps the 128 channels to hidden_dim + context_dim channels.
    The output is then split into:

    hidden — passed through tanh, bounded to [-1, 1].
        Typically used to directly modulate hidden states in a decoder
        (e.g. added or concatenated).
    context — passed through ReLU, non-negative.
        Typically used as cross-attention context or conditioning signal.

    """

    def __init__(self, config: MonoContextAdapterConfig) -> None:
        super().__init__()

        self.config = config or MonoContextAdapterConfig()

        self.pre = nn.Sequential(
            ResidualBlock(config.in_dim, config.in_dim, stride=1),
            ResidualBlock(config.in_dim, config.in_dim, stride=1),
        )
        self.proj8 = nn.Conv2d(
            config.in_dim, config.hidden_dims[0] + config.context_dims[0], kernel_size=1
        )

        self.down16 = nn.Sequential(
            ResidualBlock(config.in_dim, config.in_dim, stride=2),
            ResidualBlock(config.in_dim, config.in_dim, stride=1),
        )
        self.proj16 = nn.Conv2d(
            config.in_dim, config.hidden_dims[1] + config.context_dims[1], kernel_size=1
        )

        self.down32 = nn.Sequential(
            ResidualBlock(config.in_dim, config.in_dim, stride=2),
            ResidualBlock(config.in_dim, config.in_dim, stride=1),
        )
        self.proj32 = nn.Conv2d(
            in_channels=config.in_dim,
            out_channels=config.hidden_dims[2] + config.context_dims[2],
            kernel_size=1,
        )

        self._init_weights()

    def _init_weights(self) -> None:

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, mono_penultimate: Float[Tensor, "B C H W"]) -> list[Tensor]:

        if mono_penultimate.ndim != 4:
            raise ValueError(
                f"Expected [B,C,H,W] mono features, got {mono_penultimate.shape}"
            )
        feat8 = self.pre(mono_penultimate)
        feat16 = self.down16(feat8)
        feat32 = self.down32(feat16)
        return [self.proj8(feat8), self.proj16(feat16), self.proj32(feat32)]
