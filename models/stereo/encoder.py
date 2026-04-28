# %%
from __future__ import annotations
from config.cfg import EncoderConfig, MultiScaleContextConfig

import torch
from torch import nn, Tensor

from jaxtyping import Float


# %%
class ResidualBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:

        residual = self.skip(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + residual)


class BasicEncoder(nn.Module):
    """
    BasicEncoder is the shared feature extractor used for
    both left and right images.
    It's a standard ResNet-style encoder that downsamples
    the input by 8× total (stride-2 stem + two stride-2 stages),
    then projects to out_dim channels.
    The final features feed into the cost volume and
    context projectors.

    input (B, 3, H, W)
        └──► stem  7×7 s2 ──► (B, stem_dim, H/2,  W/2)
        └──► stage1 s1+s1 ──► (B, d0,       H/2,  W/2)
        └──► stage2 s2+s1 ──► (B, d1,       H/4,  W/4)
        └──► stage3 s2+s1 ──► (B, d2,       H/8,  W/8)
        └──► proj   1×1   ──► (B, out_dim,  H/8,  W/8)

    The clever part is the sequence batching trick in
    forward — if you pass a tuple/list of images
    (e.g. left and right), it concatenates them along
    the batch dimension, runs the full encoder once,
    then splits the result back.
    This means one forward pass extracts features for
    both images simultaneously, using the GPU more
    efficiently than two separate calls.

    """

    def __init__(self, out_dim: int, config: EncoderConfig) -> None:

        super().__init__()
        self.config = config

        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=config.stem_dim,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(config.stem_dim),
            nn.ReLU(inplace=True),
        )
        d0, d1, d2 = config.stage_dims

        self.stage1 = nn.Sequential(
            ResidualBlock(in_channels=config.stem_dim, out_channels=d0, stride=1),
            ResidualBlock(in_channels=d0, out_channels=d0, stride=1),
        )
        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=d0, out_channels=d1, stride=2),
            ResidualBlock(in_channels=d1, out_channels=d1, stride=1),
        )
        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=d1, out_channels=d2, stride=2),
            ResidualBlock(in_channels=d2, out_channels=d2, stride=1),
        )

        self.proj = nn.Conv2d(in_channels=d2, out_channels=out_dim, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self, x: Float[Tensor, "B C H W"] | list | tuple
    ) -> Float[Tensor, "B C H W"] | tuple:

        is_sequence = isinstance(x, (list, tuple))
        if is_sequence:
            batch = x[0].shape[0]
            x = torch.cat(list(x), dim=0)
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.proj(x)
        if is_sequence:
            return x.split(batch, dim=0)
        return x


class MultiScaleContextEncoder(nn.Module):
    """
    MultiScaleContextEncoder is the context encoder for the
    multi-scale update block.
    Unlike BasicEncoder which produces a single feature map
    at 1/8 resolution, this encoder produces three feature
    maps at three scales (1/8, 1/16, 1/32) simultaneously,
    each of which gets split into a hidden state initialiser
    and a context gate projection input.

    input (B, 3, H, W)
        └──► stem  s2 ──► (B, stem_dim, H/2,  W/2)
        └──► stage1 s1 ──► (B, s0,      H/2,  W/2)
        └──► stage2 s2 ──► (B, s1,      H/4,  W/4)
        └──► stage3 s2 ──► (B, s2,      H/8,  W/8)  ──► proj8  ──► (B, h0+c0, H/8,  W/8)
        └──► stage4 s2 ──► (B, s3,      H/16, W/16) ──► proj16 ──► (B, h1+c1, H/16, W/16)
        └──► stage5 s2 ──► (B, s4,      H/32, W/32) ──► proj32 ──► (B, h2+c2, H/32, W/32)

    """

    def __init__(self, config: MultiScaleContextConfig) -> None:

        super().__init__()

        cfg = config
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=cfg.stem_dim,
                kernel_size=7,
                stride=2,
                padding=3,
            ),
            nn.BatchNorm2d(cfg.stem_dim),
            nn.ReLU(inplace=True),
        )

        s0, s1, s2, s3, s4 = cfg.stage_dims  # (64, 96, 128, 160, 192)

        self.stage1 = nn.Sequential(
            ResidualBlock(in_channels=cfg.stem_dim, out_channels=s0, stride=1),
            ResidualBlock(in_channels=s0, out_channels=s0, stride=1),
        )

        self.stage2 = nn.Sequential(
            ResidualBlock(in_channels=s0, out_channels=s1, stride=2),
            ResidualBlock(in_channels=s1, out_channels=s1, stride=1),
        )

        self.stage3 = nn.Sequential(
            ResidualBlock(in_channels=s1, out_channels=s2, stride=2),
            ResidualBlock(in_channels=s2, out_channels=s2, stride=1),
        )

        self.stage4 = nn.Sequential(
            ResidualBlock(in_channels=s2, out_channels=s3, stride=2),
            ResidualBlock(in_channels=s3, out_channels=s3, stride=1),
        )

        self.stage5 = nn.Sequential(
            ResidualBlock(in_channels=s3, out_channels=s4, stride=2),
            ResidualBlock(in_channels=s4, out_channels=s4, stride=1),
        )

        h0, h1, h2 = cfg.hidden_dims  # (128, 96, 64)
        c0, c1, c2 = cfg.context_dims  # (128, 96, 64)
        self.proj8 = nn.Conv2d(in_channels=s2, out_channels=h0 + c0, kernel_size=1)
        self.proj16 = nn.Conv2d(in_channels=s3, out_channels=h1 + c1, kernel_size=1)
        self.proj32 = nn.Conv2d(in_channels=s4, out_channels=h2 + c2, kernel_size=1)

        # initialize weights
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

    def forward(
        self, x: Float[Tensor, "B C H W"]
    ) -> tuple[
        Float[Tensor, "B C H W"], Float[Tensor, "B C H W"], Float[Tensor, "B C H W"]
    ]:

        out = self.stem(x)
        out = self.stage1(out)
        out = self.stage2(out)
        feat8 = self.stage3(out)
        feat16 = self.stage4(feat8)
        feat32 = self.stage5(feat16)

        return (self.proj8(feat8), self.proj16(feat16), self.proj32(feat32))
