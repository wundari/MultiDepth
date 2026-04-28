from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.distributions import Beta

from models.stereo.encoder import ResidualBlock


@dataclass(frozen=True)
class BetaModulatorConfig:
    lbp_dim: int
    hidden_dim: int | None = None
    norm: str = "batch"


@dataclass(frozen=True)
class BetaModulationOutput:
    modulation: torch.Tensor
    alpha: torch.Tensor
    beta: torch.Tensor
    distribution: Beta


class BetaModulator(nn.Module):
    """Predict a Beta-distributed modulation from stereo/mono LBP features."""

    def __init__(self, config: BetaModulatorConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = (
            config.hidden_dim if config.hidden_dim is not None else config.lbp_dim
        )
        self.hidden_dim = hidden_dim
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                config.lbp_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, bias=True
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, bias=True
            ),
        )
        down_dim = 64 if hidden_dim * 2 < 64 else 128
        self.down = nn.Sequential(
            ResidualBlock(hidden_dim * 2, down_dim, stride=2),
            ResidualBlock(down_dim, 128, stride=1),
        )
        self.up = nn.ConvTranspose2d(128, hidden_dim * 2, kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim * 4, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.Softplus(),
            nn.Conv2d(hidden_dim, 2, kernel_size=1, padding=0, bias=False),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
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
        self,
        lbp_disp: torch.Tensor,
        lbp_depth: torch.Tensor,
        *,
        return_distribution: bool = False,
    ) -> torch.Tensor | BetaModulationOutput:
        if lbp_disp.shape != lbp_depth.shape:
            raise ValueError(
                f"LBP stereo/mono shapes must match, got {lbp_disp.shape} vs {lbp_depth.shape}"
            )
        x1 = self.conv1(torch.cat([lbp_disp, lbp_depth], dim=1))
        x2 = self.up(self.down(x1))
        if x2.shape[-2:] != x1.shape[-2:]:
            x2 = torch.nn.functional.interpolate(
                x2, size=x1.shape[-2:], mode="bilinear", align_corners=False
            )
        beta_paras = self.conv2(torch.cat([x1, x2], dim=1)) + 1.0
        alpha, beta = torch.split(beta_paras, 1, dim=1)
        distribution = Beta(alpha, beta)
        modulation = distribution.rsample() if self.training else distribution.mean
        if not return_distribution:
            return modulation
        return BetaModulationOutput(
            modulation=modulation, alpha=alpha, beta=beta, distribution=distribution
        )
