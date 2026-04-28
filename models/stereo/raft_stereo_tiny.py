from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.stereo.corr import CorrelationPyramid1D
from models.stereo.encoder import ContextEncoder, EncoderConfig, BasicEncoder
from models.stereo.update_block import StereoUpdateBlock, UpdateBlockConfig
from models.stereo.upsample import upsample_flow
from utils.geometry import make_coords_grid, normalize_uint8_image


@dataclass(frozen=True)
class RAFTStereoTinyConfig:
    feature_dim: int = 128
    hidden_dim: int = 128
    context_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 4
    iters: int = 6
    downsample_factor: int = 8
    norm: str = "instance"
    use_convex_upsampling: bool = True


class RAFTStereoTiny(nn.Module):
    """A minimal RAFT-style stereo matcher.

    What this milestone keeps from the released model
    -----------------------------------------------
    - left/right image normalization to [-1, 1]
    - 1D row-wise all-pairs correlation
    - coordinate-grid view of stereo matching
    - iterative disparity refinement
    - epipolar constraint (vertical motion forced to zero)
    - convex upsampling to full image resolution

    What this milestone intentionally omits
    ---------------------------------------
    - multi-scale GRUs
    - shared-backbone feature/context path
    - alternate CUDA correlation implementations
    - monocular priors and fusion
    - confidence prediction
    """

    def __init__(self, config: RAFTStereoTinyConfig | None = None) -> None:
        super().__init__()
        cfg = config or RAFTStereoTinyConfig()
        self.config = cfg

        encoder_cfg = EncoderConfig(norm=cfg.norm)
        self.feature_encoder = BasicEncoder(cfg.feature_dim, config=encoder_cfg)
        self.context_encoder = ContextEncoder(
            cfg.hidden_dim + cfg.context_dim, config=encoder_cfg
        )

        update_cfg = UpdateBlockConfig(
            hidden_dim=cfg.hidden_dim,
            context_dim=cfg.context_dim,
            corr_levels=cfg.corr_levels,
            corr_radius=cfg.corr_radius,
            motion_dim=128,
            upsample_factor=cfg.downsample_factor,
        )
        self.update_block = StereoUpdateBlock(update_cfg)

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        iters: int | None = None,
        flow_init: torch.Tensor | None = None,
        return_lowres: bool = False,
    ) -> list[torch.Tensor] | tuple[list[torch.Tensor], torch.Tensor]:
        self._validate_inputs(left, right)

        num_iters = self.config.iters if iters is None else iters
        if num_iters <= 0:
            raise ValueError(f"iters must be positive, got {num_iters}")

        left_norm = normalize_uint8_image(left)
        right_norm = normalize_uint8_image(right)

        fmap_left, fmap_right = self._extract_feature_pair(left_norm, right_norm)
        context = self.context_encoder(left_norm)
        hidden, context_features = torch.split(
            context,
            [self.config.hidden_dim, self.config.context_dim],
            dim=1,
        )
        hidden = torch.tanh(hidden)
        context_features = F.relu(context_features)

        batch, _, h8, w8 = fmap_left.shape
        coords0 = make_coords_grid(
            batch, h8, w8, device=fmap_left.device, dtype=fmap_left.dtype
        )
        coords1 = coords0.clone()
        if flow_init is not None:
            if flow_init.shape != coords1.shape:
                raise ValueError(
                    f"flow_init must match low-res coords shape {coords1.shape}, got {flow_init.shape}"
                )
            coords1 = coords1 + flow_init

        corr_pyramid = CorrelationPyramid1D(
            fmap_left,
            fmap_right,
            num_levels=self.config.corr_levels,
            radius=self.config.corr_radius,
        )

        predictions: list[torch.Tensor] = []
        for _ in range(num_iters):
            coords1 = coords1.detach()
            corr = corr_pyramid.sample(coords1)
            flow = coords1 - coords0
            hidden, up_mask, delta_flow = self.update_block(
                hidden, context_features, corr, flow
            )
            coords1 = coords1 + delta_flow
            flow_lowres = coords1 - coords0
            fullres_flow = upsample_flow(
                flow_lowres,
                factor=self.config.downsample_factor,
                mask_logits=up_mask if self.config.use_convex_upsampling else None,
            )
            predictions.append(fullres_flow[:, :1])

        if return_lowres:
            return predictions, (coords1 - coords0)[:, :1]
        return predictions

    def _extract_feature_pair(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pair = torch.cat([left, right], dim=0)
        encoded = self.feature_encoder(pair)
        fmap_left, fmap_right = torch.chunk(encoded, chunks=2, dim=0)

        self._validate_downsample_ratio(left, fmap_left)
        return fmap_left, fmap_right

    def _validate_inputs(self, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim != 4 or right.ndim != 4:
            raise ValueError(
                f"Expected [B, 3, H, W] inputs, got {left.shape} and {right.shape}"
            )
        if left.shape != right.shape:
            raise ValueError(
                f"Left/right shape mismatch: {left.shape} vs {right.shape}"
            )
        if left.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {left.shape}")
        factor = self.config.downsample_factor
        if left.shape[-2] % factor != 0 or left.shape[-1] % factor != 0:
            raise ValueError(
                "Milestone 2 expects image sizes divisible by the downsample factor. "
                f"Got HxW={left.shape[-2:]} with factor={factor}."
            )

    def _validate_downsample_ratio(
        self, fullres: torch.Tensor, lowres: torch.Tensor
    ) -> None:
        factor_h = fullres.shape[-2] // lowres.shape[-2]
        factor_w = fullres.shape[-1] // lowres.shape[-1]
        if (
            factor_h != self.config.downsample_factor
            or factor_w != self.config.downsample_factor
        ):
            raise RuntimeError(
                "Encoder downsample factor mismatch: "
                f"expected {self.config.downsample_factor}, got ({factor_h}, {factor_w})"
            )
