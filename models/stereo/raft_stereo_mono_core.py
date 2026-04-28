from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.mono.context_adapter import MonoContextAdapter, MonoContextAdapterConfig
from models.mono.depthanything_wrapper import (
    DepthAnythingBackboneOutput,
    MockDepthAnythingBackbone,
)
from models.stereo.corr import CorrelationPyramid1D
from models.stereo.encoder import BasicEncoder, EncoderConfig
from models.stereo.update_block import (
    MultiScaleStereoUpdateBlock,
    MultiScaleUpdateConfig,
)
from models.stereo.upsample import upsample_flow
from utils.geometry import make_coords_grid, normalize_uint8_image


@dataclass(frozen=True)
class RAFTStereoMonoCoreConfig:
    feature_dim: int = 128
    hidden_dims: tuple[int, int, int] = (128, 96, 64)
    context_dims: tuple[int, int, int] = (128, 96, 64)
    corr_levels: int = 4
    corr_radius: int = 4
    iters: int = 8
    downsample_factor: int = 8
    min_context_factor: int = 32
    update_16_every: int = 2
    update_32_every: int = 4
    norm: str = "instance"
    use_convex_upsampling: bool = True
    mono_penultimate_dim: int = 128


class RAFTStereoMonoCore(nn.Module):
    """Stereo core guided by a frozen monocular depth prior.

    This milestone mirrors the released RAFTStereoDepthAny idea:
    - stereo cost volume still comes from left/right feature encoders
    - hidden/context states come from a frozen DepthAnything-like branch
    - the monocular branch predicts affine-invariant inverse depth and provides
      penultimate features that are adapted into recurrent stereo context
    """

    def __init__(
        self,
        config: RAFTStereoMonoCoreConfig | None = None,
        *,
        mono_backbone: nn.Module | None = None,
    ) -> None:
        super().__init__()
        cfg = config or RAFTStereoMonoCoreConfig()
        self.config = cfg
        self.feature_encoder = BasicEncoder(
            cfg.feature_dim, config=EncoderConfig(norm=cfg.norm)
        )
        self.mono_backbone = (
            mono_backbone
            if mono_backbone is not None
            else MockDepthAnythingBackbone(feature_dim=cfg.mono_penultimate_dim)
        )
        self.mono_adapter = MonoContextAdapter(
            MonoContextAdapterConfig(
                in_dim=cfg.mono_penultimate_dim,
                hidden_dims=cfg.hidden_dims,
                context_dims=cfg.context_dims,
                norm=cfg.norm,
            )
        )
        self.update_block = MultiScaleStereoUpdateBlock(
            MultiScaleUpdateConfig(
                hidden_dims=cfg.hidden_dims,
                context_dims=cfg.context_dims,
                corr_levels=cfg.corr_levels,
                corr_radius=cfg.corr_radius,
                motion_dim=128,
                upsample_factor=cfg.downsample_factor,
            )
        )

    def forward(
        self,
        left: torch.Tensor,
        right: torch.Tensor,
        *,
        iters: int | None = None,
        flow_init: torch.Tensor | None = None,
        return_lowres: bool = False,
        return_mono: bool = False,
    ):
        self._validate_inputs(left, right)
        num_iters = self.config.iters if iters is None else iters
        left_norm = normalize_uint8_image(left)
        right_norm = normalize_uint8_image(right)
        fmap_left, fmap_right = self.feature_encoder([left_norm, right_norm])
        self._validate_downsample_ratio(left, fmap_left)

        mono_output = self._run_mono(left)
        hidden_states, contexts = self.mono_adapter(mono_output.penultimate)
        context_gates = self.update_block.prepare_contexts(contexts)

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
        for iteration in range(num_iters):
            coords1 = coords1.detach()
            corr = corr_pyramid.sample(coords1)
            flow = coords1 - coords0
            update_32 = (iteration % self.config.update_32_every) == 0
            update_16 = (iteration % self.config.update_16_every) == 0 or update_32
            hidden_states, up_mask, delta_flow = self.update_block(
                hidden_states,
                context_gates,
                corr,
                flow,
                update_8=True,
                update_16=update_16,
                update_32=update_32,
            )
            coords1 = coords1 + delta_flow
            flow_lowres = coords1 - coords0
            fullres_flow = upsample_flow(
                flow_lowres,
                factor=self.config.downsample_factor,
                mask_logits=up_mask if self.config.use_convex_upsampling else None,
            )
            predictions.append(fullres_flow[:, :1])

        outputs: list[object] = [predictions]
        if return_lowres:
            outputs.append((coords1 - coords0)[:, :1])
        if return_mono:
            outputs.append(mono_output.inverse_depth)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def _run_mono(self, left: torch.Tensor) -> DepthAnythingBackboneOutput:
        output = self.mono_backbone(left)
        if not isinstance(output, DepthAnythingBackboneOutput):
            raise TypeError("mono_backbone must return DepthAnythingBackboneOutput")
        return output

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
        factor = max(self.config.downsample_factor, self.config.min_context_factor)
        if left.shape[-2] % factor != 0 or left.shape[-1] % factor != 0:
            raise ValueError(
                "Milestone 4 expects image sizes divisible by the largest recurrent scale. "
                f"Got HxW={left.shape[-2:]} with required factor={factor}."
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
                f"Encoder downsample factor mismatch: expected {self.config.downsample_factor}, got ({factor_h}, {factor_w})"
            )
