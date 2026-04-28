from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from illusion_depth.models.fusion.refine import StereoMonoRefinement, StereoMonoRefinementConfig
from illusion_depth.models.mono.context_adapter import MonoContextAdapter, MonoContextAdapterConfig
from illusion_depth.models.mono.depthanything_wrapper import DepthAnythingConfig, FrozenDepthAnythingV2
from illusion_depth.models.priors.beta_modulator import BetaModulator, BetaModulatorConfig, BetaModulationOutput
from illusion_depth.models.priors.lbp import LBPConfig, LocalBinaryPattern
from illusion_depth.models.stereo.corr import CorrelationPyramid1D
from illusion_depth.models.stereo.encoder import BasicEncoder, EncoderConfig
from illusion_depth.models.stereo.update_block import MultiScaleStereoUpdateBlock, MultiScaleUpdateConfig
from illusion_depth.models.stereo.upsample import upsample_flow
from illusion_depth.utils.geometry import make_coords_grid, normalize_uint8_image
from illusion_depth.utils.schedules import modulation_weight


@dataclass(frozen=True)
class RAFTStereoMonoBetaFusionConfig:
    feature_dim: int = 128
    hidden_dims: tuple[int, int, int] = (128, 96, 64)
    context_dims: tuple[int, int, int] = (128, 96, 64)
    corr_levels: int = 4
    corr_radius: int = 4
    iters: int = 6
    downsample_factor: int = 8
    min_context_factor: int = 32
    update_16_every: int = 2
    update_32_every: int = 4
    norm: str = "instance"
    use_convex_upsampling: bool = True
    mono_penultimate_dim: int = 128
    modulation_schedule: str = "linear"
    modulation_ratio: float = 1.0
    lbp_neighbor_offsets: tuple[tuple[int, int], ...] = ((-1, -1), (1, 1), (1, -1), (-1, 1))
    use_hidden_features_in_confidence: bool = False
    use_beta_statistics_in_confidence: bool = True
    global_pool_affine: bool = False


class RAFTStereoMonoBetaFusionCore(nn.Module):
    """Stereo core + frozen monocular prior + beta modulation + confidence fusion.

    Important sign convention
    -------------------------
    `disp_predictions` are returned in the same negative horizontal-flow convention
    used by the released training code. Internally, the final refinement step flips
    the sign to positive disparity, applies affine fusion, then flips the result back.
    """

    def __init__(self, config: RAFTStereoMonoBetaFusionConfig | None = None) -> None:
        super().__init__()
        cfg = config or RAFTStereoMonoBetaFusionConfig()
        self.config = cfg

        self.feature_encoder = BasicEncoder(cfg.feature_dim, config=EncoderConfig(norm=cfg.norm))
        self.mono = FrozenDepthAnythingV2(
            config=DepthAnythingConfig(
                penultimate_dim=cfg.mono_penultimate_dim,
                output_stride=cfg.downsample_factor,
                resize_long_side=None,
                freeze_backbone=True,
            )
        )
        self.context_adapter = MonoContextAdapter(
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
        self.lbp = LocalBinaryPattern(LBPConfig(neighbor_offsets=cfg.lbp_neighbor_offsets))
        self.beta_modulator = BetaModulator(
            BetaModulatorConfig(lbp_dim=self.lbp.num_neighbors, hidden_dim=None, norm=cfg.norm)
        )
        self.refinement = StereoMonoRefinement(
            StereoMonoRefinementConfig(
                corr_levels=cfg.corr_levels,
                corr_radius=cfg.corr_radius,
                hidden_dim=cfg.hidden_dims[0],
                upsample_factor=cfg.downsample_factor,
                use_hidden_features=cfg.use_hidden_features_in_confidence,
                use_beta_statistics=cfg.use_beta_statistics_in_confidence,
                global_pool_affine=cfg.global_pool_affine,
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
    ) -> dict[str, torch.Tensor | list[torch.Tensor] | list[torch.Tensor]]:
        self._validate_inputs(left, right)
        num_iters = self.config.iters if iters is None else iters

        left_norm = normalize_uint8_image(left)
        right_norm = normalize_uint8_image(right)
        fmap_left, fmap_right = self.feature_encoder([left_norm, right_norm])
        self._validate_downsample_ratio(left, fmap_left)

        mono = self.mono(left)
        depth_lowres = mono.inverse_depth
        mono_features = mono.penultimate
        if depth_lowres.shape[-2:] != fmap_left.shape[-2:]:
            depth_lowres = F.interpolate(depth_lowres, size=fmap_left.shape[-2:], mode="bilinear", align_corners=False)
        if mono_features.shape[-2:] != fmap_left.shape[-2:]:
            mono_features = F.interpolate(mono_features, size=fmap_left.shape[-2:], mode="bilinear", align_corners=False)

        context_outputs = self.context_adapter(mono_features)
        hidden_states, contexts = self._split_context_outputs(context_outputs)
        context_gates = self.update_block.prepare_contexts(contexts)

        batch, _, h_low, w_low = fmap_left.shape
        coords0 = make_coords_grid(batch, h_low, w_low, device=fmap_left.device, dtype=fmap_left.dtype)
        coords1 = coords0.clone()
        if flow_init is not None:
            if flow_init.shape != coords1.shape:
                raise ValueError(f"flow_init must match low-res coords shape {coords1.shape}, got {flow_init.shape}")
            coords1 = coords1 + flow_init

        corr_pyramid = CorrelationPyramid1D(fmap_left, fmap_right, num_levels=self.config.corr_levels, radius=self.config.corr_radius)

        disp_predictions: list[torch.Tensor] = []
        modulation_predictions: list[torch.Tensor] = []
        beta_out: BetaModulationOutput | None = None
        up_mask = None
        for iteration in range(num_iters):
            coords1 = coords1.detach()
            corr = corr_pyramid.sample(coords1)
            flow = coords1 - coords0

            disp_lbp = self.lbp(flow[:, :1])
            depth_lbp = self.lbp(depth_lowres)
            beta_out = self.beta_modulator(disp_lbp, depth_lbp, return_distribution=True)
            modulation_predictions.append(beta_out.modulation)

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

            weight = modulation_weight(
                iteration,
                num_iters,
                mode=self.config.modulation_schedule,
                ratio=self.config.modulation_ratio,
            )
            delta_flow_x = delta_flow[:, :1] * (1.0 + beta_out.modulation * weight)
            delta_flow = torch.cat([delta_flow_x, torch.zeros_like(delta_flow[:, 1:2])], dim=1)
            coords1 = coords1 + delta_flow

            flow_lowres = coords1 - coords0
            fullres_flow = upsample_flow(
                flow_lowres,
                factor=self.config.downsample_factor,
                mask_logits=up_mask if self.config.use_convex_upsampling else None,
            )
            disp_predictions.append(fullres_flow[:, :1])

        if beta_out is None or up_mask is None:
            raise RuntimeError("Model executed zero iterations; no refinement inputs were produced")

        corr = corr_pyramid.sample(coords1.detach())
        disp_positive = -(coords1 - coords0)[:, :1]
        refine_out = self.refinement(
            disp_positive,
            depth_lowres,
            hidden_states[0],
            corr,
            beta_distribution=beta_out.distribution if self.config.use_beta_statistics_in_confidence else None,
        )

        depth_registered_neg_up = upsample_flow(
            -refine_out.depth_registered,
            factor=self.config.downsample_factor,
            mask_logits=refine_out.up_mask if self.config.use_convex_upsampling else None,
        )
        fused_neg_up = upsample_flow(
            -refine_out.fused_disp,
            factor=self.config.downsample_factor,
            mask_logits=refine_out.up_mask if self.config.use_convex_upsampling else None,
        )
        disp_predictions.append(depth_registered_neg_up)
        disp_predictions.append(fused_neg_up)

        result: dict[str, torch.Tensor | list[torch.Tensor] | list[torch.Tensor]] = {
            "disp_predictions": disp_predictions,
            "conf": refine_out.confidence_logits,
            "confidence_prob": refine_out.confidence,
            "depth": depth_lowres,
            "depth_registered": refine_out.depth_registered,
            "stereo_disp_positive": disp_positive,
            "beta_modulations": modulation_predictions,
            "affine_a": refine_out.a,
            "affine_b": refine_out.b,
        }
        if return_lowres:
            result["lowres_fused_disp"] = refine_out.fused_disp
            result["lowres_stereo_disp"] = disp_positive
        return result

    def _split_context_outputs(self, context_outputs: list[torch.Tensor]):
        hidden_states = []
        contexts = []
        for out, hidden_dim, context_dim in zip(context_outputs, self.config.hidden_dims, self.config.context_dims):
            hidden, context = torch.split(out, [hidden_dim, context_dim], dim=1)
            hidden_states.append(torch.tanh(hidden))
            contexts.append(F.relu(context))
        return tuple(hidden_states), tuple(contexts)

    def _validate_inputs(self, left: torch.Tensor, right: torch.Tensor) -> None:
        if left.ndim != 4 or right.ndim != 4:
            raise ValueError(f"Expected [B, 3, H, W] inputs, got {left.shape} and {right.shape}")
        if left.shape != right.shape:
            raise ValueError(f"Left/right shape mismatch: {left.shape} vs {right.shape}")
        if left.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {left.shape}")
        factor = max(self.config.downsample_factor, self.config.min_context_factor)
        if left.shape[-2] % factor != 0 or left.shape[-1] % factor != 0:
            raise ValueError(
                "Milestone 6 expects image sizes divisible by the largest recurrent scale. "
                f"Got HxW={left.shape[-2:]} with required factor={factor}."
            )

    def _validate_downsample_ratio(self, fullres: torch.Tensor, lowres: torch.Tensor) -> None:
        factor_h = fullres.shape[-2] // lowres.shape[-2]
        factor_w = fullres.shape[-1] // lowres.shape[-1]
        if factor_h != self.config.downsample_factor or factor_w != self.config.downsample_factor:
            raise RuntimeError(
                f"Encoder downsample factor mismatch: expected {self.config.downsample_factor}, got ({factor_h}, {factor_w})"
            )
