# %%
import torch
import torch.nn as nn
from torch import Tensor

from models.stereo.corr import CorrelationPyramid1D
from models.stereo.encoder import BasicEncoder, MultiScaleContextEncoder
from models.stereo.update_block import MultiScaleStereoUpdateBlock
from models.stereo.upsample import upsample_flow
from utils.geometry import make_coords_grid, normalize_uint8_image
from config.cfg import (
    EncoderConfig,
    RAFTStereoCoreConfig,
    MultiScaleContextConfig,
    MultiScaleUpdateConfig,
)

from jaxtyping import Float


# %%
class RAFTStereoCore(nn.Module):
    """
    RAFTStereoCore is the top-level model —
    it wires every module into a single forward pass. The logic is:

    Encode features from both images (for cost volume)
    Encode context from left image only (for GRU initialisation and gating)
    Build a correlation pyramid from the feature maps
    Run num_iters GRU refinement steps, each producing a better disparity estimate
    Return all intermediate predictions (for sequence loss during training)
    """

    def __init__(self, config: RAFTStereoCoreConfig) -> None:

        super().__init__()
        self.config = config

        self.feature_encoder = BasicEncoder(config.feature_dim, config=EncoderConfig())
        self.context_encoder = MultiScaleContextEncoder(
            MultiScaleContextConfig(
                hidden_dims=config.hidden_dims, context_dims=config.context_dims
            )
        )
        self.update_block = MultiScaleStereoUpdateBlock(
            MultiScaleUpdateConfig(
                hidden_dims=config.hidden_dims,
                context_dims=config.context_dims,
                corr_levels=config.corr_levels,
                corr_radius=config.corr_radius,
                motion_dim=128,
                upsample_factor=config.downsample_factor,
            )
        )

    def forward(
        self,
        left: Float[Tensor, "B C H W"],
        right: Float[Tensor, "B C H W"],
        iters: int | None = None,
        flow_init: Float[Tensor, "B C H W"] | None = None,
        return_lowres: bool = False,
    ):

        self._validate_inputs(left, right)
        num_iters = self.config.iters if iters is None else iters
        left_norm = normalize_uint8_image(left)
        right_norm = normalize_uint8_image(right)
        feat_map_l, feat_map_r = self.feature_encoder([left_norm, right_norm])
        self._validate_downsample_ratio(left, feat_map_l)
        context_outputs = self.context_encoder(left_norm)
        hidden_states, contexts = self._split_context_outputs(context_outputs)
        context_gates = self.update_block.prepare_contexts(contexts)

        batch, _, h8, w8 = feat_map_l.shape
        coords0 = make_coords_grid(
            batch, h8, w8, device=feat_map_l.device, dtype=feat_map_l.dtype
        )
        coords1 = coords0.clone()

        if flow_init is not None:
            if flow_init.shape != coords1.shape:
                raise ValueError(
                    f"flow_init must match low-res corrds shape {coords1.shape}, got {flow_init.shape}"
                )
            coords1 = coords1 + flow_init
        corr_pyramid = CorrelationPyramid1D(
            feat_map_l,
            feat_map_r,
            num_levels=self.config.corr_levels,
            radius=self.config.corr_radius,
        )

        predictions: list[Tensor] = []
        for i in range(num_iters):
            coords1 = coords1.detach()
            corr = corr_pyramid.sample(coords1)
            flow = coords1 - coords0
            update_32 = (i % self.config.update_32_every) == 0
            update_16 = (i % self.config.update_16_every) == 0 or update_32

            hidden_states, up_mask, delta_flow = self.update_block(
                hidden_states,
                context_gates,
                corr=corr,
                flow=flow,
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
        if return_lowres:
            return predictions, (coords1 - coords0)[:, :1]
        return predictions

    def _split_context_outputs(
        self,
        context_outputs: tuple[Tensor, Tensor, Tensor],
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...]]:
        hidden_states, contexts = [], []
        h_dims = self.config.hidden_dims  # (h0, h1, h2)
        for feat, h_dim in zip(context_outputs, h_dims):
            h, ctx = feat.split([h_dim, feat.shape[1] - h_dim], dim=1)
            hidden_states.append(torch.tanh(h))  # tanh-init is standard for GRU hidden
            contexts.append(ctx)
        return tuple(hidden_states), tuple(contexts)

    def _validate_inputs(
        self, left: Float[Tensor, "B C H W"], right: Float[Tensor, "B C H W"]
    ) -> None:

        if left.ndim != 4 or right.ndim != 4:
            raise ValueError(
                f"Expected [B, 3, H, W] inputs, got {left.shape} and {right.shape}"
            )

        if left.shape != right.shape:
            raise ValueError(
                f"Left/right shape mismatch: {left.shape} vs. {right.shape}"
            )

        if left.shape[1] != 3:
            raise ValueError(f"Expected 3-channel RGB input, got {left.shape}")

        factor = max(self.config.downsample_factor, self.config.min_context_factor)
        H, W = left.shape[-2], left.shape[-1]
        if H % factor != 0 or W % factor != 0:
            raise ValueError(f"Input H×W ({H}×{W}) must be divisible by {factor}")

    def _validate_downsample_ratio(
        self, fullres: Float[Tensor, "B C H W"], lowres: Float[Tensor, "B C H W"]
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
