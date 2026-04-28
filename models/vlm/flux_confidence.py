# %%
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from models.vlm.prompts import ReflectiveObjectPromptBuilder
from models.vlm.qwen_encoder import QwenVisionBackbone
from config.cfg import FluxConfidenceConfig, QwenVisionConfig
from jaxtyping import Float


# %%
class FluxStyleCrossAttentionBlock(nn.Module):
    """
    This implements a pre-norm Transformer block with three sequential
    sub-layers, each following the pattern of normalise → transform → residual add:

    1. Self-Attention — x attends to itself.
        The query, key, and value are all derived from x,
        letting each position gather context from the rest of the sequence.

    2. Cross-Attention — x attends to an external conditioning signal cond.
        x provides the queries (what to look for), while cond provides
        the keys and values (what to look at).
        This is how the block injects external information —
        e.g. VLM prompt features — into the main feature stream.

    3. FFN — a standard two-layer MLP (dim → hidden_dim → dim) with GELU,
        applied position-wise to mix features after attention.

    The pre-norm design (normalise before each sub-layer rather than after)
    is standard in modern architectures and generally trains more stably.

    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4) -> None:
        super().__init__()

        self.self_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

        self.cross_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True
        )

        # LayerNorm for the conditioning input
        self.cond_norm = nn.LayerNorm(dim)

        self.ffn_norm = nn.LayerNorm(dim)

        hidden_dim = dim * mlp_ratio

        self.ffn = nn.Sequential(
            nn.Linear(in_features=dim, out_features=hidden_dim),
            nn.GELU(),
            nn.Linear(in_features=hidden_dim, out_features=dim),
        )

    def forward(
        self, x: Float[Tensor, "B N dim"], cond: Float[Tensor, "B C H W"]
    ) -> Float[Tensor, "B N dim"]:

        normed = self.self_norm(x)
        x = (
            x
            + self.self_attn(
                normed,  # Q
                normed,  # K
                normed,  # V
                need_weights=False,
            )[0]
        )

        cond_normed = self.cond_norm(cond)
        x = (
            x
            + self.cross_attn(
                self.cross_norm(x),
                cond_normed,
                cond_normed,
                need_weights=False,
            )[0]
        )
        x = x + self.ffn(self.ffn_norm(x))

        return x


class FluxStyleConfidenceDecoder(nn.Module):
    """
    Convert stereo/mono evidence plus Qwen2VL tokens into confidence logits.

    The decoder's job is to take raw stereo/mono feature maps and
    VLM-derived context, and produce a per-pixel confidence map indicating
    where depth estimates are reliable.

    """

    def __init__(self, config: FluxConfidenceConfig) -> None:
        super().__init__()

        self.config = config

        # latent_proj: 1×1 conv projects the input feature channels
        # to a uniform latent_dim
        self.latent_proj = nn.Conv2d(
            in_channels=config.latent_input_channels,
            out_channels=config.latent_dim,
            kernel_size=1,
        )

        # cond_proj: Linear layer projects Qwen token features
        # from flux_dim → latent_dim to match the latent space
        self.cond_proj = nn.Linear(
            in_features=config.qwen.flux_dim, out_features=config.latent_dim
        )
        self.blocks = nn.ModuleList(
            [
                FluxStyleCrossAttentionBlock(
                    dim=config.latent_dim,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self.out_norm = nn.LayerNorm(config.latent_dim)
        self.out_head = nn.Sequential(
            nn.Conv2d(
                in_channels=config.latent_dim,
                out_channels=config.latent_dim // 2,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
            nn.Conv2d(
                in_channels=config.latent_dim // 2, out_channels=1, kernel_size=1
            ),
        )

    def forward(
        self,
        latent_inputs: Float[Tensor, "B latent_input_channels H W"],
        qwen_tokens: Float[Tensor, "B N flux_dim"],
        qwen_prompt_embedding: Float[Tensor, "B flux_dim"],
    ) -> tuple[Tensor, Tensor]:

        B, _, H, W = latent_inputs.shape

        latent = (
            self.latent_proj(latent_inputs).flatten(2).transpose(1, 2)
        )  # [B, HW, D]

        cond_prompt = qwen_prompt_embedding.unsqueeze(1)
        cond = torch.cat(
            [self.cond_proj(qwen_tokens), self.cond_proj(cond_prompt)], dim=1
        )

        for block in self.blocks:
            latent = block(latent, cond)

        latent = self.out_norm(latent)
        latent_map = latent.transpose(1, 2).reshape(B, -1, H, W)
        logits = self.out_head(latent_map)
        confidence = torch.sigmoid(logits)

        return logits, confidence


class QwenFluxConfidenceBranch(nn.Module):
    """
    Qwen2VL * Flux-style confidence branch.

    This is the top-level module that wires all previous components
    into a single forward pass, taking raw stereo inputs and producing
    a per-pixel confidence map.

    Inputs
    ------
    left_rgb:    [B, 3, H, W] raw or normalized RGB image
    cost_volume: [B, Cc, h, w]
    stereo_disp: [B, 1, h, w] positive disparity
    mono_disp:   [B, 1, h, w] positive disparity-like mono prior
    """

    def __init__(self, config: FluxConfidenceConfig) -> None:
        """
        prompt_builder Builds the default reflective-object prompt
            if none is provided at inferenceqwen

        QwenVisionBackbone — encodes the RGB image + prompt into spatial
            tokens and embeddingsdecoder

        FluxStyleConfidenceDecoder — fuses stereo evidence with VLM context
            into confidence logits
        """

        super().__init__()

        self.config = config
        self.prompt_builder = ReflectiveObjectPromptBuilder()
        self.qwen = QwenVisionBackbone(config.qwen)
        self.decoder = FluxStyleConfidenceDecoder(config)

    def forward(
        self,
        left_rgb: Float[Tensor, "B C H W"],
        cost_volume: Float[Tensor, "B Cc H W"],
        stereo_disp: Float[Tensor, "B 1 H W"],
        mono_disp: Float[Tensor, "B 1 H W"],
        hidden: Float[Tensor, "B Ch H W"] | None = None,
        beta_distribution: Beta | None = None,
        prompt: str | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        """
        forward data flow:

        1. Validate inputs — three guards check shape consistency and that
            optional features (hidden, beta_distribution) are supplied when
            the config demands them

        2. Encode image — QwenVisionBackbone processes left_rgb with the prompt,
            returning spatial features, tokens, pooled embedding, and a prompt
            embedding

        3. Align spatial resolution — if the Qwen spatial output doesn't match
            the disparity map resolution (likely due to the 8× stride in
            the backbone), it's bilinearly upsampled to match

        4. Build latent input — stereo evidence is concatenated channel-wise:
            always [cost_volume, stereo_disp, mono_disp], optionally extended
            with hidden features and beta_distribution statistics (mean + variance)

        5. Decode — the concatenated latent and VLM tokens are passed to the
            decoder, returning raw logits and sigmoid confidence

        6. Return — logits and confidence for training/inference, plus an
            aux dict carrying intermediate VLM features for supervision or debugging


        Returns:
            tuple[Tensor, Tensor, dict[str, Tensor]]: _description_
        """

        if stereo_disp.shape != mono_disp.shape:
            raise ValueError(
                f"stereo_disp/mono_disp mismatch: {stereo_disp.shape} vs. {mono_disp.shape}"
            )

        if cost_volume.shape[-2:] != stereo_disp.shape[-2:]:
            raise ValueError(
                "cost_volume and disparity maps must have identical spatial size"
            )

        if self.config.use_hidden_features and hidden is None:
            raise ValueError(
                "Flux confidence branch expects hidden features but hidden = None"
            )

        if self.config.use_beta_statistics and beta_distribution is None:
            raise ValueError(
                f"Flux confidence branch expects beta statistics but beta_distributions=None"
            )

        prompt = prompt if prompt is not None else self.prompt_builder.build().full_text
        qwen_out = self.qwen(
            left_rgb,
            prompt=prompt,
        )

        qwen_spatial = qwen_out.spatial
        if qwen_spatial.shape[-2:] != stereo_disp.shape[-2:]:
            qwen_spatial = F.interpolate(
                qwen_spatial,
                size=stereo_disp.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        latent_inputs = [cost_volume, stereo_disp, mono_disp]
        if self.config.use_hidden_features and hidden is not None:
            latent_inputs.append(hidden)

        if self.config.use_beta_statistics and beta_distribution is not None:
            latent_inputs.extend([beta_distribution.mean, beta_distribution.variance])

        latent_inputs = torch.cat(latent_inputs, dim=1)
        logits, confidence = self.decoder(
            latent_inputs, qwen_out.tokens, qwen_out.prompt_embedding
        )

        aux = {"qwen_spatial": qwen_spatial, "qwen_tokens_pooled": qwen_out.pooled}

        return logits, confidence, aux
