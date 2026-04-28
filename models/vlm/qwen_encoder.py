# %%
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from models.vlm.prompts import ReflectiveObjectPromptBuilder
from config.cfg import QwenVisionOutput, MockQwen2VLConfig, QwenVisionConfig
from jaxtyping import Float


# %%
class BytePromptEncoder(nn.Module):
    """
    Tiny prompt encoder for the teaching scaffold.

    This is not Qwen2VL. It is a deterministic, trainable stand-in that lets the rest
    of the architecture run and be unit-tested. The integration point for the real
    Qwen2VL checkpoint is isolated in `QwenVisionBackbone`.

    A lightweight text → embedding module. It has no tokenizer dependency — it
    converts prompt strings directly to UTF-8 bytes (vocab size 256), embeds each byte,
    mean-pools them into a single vector per prompt, then refines it through a small
    2-layer MLP with GELU activation. The result is one embed_dim vector per prompt
    in the batch.

    ComponentPurposeself.embeddingMaps each byte value (0–255) to a learned embed_dim
    vectorself.projTwo-layer Linear → GELU → Linear MLP to refine the pooled byte embedding

    Note: the forward return annotation says "B C H W" but the actual output is
    (B, embed_dim) — a 2D tensor, not a 4D spatial one.
    Same annotation bug as LoRALinear from earlier.
    """

    def __init__(self, embed_dim: int, vocab_size: int = 256) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)
        )

    def forward(
        self, prompts: list[str], device: torch.device
    ) -> Float[Tensor, "B embed_dim"]:

        encoded = []
        for prompt in prompts:
            raw = prompt.encode("utf-8", errors="ignore")[:512]
            if len(raw) == 0:
                raw = b" "
            ids = torch.tensor(list(raw), dtype=torch.long, device=device)
            emb = self.embedding(ids)
            encoded.append(emb.mean(dim=0))
        pooled = torch.stack(encoded, dim=0)

        return self.proj(pooled)


class MockQwen2VLBackbone(nn.Module):
    """
    Small image backbone that mimics the interface of a Qwen2VL visual encoder

    A convolutional image encoder that mimics the interface of the real Qwen2VL
    visual encoder. It processes a [B, 3, H, W] image through four successive
    Conv2d → BatchNorm2d → GELU blocks, progressively expanding channels
    (3 → 64 → 128 → embed_dim → embed_dim) while downsampling the spatial
    resolution 8× (stride-2 three times).
    If config.freeze_backbone is set, the entire stem is frozen.
    """

    def __init__(self, config: MockQwen2VLConfig) -> None:

        super().__init__()

        self.config = config
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3
            ),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(
                in_channels=128,
                out_channels=config.embed_dim,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
            nn.Conv2d(
                in_channels=config.embed_dim,
                out_channels=config.embed_dim,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm2d(config.embed_dim),
            nn.GELU(),
        )

        if config.freeze_backbone:
            self.requires_grad_(False)

    def forward(self, images: Float[Tensor, "B C H W"]) -> Float[Tensor, "B C H W"]:
        return self.stem(images)


# class QwenConnector(nn.Module):
#     """
#     Project Qwen2VL token features into the Flux conditioning space.

#     A thin single linear projection layer.
#     It bridges the Qwen2VL feature dimension (input_dim) to the Flux
#     conditioning dimension (output_dim).
#     No activation — just a linear map.
#     """

#     def __init__(self, input_dim: int, output_dim: int) -> None:
#         super().__init__()
#         self.linear = nn.Linear(in_features=input_dim, out_features=output_dim)

#     def forward(
#         self, tokens: Float[Tensor, "B embed_dim"]
#     ) -> Float[Tensor, "B N output_dim"]:
#         return self.linear(tokens)


class QwenVisionBackbone(nn.Module):
    """
    Prompt-conditioned visual encoder.

    backend = "mock" -> fully working teaching implementation
    backend = "real_qwen2vl" intentionally raises with an explicit message because the released
    repo uses a custom "Qwen2VLSimplifiedModel" from its bundled "qwen2vl-flux" code rather
    than a single standard Transformers API call.

    The top-level orchestrator that wires everything together into a prompt-conditioned visual encoder. Its forward pass:

    1. Builds a batch of text prompts via ReflectiveObjectPromptBuilder
    2. Encodes them to embeddings with BytePromptEncoder
    3. Encodes the image spatially with MockQwen2VLBackbone (or real Qwen2VL)
    4. Fuses prompt and spatial features by broadcasting the prompt embedding over H×W
    5. Flattens spatial features into tokens, projects them through QwenConnector
    6. Returns a QwenVisionOutput with tokens, pooled, spatial, and prompt_embedding

    """

    def __init__(self, config: QwenVisionConfig) -> None:

        super().__init__()
        self.config = config
        self.prompt_builder = ReflectiveObjectPromptBuilder()
        self.prompt_encoder = BytePromptEncoder(config.qwen_dim)

        if config.backend == "mock":
            self.visual = MockQwen2VLBackbone(
                MockQwen2VLConfig(
                    embed_dim=config.qwen_dim,
                    freeze_backbone=config.freeze_visual_backbone,
                    output_stride=config.output_stride,
                )
            )
        elif config.backend == "real_qwen2vl":
            raise NotImplementedError(
                "The released repo loads a custom Qwen2VLSimplifiedModel "
                "from its bundled qwen2vl-flux code. Wire that model into "
                "this class if you want exact heavyweight backend"
            )
        else:
            raise ValueError(f"Unknown Qwen backend: {config.backend}")
        # self.connector = QwenConnector(config.qwen_dim, config.flux_dim)
        self.connector = nn.Linear(
            in_features=config.qwen_dim, out_features=config.flux_dim
        )

    def forward(
        self, images: Float[Tensor, "B C H W"], prompt: str | None = None
    ) -> QwenVisionOutput:

        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W] RGB input, got {images.shape}")

        batch_size = images.shape[0]
        prompts = [
            self.prompt_builder.build().full_text if prompt is None else prompt
            for _ in range(batch_size)
        ]
        prompt_emb = self.prompt_encoder(prompts, device=images.device)

        spatial = self.visual(images)
        spatial = spatial + prompt_emb[:, :, None, None]

        B, C, H, W = spatial.shape
        tokens = spatial.flatten(2).transpose(1, 2)
        tokens = self.connector(tokens)
        pooled = tokens.mean(dim=1)
        spatial_flux = tokens.transpose(1, 2).reshape(B, -1, H, W)
        prompt_flux = self.connector(prompt_emb.unsqueeze(1)).squeeze(1)

        return QwenVisionOutput(
            tokens=tokens,
            pooled=pooled,
            spatial=spatial_flux,
            prompt_embedding=prompt_flux,
        )
