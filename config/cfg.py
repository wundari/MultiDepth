# %%
import torch
from dataclasses import dataclass, field


# %%
@dataclass(frozen=True)
class EncoderConfig:
    stem_dim: int = 64
    stage_dims: tuple[int, int, int] = (64, 96, 128)
    # norm: str = "instance"


@dataclass(frozen=True)
class MultiScaleContextConfig:
    hidden_dims: tuple[int, int, int] = (128, 96, 64)
    context_dims: tuple[int, int, int] = (128, 96, 64)
    stem_dim: int = 64
    stage_dims: tuple[int, int, int, int, int] = (64, 96, 128, 160, 192)
    norm: str = "instance"


@dataclass(frozen=True)
class UpdateBlockConfig:
    hidden_dim: int = 128
    context_dim: int = 128
    corr_levels: int = 4
    corr_radius: int = 4
    motion_dim: int = 128
    upsample_factor: int = 8

    @property
    def corr_channels(self) -> int:
        return self.corr_levels * (2 * self.corr_radius + 1)


@dataclass(frozen=True)
class MultiScaleUpdateConfig:
    hidden_dims: tuple[int, int, int] = (128, 96, 64)
    context_dims: tuple[int, int, int] = (128, 96, 64)
    corr_levels: int = 4
    corr_radius: int = 4
    motion_dim: int = 128
    upsample_factor: int = 8

    @property
    def corr_channels(self) -> int:
        return self.corr_levels * (2 * self.corr_radius + 1)


@dataclass(frozen=True)
class RAFTStereoCoreConfig:
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


@dataclass(frozen=True)
class MonoContextAdapterConfig:
    in_dim: int = 128
    hidden_dims: tuple[int, int, int] = (128, 96, 64)
    context_dims: tuple[int, int, int] = (128, 96, 64)
    norm: str = "instance"


# @dataclass(frozen=True)
# class DepthAnythingConfig:
#     encoder: str = "vitl"
#     target_size: int = 518
#     resize_multiple: int = 14
#     output_stride: int = 8
#     freeze: bool = True
#     weights_path: str | None = None


@dataclass(frozen=True)
class DepthAnythingConfig:
    """Configuration for the frozen monocular prior wrapper.

    Notes
    -----
    The released project resizes the RGB image to a long side around 518 and rounds
    the new size to a multiple of 14 before running DepthAnythingV2. This tutorial
    wrapper makes that behavior optional so the code remains readable and testable.
    """

    penultimate_dim: int = 128
    output_stride: int = 8
    resize_long_side: int | None = None
    resize_multiple_of: int = 14
    freeze_backbone: bool = True
    use_imagenet_normalization: bool = True


@dataclass(frozen=True)
class DepthAnythingBackboneOutput:
    inverse_depth: torch.Tensor
    penultimate: torch.Tensor


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0


@dataclass(frozen=True)
class QwenVisionOutput:
    tokens: torch.Tensor  # [B, N, C]
    pooled: torch.Tensor  # [B, C]
    spatial: torch.Tensor  # [B, C, H/8, W/8]
    prompt_embedding: torch.Tensor  # [B, C]


@dataclass(frozen=True)
class MockQwen2VLConfig:
    embed_dim: int = 256
    patch_dim: int = 256
    norm: str = "instance"
    freeze_backbone: bool = False
    output_stride: int = 8


@dataclass(frozen=True)
class QwenVisionConfig:
    backend: str = "mock"
    qwen_dim: int = 256
    flux_dim: int = 256
    output_stride: int = 8
    freeze_visual_backbone: bool = False
    checkpoint_dir: str | None = None


@dataclass(frozen=True)
class PromptBundle:
    """
    Structured prompt pieces for the VLM confidence branch.

    The released code hardcodes a prompt asking the VLM to highlight transparent
    or reflective objects such as mirrors, glass, windows, and showcases.
    We keep the same intent, but make the prompt builder explicit and swappable.
    """

    system: str
    user: str

    @property
    def full_text(self) -> str:
        return f"{self.system}\n\n{self.user}".strip()


@dataclass(frozen=True)
class FluxConfidenceConfig:
    qwen: QwenVisionConfig = QwenVisionConfig()
    corr_levels: int = 4
    corr_radius: int = 4
    hidden_dim: int = 128
    latent_dim: int = 256
    num_blocks: int = 3
    num_heads: int = 8
    mlp_ratio: int = 4
    use_hidden_features: bool = True
    use_beta_statistics: bool = True

    @property
    def corr_channels(self) -> int:
        return self.corr_levels * (2 * self.corr_radius + 1)

    @property
    def latent_input_channels(self) -> int:
        channels = self.corr_channels + 2  # stereo disp + mono disp
        if self.use_hidden_features:
            channels += self.hidden_dim
        if self.use_beta_statistics:
            channels += 2
        return channels


@dataclass(frozen=True)
class AffineAlignmentConfig:
    hidden_dim: int = 32
    global_pool: bool = False


@dataclass(frozen=True)
class AffineAlignmentOutput:
    a: torch.Tensor
    b: torch.Tensor
    registered_depth: torch.Tensor


@dataclass(frozen=True)
class StereoMonoVLMRefinementConfig:
    corr_levels: int = 4
    corr_radius: int = 4
    hidden_dim: int = 128
    upsample_factor: int = 8
    use_hidden_features: bool = True
    use_beta_statistics: bool = True
    global_pool_affine: bool = False
    flux_confidence: FluxConfidenceConfig = field(default_factory=FluxConfidenceConfig)


@dataclass(frozen=True)
class StereoMonoVLMRefinementOutput:
    fused_disp: torch.Tensor
    up_mask: torch.Tensor
    depth_registered: torch.Tensor
    confidence_logits: torch.Tensor
    confidence: torch.Tensor
    a: torch.Tensor
    b: torch.Tensor
    vlm_aux: dict[str, torch.Tensor]
