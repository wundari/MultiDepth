# %%
from pathlib import Path
from dataclasses import dataclass

# %%


@dataclass(frozen=True)
class TrainConfig:
    device: str = "cuda"
    amp: bool = True
    max_steps: int = 1000
    log_every: int = 10
    save_every: int = 100
    output_dir: str = "./outputs"
    max_grad_norm: float = 1.0

    # dataloader config
    root: Path = Path("/media/wundari/S990Pro2_4TB/Dataset/3D_visual_illusion")
    batch_size: int = 16
    num_workers: int = 0
    crop_size: tuple[int, int] = (256, 384)  # [h, w]
    use_mask_as_valid: bool = True

    # model config
    backend: str = "mock"  # backbone
    stage: str = "vlm_adapters"

    # optimizer config
    warmup_steps: int = 100
    max_steps: int = 1000
