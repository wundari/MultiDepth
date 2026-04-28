# %%
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from data.hf_index import build_hf_training_index
from models.engine.trainer import TrainConfig, train_loop
from models.fusion.refine_vlm import StereoMonoVLMRefinementConfig
from models.stereo.raft_stereo_mono_beta_vlm_flux_core import (
    RAFTStereoMonoBetaVLMFluxConfig,
    RAFTStereoMonoBetaVLMFluxCore,
)
from models.vlm.flux_confidence import FluxConfidenceConfig
from models.vlm.qwen_encoder import QwenVisionConfig
from training.optim import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from training.checkpoint import (
    load_checkpoint,
    save_checkpoint,
)
from training.dataloader import StereoBatchCollator
from training.stages import StagePreset, apply_training_stage
from torch.utils.data import DataLoader
from data.dataset import IllusionDepthDataset

# %%


def _write_png(path: Path, array: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def _make_toy_root(tmp_path: Path) -> Path:
    root = tmp_path / "toy_hf"
    split = root / "fooling3D"
    left = np.full((64, 96, 3), 120, dtype=np.uint8)
    right = np.full((64, 96, 3), 122, dtype=np.uint8)
    depth = np.full((64, 96), 10, dtype=np.uint16)
    mask = np.zeros((64, 96), dtype=np.uint8)
    mask[8:56, 16:80] = 255
    _write_png(split / "left" / "video_000" / "0001.png", left)
    _write_png(split / "right" / "video_000" / "0001.png", right)
    _write_png(split / "depth" / "video_000" / "0001.png", depth)
    _write_png(split / "mask" / "video_000" / "0001-illusion.jpg", mask)
    (root / "scale_factors.csv").write_text(
        "video_frame_sequence_right/video_000/0001.png,2.0\n"
    )
    return root


def _make_small_model() -> RAFTStereoMonoBetaVLMFluxCore:
    qwen = QwenVisionConfig(backend="mock", qwen_dim=64, flux_dim=64)
    flux = FluxConfidenceConfig(
        qwen=qwen, hidden_dim=32, latent_dim=64, num_blocks=1, num_heads=4
    )
    refine = StereoMonoVLMRefinementConfig(hidden_dim=32, flux_confidence=flux)
    cfg = RAFTStereoMonoBetaVLMFluxConfig(
        feature_dim=64,
        hidden_dims=(32, 24, 16),
        context_dims=(32, 24, 16),
        iters=2,
        mono_penultimate_dim=64,
        vlm_refinement=refine,
    )
    return RAFTStereoMonoBetaVLMFluxCore(cfg)


def test_stage_freezing_and_lora_targets(tmp_path: Path) -> None:
    model = _make_small_model()
    report = apply_training_stage(model, StagePreset.vlm_adapters())
    trainable = report["trainable"]
    assert any("qwen.connector" in name for name in trainable)
    assert any("lora_a" in name or "lora_b" in name for name in trainable)
    assert all(not name.startswith("mono.backbone") for name in trainable)


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    model = _make_small_model()
    apply_training_stage(model, StagePreset.vlm_adapters())
    optimizer = build_optimizer(model, lr=1e-4, weight_decay=1e-5, lr_multipliers={})
    scheduler = build_scheduler(optimizer, OptimizerConfig(total_steps=4))
    ckpt_path = tmp_path / "ckpt.pt"
    save_checkpoint(
        ckpt_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        step=7,
        stage_name="vlm_adapters",
    )

    model2 = _make_small_model()
    apply_training_stage(model2, StagePreset.vlm_adapters())
    optimizer2 = build_optimizer(model2, lr=1e-4, weight_decay=1e-5, lr_multipliers={})
    scheduler2 = build_scheduler(optimizer2, OptimizerConfig(total_steps=4))
    info = load_checkpoint(
        ckpt_path,
        model=model2,
        optimizer=optimizer2,
        scheduler=scheduler2,
        strict=False,
    )
    assert info["epoch"] == 3
    assert info["step"] == 7


def test_single_train_step_smoke(tmp_path: Path) -> None:
    torch.set_num_threads(1)
    root = _make_toy_root(tmp_path)
    records = build_hf_training_index(root)
    dataset = IllusionDepthDataset(records, use_mask_as_valid=True)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=StereoBatchCollator(
            crop_size=(64, 96), divisor=32, random_crop=False
        ),
    )

    model = _make_small_model()
    stage = StagePreset.vlm_adapters()
    apply_training_stage(model, stage)
    optimizer = build_optimizer(
        model,
        lr=stage.base_lr,
        weight_decay=stage.weight_decay,
        lr_multipliers=stage.lr_multipliers,
    )
    scheduler = build_scheduler(
        optimizer, OptimizerConfig(total_steps=1, warmup_steps=0)
    )
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    cfg = TrainConfig(
        device="cpu",
        amp=False,
        max_steps=1,
        log_every=1,
        save_every=1000,
        output_dir=str(tmp_path),
    )
    metrics = train_loop(
        model, loader, optimizer, scheduler, scaler, cfg, stage_name=stage.name
    )
    assert "loss" in metrics
