# %%
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from models.stereo.raft_stereo_mono_beta_vlm_flux_core import (
    RAFTStereoMonoBetaVLMFluxConfig,
    RAFTStereoMonoBetaVLMFluxCore,
)
from models.vlm.qwen_encoder import QwenVisionConfig
from models.vlm.flux_confidence import FluxConfidenceConfig
from models.fusion.refine_vlm import StereoMonoVLMRefinementConfig
from training.dataloader import build_training_dataloader
from training.optim import (
    OptimizerConfig,
    build_optimizer,
    build_scheduler,
)
from training.checkpoint import load_checkpoint
from training.stages import (
    StagePreset,
    StageSpec,
    summarize_trainable_parameters,
    apply_training_stage,
)

from engine.trainer import train_loop
from config.cfg_train import TrainConfig

# %%


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 8 training script for the HF 3D illusion dataset"
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Extracted HF dataset root"
    )
    parser.add_argument(
        "--index-path", type=Path, default=None, help="Optional cached JSONL index"
    )
    parser.add_argument("--output-dir", type=Path, default="outputs")
    parser.add_argument(
        "--stage",
        choices=["vlm_adapters", "fusion_finetune", "full_finetune"],
        default="vlm_adapters",
    )
    parser.add_argument(
        "--restore",
        type=Path,
        default=None,
        help="Optional checkpoint to warm-start from",
    )
    parser.add_argument("--strict-load", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--crop-h", type=int, default=256)
    parser.add_argument("--crop-w", type=int, default=384)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--backend", choices=["mock", "real_qwen2vl"], default="mock")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--cpu-threads",
        type=int,
        default=None,
        help="Optional CPU thread cap for debugging / small tests",
    )
    parser.add_argument("--disable-amp", action="store_true")
    return parser.parse_args()


def build_model(backend: str) -> RAFTStereoMonoBetaVLMFluxCore:
    qwen_cfg = QwenVisionConfig(
        backend=backend, freeze_visual_backbone=(backend == "real_qwen2vl")
    )
    flux_cfg = FluxConfidenceConfig(qwen=qwen_cfg)
    refine_cfg = StereoMonoVLMRefinementConfig(flux_confidence=flux_cfg)
    model_cfg = RAFTStereoMonoBetaVLMFluxConfig(vlm_refinement=refine_cfg)
    return RAFTStereoMonoBetaVLMFluxCore(model_cfg)


def main() -> None:
    args = parse_args()
    if args.cpu_threads is not None and args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device if torch.cuda.is_available() else "mps")

    loader = build_training_dataloader(
        args.root,
        # index_path=args.index_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crop_size=(args.crop_h, args.crop_w),
        use_mask_as_valid=True,
    )
    model = build_model(args.backend).to(device)

    stage = StagePreset.by_name(args.stage)
    if args.lr is not None:
        stage = StageSpec(
            name=stage.name,
            train_parameter_patterns=stage.train_parameter_patterns,
            lora_module_patterns=stage.lora_module_patterns,
            always_frozen_patterns=stage.always_frozen_patterns,
            base_lr=args.lr,
            lr_multipliers=stage.lr_multipliers,
            weight_decay=(
                stage.weight_decay if args.weight_decay is None else args.weight_decay
            ),
            lora=stage.lora,
        )
    stage_report = apply_training_stage(model, stage)
    summary = summarize_trainable_parameters(model)
    print(f"Stage: {stage.name}")
    print(f"LoRA targets: {len(stage_report['lora_targets'])}")
    print(f"Trainable params: {summary['trainable']} / {summary['total']}")

    optim_cfg = OptimizerConfig(
        lr=stage.base_lr,
        weight_decay=stage.weight_decay,
        warmup_steps=args.warmup_steps,
        total_steps=args.max_steps,
    )
    optimizer = build_optimizer(
        model,
        lr=optim_cfg.lr,
        weight_decay=optim_cfg.weight_decay,
        lr_multipliers=stage.lr_multipliers,
        betas=optim_cfg.betas,
        eps=optim_cfg.eps,
    )
    scheduler = build_scheduler(optimizer, optim_cfg)
    scaler = torch.amp.GradScaler(
        device.type, enabled=(not args.disable_amp and device.type == "cuda")
    )

    start_epoch = 0
    start_step = 0
    if args.restore is not None:
        restore_info = load_checkpoint(
            args.restore,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            strict=args.strict_load,
        )
        start_epoch = restore_info["epoch"]
        start_step = restore_info["step"]
        print(f"Loaded {args.restore}")
        print(
            f"Missing keys: {len(restore_info['missing_keys'])}, unexpected keys: {len(restore_info['unexpected_keys'])}"
        )

    train_cfg = TrainConfig(
        device=device.type,
        amp=not args.disable_amp,
        max_steps=args.max_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        output_dir=str(args.output_dir),
        max_grad_norm=optim_cfg.max_grad_norm,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    final_metrics = train_loop(
        model,
        loader,
        optimizer,
        scheduler,
        scaler,
        train_cfg,
        start_epoch=start_epoch,
        start_step=start_step,
        stage_name=stage.name,
    )
    print("Final metrics:", final_metrics)


if __name__ == "__main__":
    main()
