# %%
import torch
import torch.nn as nn

from multiprocessing import freeze_support
from pathlib import Path
from time import perf_counter

from models.stereo.raft_stereo_mono_beta_vlm_flux_core import (
    RAFTStereoMonoBetaVLMFluxConfig,
    RAFTStereoMonoBetaVLMFluxCore,
)
from models.vlm.qwen_encoder import QwenVisionConfig
from models.vlm.flux_confidence import FluxConfidenceConfig
from models.fusion.refine_vlm import StereoMonoVLMRefinementConfig
from training.dataloader import build_training_dataloader
from training.stages import (
    StagePreset,
    apply_training_stage,
    summarize_trainable_parameters,
)
from training.optim import OptimizerConfig, build_optimizer, build_scheduler
from training.checkpoint import load_checkpoint, save_checkpoint

from losses.confidence import sequence_l1_with_confidence_loss

from config.cfg_train import TrainConfig


def build_model(backend: str) -> RAFTStereoMonoBetaVLMFluxCore:
    qwen_cfg = QwenVisionConfig(
        backend=backend, freeze_visual_backbone=(backend == "real_qwen2vl")
    )
    flux_cfg = FluxConfidenceConfig(qwen=qwen_cfg)
    refine_cfg = StereoMonoVLMRefinementConfig(flux_confidence=flux_cfg)
    model_cfg = RAFTStereoMonoBetaVLMFluxConfig(vlm_refinement=refine_cfg)
    return RAFTStereoMonoBetaVLMFluxCore(model_cfg)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainConfig,
    device: torch.device,
    epoch: int = 0,
    start_step: int = 0,
    stage_name: str = "",
) -> tuple[int, dict[str, float]]:

    model.train()

    metrics: dict[str, float] = {}
    log_totals: dict[str, float] = {}
    log_count = 0
    step = start_step
    last_log = perf_counter()

    for batch in loader:

        if step >= config.max_steps:
            break

        left = batch["left"].to(device, non_blocking=True)
        right = batch["right"].to(device, non_blocking=True)
        target = batch["target_flow"].to(device, non_blocking=True)
        valid = batch["valid"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        use_amp = config.amp and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=use_amp,
        ):
            outputs = model(left, right)
            loss, batch_metrics = sequence_l1_with_confidence_loss(
                outputs["disp_predictions"], target, valid, outputs["conf"]
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        step += 1
        metrics = {**batch_metrics}
        metrics["lr"] = optimizer.param_groups[0]["lr"]
        for key, value in batch_metrics.items():
            log_totals[key] = log_totals.get(key, 0.0) + float(value)
        log_count += 1

        if step % config.log_every == 0:
            now = perf_counter()
            avg_metrics = {
                key: value / max(log_count, 1) for key, value in log_totals.items()
            }
            metrics.update({f"avg_{key}": value for key, value in avg_metrics.items()})
            metrics["iter_time_sec"] = (now - last_log) / float(max(log_count, 1))
            last_log = now
            log_totals.clear()
            log_count = 0

            print(
                f"[epoch {epoch:03d} step {step:06d}] "
                f"loss={avg_metrics['loss']:.4f} flow={avg_metrics['flow_loss']:.4f} "
                f"conf={avg_metrics['confidence_loss']:.4f} "
                f"epe={avg_metrics['epe']:.4f} "
                f"valid={avg_metrics['valid_fraction']:.3f} "
                f"target={avg_metrics['target_abs_mean']:.1f} "
                f"lr={metrics['lr']:.2e}"
            )

        if step % config.save_every == 0:
            ckpt_path = Path(config.output_dir) / f"checkpoint_step_{step:06d}.pt"
            save_checkpoint(
                ckpt_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=step,
                stage_name=stage_name,
                metrics=metrics,
            )

    return step, metrics


def main() -> int:
    train_config = TrainConfig()

    loader = build_training_dataloader(
        root=train_config.root,
        batch_size=train_config.batch_size,
        num_workers=train_config.num_workers,
        crop_size=train_config.crop_size,
        use_mask_as_valid=train_config.use_mask_as_valid,
    )

    device = torch.device(train_config.device)
    model = build_model(train_config.backend).to(device)
    stage = StagePreset.by_name(train_config.stage)
    if stage.name == "vlm_adapters" and train_config.restore is None:
        raise ValueError(
            "stage='vlm_adapters' freezes the geometry stack and requires "
            "TrainConfig.restore to point at a pretrained checkpoint. Use "
            "stage='scratch_finetune' when training this mock-backed model from scratch."
        )
    stage_report = apply_training_stage(model, stage)
    summary = summarize_trainable_parameters(model)
    print(f"Stage: {stage.name}")
    print(f"LoRA targets: {len(stage_report['lora_targets'])}")
    print(f"Trainable params: {summary['trainable']} / {summary['total']}")

    optim_cfg = OptimizerConfig(
        lr=stage.base_lr,
        weight_decay=stage.weight_decay,
        warmup_steps=train_config.warmup_steps,
        total_steps=train_config.max_steps,
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
        device.type,
        enabled=train_config.amp and device.type == "cuda",
    )

    start_epoch = 0
    start_step = 0
    if train_config.restore is not None:
        restore_info = load_checkpoint(
            train_config.restore,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            strict=train_config.strict_load,
        )
        start_epoch = restore_info["epoch"]
        start_step = restore_info["step"]
        print(f"Loaded checkpoint: {train_config.restore}")
        print(
            f"Missing keys: {len(restore_info['missing_keys'])}, "
            f"unexpected keys: {len(restore_info['unexpected_keys'])}"
        )

    step = start_step
    epoch = start_epoch
    metrics: dict[str, float] = {}
    while step < train_config.max_steps:
        previous_step = step
        step, metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            train_config,
            device,
            epoch=epoch,
            start_step=step,
            stage_name=stage.name,
        )
        if step == previous_step:
            break
        epoch += 1

    save_checkpoint(
        Path(train_config.output_dir) / "checkpoint_last.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=step,
        stage_name=stage.name,
        metrics=metrics,
    )
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())


# %%
