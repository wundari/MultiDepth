# %%
from __future__ import annotations
import torch
import torch.nn as nn

from pathlib import Path
from time import perf_counter

from losses.confidence import sequence_l1_with_confidence_loss
from training.checkpoint import save_checkpoint

from config.cfg_train import TrainConfig

# %%


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainConfig,
    epoch: int = 0,
    start_step: int = 0,
    stage_name: str = "",
) -> tuple[int, dict[str, float]]:

    device = torch.device(config.device if torch.cuda.is_available() else "mps")

    model.train()

    metrics: dict[str, float] = {}
    step = start_step
    last_log = perf_counter()

    for batch in loader:

        if step >= config.max_steps:
            break

        left = batch["left"].to(device)
        right = batch["right"].to(device)
        target = batch["target_flow"].to(device)
        valid = batch["valid"].to(device)

        optimizer.zero_grad(set_to_none=True)
        autocast_enabled = config.amp and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            enabled=autocast_enabled,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
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

        if step % config.log_every == 0:
            now = perf_counter()
            metrics["iter_time_sec"] = (now - last_log) / float(config.log_every)
            last_log = now

            print(
                f"[epoch {epoch:03d} step {step:06d}] "
                f"loss={metrics['loss']:.4f} flow={metrics['flow_loss']:.4f} "
                f"conf={metrics['confidence_loss']:.4f} "
                f"epe={metrics['epe']:.4f} lr={metrics['lr']:.2e}"
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


def train_loop(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: torch.cuda.amp.GradScaler,
    config: TrainConfig,
    start_epoch: int = 0,
    start_step: int = 0,
    stage_name: str = "",
) -> dict[str, float]:

    step = start_step
    epoch = start_epoch
    final_metrics: dict[str, float] = {}

    while step < config.max_steps:
        prev_step = step
        step, final_metrics = train_one_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            config,
            epoch=epoch,
            start_step=step,
            stage_name=stage_name,
        )
        if step == prev_step:
            break

        epoch += 1

    final_ckpt = Path(config.output_dir) / "checkpoint_last.pt"
    save_checkpoint(
        final_ckpt,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=epoch,
        step=step,
        stage_name=stage_name,
        metrics=final_metrics,
    )

    return final_metrics
