from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    epoch: int = 0,
    step: int = 0,
    stage_name: str = "",
    metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "stage_name": stage_name,
        "metrics": metrics or {},
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    strict: bool = False,
    map_location: str = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    state = checkpoint["model"] if "model" in checkpoint else checkpoint
    load_result = model.load_state_dict(state, strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    return {
        "missing_keys": list(getattr(load_result, "missing_keys", [])),
        "unexpected_keys": list(getattr(load_result, "unexpected_keys", [])),
        "epoch": int(checkpoint.get("epoch", 0)),
        "step": int(checkpoint.get("step", 0)),
        "stage_name": checkpoint.get("stage_name", ""),
        "metrics": checkpoint.get("metrics", {}),
        "extra": checkpoint.get("extra", {}),
    }
