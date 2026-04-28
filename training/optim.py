from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch
from typing import Iterable

import torch
import torch.nn as nn


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 2e-4
    weight_decay: float = 1e-5
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    scheduler: str = "cosine"
    warmup_steps: int = 100
    total_steps: int = 1000
    min_lr_ratio: float = 0.1
    max_grad_norm: float = 1.0


def _matches_prefix(name: str, prefixes: Iterable[str]) -> str | None:
    for prefix in prefixes:
        if name.startswith(prefix):
            return prefix
    return None


def _is_no_decay_parameter(name: str, parameter: nn.Parameter) -> bool:
    if parameter.ndim <= 1:
        return True
    lowered = name.lower()
    return lowered.endswith("bias") or ".norm" in lowered or "embedding" in lowered


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
    lr_multipliers: dict[str, float] | None = None,
    betas: tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    lr_multipliers = lr_multipliers or {}

    groups: dict[tuple[float, float], dict] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        prefix = _matches_prefix(name, sorted(lr_multipliers.keys(), key=len, reverse=True))
        lr_scale = lr_multipliers.get(prefix, 1.0) if prefix is not None else 1.0
        group_lr = lr * lr_scale
        group_wd = 0.0 if _is_no_decay_parameter(name, parameter) else weight_decay
        key = (group_lr, group_wd)
        if key not in groups:
            groups[key] = {"params": [], "lr": group_lr, "weight_decay": group_wd}
        groups[key]["params"].append(parameter)

    if not groups:
        raise ValueError("No trainable parameters found while building optimizer")

    optimizer = torch.optim.AdamW(list(groups.values()), betas=betas, eps=eps)
    return optimizer


def build_scheduler(optimizer: torch.optim.Optimizer, config: OptimizerConfig):
    if config.scheduler == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

    def lr_lambda(step: int) -> float:
        warmup = max(config.warmup_steps, 1)
        if step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(config.total_steps - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        if config.scheduler == "cosine":
            cosine = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.141592653589793))).item()
            return config.min_lr_ratio + (1.0 - config.min_lr_ratio) * cosine
        raise ValueError(f"Unknown scheduler: {config.scheduler}")

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
