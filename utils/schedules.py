from __future__ import annotations

import math


def modulation_weight(
    iteration: int, total_iterations: int, *, mode: str = "linear", ratio: float = 1.0
) -> float:
    if total_iterations <= 0:
        raise ValueError(f"total_iterations must be positive, got {total_iterations}")
    progress = (iteration + 1) / total_iterations
    mode = mode.lower()
    if mode == "linear":
        return ratio * progress
    if mode == "sigmoid":
        return ratio * (1.0 / (1.0 + math.exp(-12.0 * (progress - 0.5))))
    raise ValueError(f"Unsupported modulation schedule: {mode}")
