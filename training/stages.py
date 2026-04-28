# %%
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Iterable

import torch
import torch.nn as nn

from models.vlm.lora import LoRAConfig, LoRALinear, apply_lora_by_name

# %%


@dataclass(frozen=True)
class StageSpec:
    name: str
    train_parameter_patterns: tuple[str, ...]
    lora_module_patterns: tuple[str, ...] = ()
    always_frozen_patterns: tuple[str, ...] = ("mono.backbone*",)
    base_lr: float = 2e-4
    lr_multipliers: dict[str, float] = field(default_factory=dict)
    weight_decay: float = 1e-5
    lora: LoRAConfig = field(default_factory=LoRAConfig)


class StagePreset:
    """Readable stage presets for the milestone-7 full model.

    These are not claimed to be byte-for-byte identical to the released training
    scripts. They are a clean reimplementation of the *policy* that the released
    code suggests: keep large pretrained priors frozen, train adapters and fusion
    first, then progressively unfreeze the geometry stack.
    """

    @staticmethod
    def vlm_adapters() -> StageSpec:
        return StageSpec(
            name="vlm_adapters",
            train_parameter_patterns=(
                "refinement.confidence_head.qwen.connector.*",
                "refinement.confidence_head.qwen.prompt_encoder.*",
                "refinement.confidence_head.decoder.*",
                "refinement.affine_alignment.*",
                "refinement.mask_head.*",
            ),
            lora_module_patterns=(
                "refinement.confidence_head.decoder.blocks.*.self_attn.out_proj",
                "refinement.confidence_head.decoder.blocks.*.cross_attn.out_proj",
                "refinement.confidence_head.decoder.blocks.*.ffn.0",
                "refinement.confidence_head.decoder.blocks.*.ffn.2",
                "refinement.confidence_head.qwen.connector.linear",
            ),
            base_lr=2e-4,
            lr_multipliers={
                "refinement.affine_alignment": 0.5,
                "refinement.mask_head": 0.5,
            },
            weight_decay=1e-5,
        )

    @staticmethod
    def fusion_finetune() -> StageSpec:
        return StageSpec(
            name="fusion_finetune",
            train_parameter_patterns=(
                "context_adapter.*",
                "update_block.*",
                "beta_modulator.*",
                "refinement.*",
                "lbp.*",
            ),
            lora_module_patterns=StagePreset.vlm_adapters().lora_module_patterns,
            base_lr=1e-4,
            lr_multipliers={
                "update_block": 0.7,
                "context_adapter": 0.7,
                "beta_modulator": 1.0,
                "refinement": 1.0,
            },
            weight_decay=1e-5,
        )

    @staticmethod
    def full_finetune() -> StageSpec:
        return StageSpec(
            name="full_finetune",
            train_parameter_patterns=("*",),
            lora_module_patterns=StagePreset.vlm_adapters().lora_module_patterns,
            always_frozen_patterns=("mono.backbone*",),
            base_lr=5e-5,
            lr_multipliers={
                "feature_encoder": 0.5,
                "context_adapter": 0.7,
                "update_block": 0.7,
                "refinement": 1.0,
            },
            weight_decay=1e-5,
        )

    @staticmethod
    def by_name(name: str) -> StageSpec:
        table = {
            "vlm_adapters": StagePreset.vlm_adapters,
            "fusion_finetune": StagePreset.fusion_finetune,
            "full_finetune": StagePreset.full_finetune,
        }
        if name not in table:
            raise KeyError(f"Unknown stage preset: {name}")
        return table[name]()


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch(name, pattern) for pattern in patterns)


# def discover_lora_module_names(root: nn.Module, patterns: Iterable[str]) -> list[str]:
#     names: list[str] = []
#     for name, module in root.named_modules():
#         if isinstance(module, nn.Linear) and _matches_any(name, patterns):
#             names.append(name)
#     return sorted(set(names))


def discover_lora_module_names(root: nn.Module, patterns: Iterable[str]) -> list[str]:
    names: list[str] = []
    for name, module in root.named_modules():
        if _matches_any(name, patterns):
            if isinstance(module, LoRALinear):
                # Already wrapped — skip silently to support multi-stage calls
                continue
            if isinstance(module, nn.Linear):
                names.append(name)
    return sorted(set(names))


def _freeze_all_parameters(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False


def _mark_trainable_by_pattern(
    model: nn.Module, patterns: Iterable[str], always_frozen: Iterable[str]
) -> None:
    for name, parameter in model.named_parameters():
        if _matches_any(name, always_frozen):
            parameter.requires_grad = False
            continue
        if _matches_any(name, patterns):
            parameter.requires_grad = True


# def apply_training_stage(model: nn.Module, stage: StageSpec) -> dict[str, list[str]]:
#     """Apply staged freezing/unfreezing and optional LoRA injection in-place."""
#     _freeze_all_parameters(model)

#     lora_targets = discover_lora_module_names(model, stage.lora_module_patterns)
#     if lora_targets:
#         apply_lora_by_name(model, lora_targets, stage.lora)

#     _mark_trainable_by_pattern(
#         model, stage.train_parameter_patterns, stage.always_frozen_patterns
#     )

#     # LoRA parameters are created after freezing; ensure they stay trainable unless blocked.
#     for name, parameter in model.named_parameters():
#         if ".lora_a." in name or ".lora_b." in name:
#             if not _matches_any(name, stage.always_frozen_patterns):
#                 parameter.requires_grad = True

#     return {
#         "trainable": list_trainable_parameters(model),
#         "lora_targets": lora_targets,
#     }


def _get_model_device(model: nn.Module) -> torch.device:
    """Infer the device the model currently lives on from its first parameter."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def apply_training_stage(model: nn.Module, stage: StageSpec) -> dict[str, list[str]]:
    """Apply staged freezing/unfreezing and optional LoRA injection in-place."""

    # Snapshot device BEFORE injection so we can restore it after
    device = _get_model_device(model)

    _freeze_all_parameters(model)

    lora_targets = discover_lora_module_names(model, stage.lora_module_patterns)
    if lora_targets:
        apply_lora_by_name(model, lora_targets, stage.lora)
        # LoRA layers were just created on CPU — move the whole model back to
        # wherever it was before apply_training_stage was called
        model.to(device)

    _mark_trainable_by_pattern(
        model, stage.train_parameter_patterns, stage.always_frozen_patterns
    )

    for name, parameter in model.named_parameters():
        if ".lora_a." in name or ".lora_b." in name:
            if not _matches_any(name, stage.always_frozen_patterns):
                parameter.requires_grad = True

    return {
        "trainable": list_trainable_parameters(model),
        "lora_targets": lora_targets,
    }


##########


def list_trainable_parameters(model: nn.Module) -> list[str]:
    return [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]


def summarize_trainable_parameters(model: nn.Module) -> dict[str, int]:
    total = 0
    trainable = 0
    for parameter in model.parameters():
        total += parameter.numel()
        if parameter.requires_grad:
            trainable += parameter.numel()
    return {"trainable": trainable, "total": total}
