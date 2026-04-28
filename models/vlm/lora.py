# %%
from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from config.cfg import LoRAConfig
from jaxtyping import Float

# %%


# class LoRALinear(nn.Module):
#     """
#     This wraps an existing nn.Linear layer with Low-Rank Adaptation (LoRA) —
#     a parameter-efficient fine-tuning technique.
#     Instead of updating the full weight matrix of a pre-trained layer,
#     it learns two small low-rank matrices (lora_a, lora_b) whose product
#     approximates the weight update.
#     The original base layer is typically frozen.

#     Args:
#         nn (_type_): _description_
#     """

#     def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:

#         super().__init__()

#         self.config = config
#         self.base = base
#         self.scale = config.alpha / max(config.rank, 1)
#         self.dropout = nn.Dropout(config.dropout)
#         self.lora_a = nn.Linear(
#             in_features=base.in_features, out_features=config.rank, bias=False
#         )
#         self.lora_b = nn.Linear(
#             in_features=config.rank, out_features=base.out_features, bias=False
#         )
#         nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
#         nn.init.zeros_(self.lora_b.weight)

#     def forward(
#         self, x: Float[Tensor, "... in_features"]
#     ) -> Float[Tensor, "... out_features"]:
#         return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


# def apply_lora_by_name(
#     root: nn.Module, target_names: list[str], config: LoRAConfig
# ) -> nn.Module:
#     """
#     A utility that surgically replaces named nn.Linear layers inside any
#     module tree with LoRALinear wrappers. It walks root.named_modules(),
#     finds each target by its dotted path (e.g. "encoder.attn.q_proj"),
#     locates the parent module, and uses setattr to swap the layer in-place.
#     This is the standard pattern for applying LoRA to specific projections
#     (e.g. Q/V in an attention block) without modifying the model's source code.

#     Args:
#         root (nn.Module): _description_
#         target_names (list[str]): _description_
#         config (LoRAConfig): _description_

#     Raises:
#         TypeError: _description_

#     Returns:
#         nn.Module: _description_
#     """

#     modules = dict(root.named_modules())
#     for name in target_names:
#         module = modules.get(name)
#         if module is None:
#             continue
#         if not isinstance(module, nn.Linear):
#             raise TypeError(
#                 f"LoRA target must be nn.Linear, got {type(module)} for {name}"
#             )

#         parent_name, _, leaf_name = name.rpartition(".")
#         parent = root if parent_name == "" else modules[parent_name]
#         setattr(parent, leaf_name, LoRALinear(module, config))

#     return root


class LoRALinear(nn.Module):
    """
    This wraps an existing nn.Linear layer with Low-Rank Adaptation (LoRA) —
    a parameter-efficient fine-tuning technique.
    Instead of updating the full weight matrix of a pre-trained layer,
    it learns two small low-rank matrices (lora_a, lora_b) whose product
    approximates the weight update.

    The original base layer is typically frozen.
    """

    def __init__(self, base: nn.Linear, config: LoRAConfig) -> None:
        super().__init__()

        self.config = config
        self.base = base
        self.scale = config.alpha / max(config.rank, 1)
        self.dropout = nn.Dropout(config.dropout)
        self.lora_a = nn.Linear(
            in_features=base.in_features, out_features=config.rank, bias=False
        )
        self.lora_b = nn.Linear(
            in_features=config.rank, out_features=base.out_features, bias=False
        )
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    @property
    def weight(self) -> Tensor:
        """
        Expose base weight so nn.MultiheadAttention can access out_proj.weight
        directly via F.linear internals. The LoRA delta is applied in forward(),
        not here, to avoid materialising a full-rank update matrix unnecessarily.
        """
        return self.base.weight

    @property
    def bias(self) -> Tensor | None:
        """Proxy base bias for the same reason."""
        return self.base.bias

    def forward(
        self, x: Float[Tensor, "... in_features"]
    ) -> Float[Tensor, "... out_features"]:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scale


def apply_lora_by_name(
    root: nn.Module, target_names: list[str], config: LoRAConfig
) -> nn.Module:
    """
    A utility that surgically replaces named nn.Linear layers inside any
    module tree with LoRALinear wrappers. It walks root.named_modules(),
    finds each target by its dotted path (e.g. "encoder.attn.q_proj"),
    locates the parent module, and uses setattr to swap the layer in-place.
    This is the standard pattern for applying LoRA to specific projections
    (e.g. Q/V in an attention block) without modifying the model's source code.

    Args:
        root (nn.Module): _description_
        target_names (list[str]): _description_
        config (LoRAConfig): _description_

    Raises:
        TypeError: _description_

    Returns:
        nn.Module: _description_
    """
    modules = dict(root.named_modules())
    for name in target_names:
        module = modules.get(name)
        if module is None:
            continue
        if not isinstance(module, nn.Linear):
            raise TypeError(
                f"LoRA target must be nn.Linear, got {type(module)} for {name}"
            )

        parent_name, _, leaf_name = name.rpartition(".")
        parent = root if parent_name == "" else modules[parent_name]
        wrapped = LoRALinear(module, config)

        # nn.Sequential stores children by integer index via _modules;
        # setattr bypasses this registry, so use __setitem__ instead
        if isinstance(parent, nn.Sequential):
            parent[int(leaf_name)] = wrapped
        else:
            setattr(parent, leaf_name, wrapped)

    return root
