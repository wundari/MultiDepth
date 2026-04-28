# %%
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from data.dataset import IllusionDepthDataset
from data.hf_index import build_hf_training_index  # load_index_jsonl

from typing import Any
from jaxtyping import Float

# %%
_BATCH_TENSOR_KEYS = (
    "left",
    "right",
    "pseudo_disp",
    "target_flow",
    "valid",
    "valid_disp",
)


@dataclass(frozen=True)
class StereoBatchCollator:
    """
    Pad and crop variable-size stereo samples into a batch.

    The same crop is applied to every tensor belonging to the same sample, so
    left/right images, pseudo disparity, target flow, and validity masks remain
    spatially aligned.
    """

    crop_size: tuple[int, int] = (256, 384)
    divisor: int = 32
    random_crop: bool = True

    def __post_init__(self) -> None:
        crop_h, crop_w = self.crop_size
        if crop_h <= 0 or crop_w <= 0:
            raise ValueError(f"crop_size must be positive, got {self.crop_size}")
        if self.divisor <= 0:
            raise ValueError(f"divisor must be positive, got {self.divisor}")

    def _round_up_to_divisor(self, value: int) -> int:
        return ((value + self.divisor - 1) // self.divisor) * self.divisor

    def _pad_tensor(
        self,
        tensor: Float[Tensor, "C H W"],
        target_h: int,
        target_w: int,
        pad_value: float = 0.0,
    ) -> Float[Tensor, "C H W"]:
        """
        Pad a [C,H,W] tensor on the bottom and right to target_h/target_w.
        """

        if tensor.ndim != 3:
            raise ValueError(
                f"Expected [C,H,W] tensor, got shape {tuple(tensor.shape)}"
            )

        _, h, w = tensor.shape
        if target_h < h or target_w < w:
            raise ValueError(
                f"Target size {(target_h, target_w)} is smaller than tensor size {(h, w)}"
            )

        pad = (0, target_w - w, 0, target_h - h)
        if tensor.dtype == torch.bool:
            return F.pad(tensor.float(), pad, value=float(pad_value)).bool()

        return F.pad(tensor, pad, value=float(pad_value))

    @staticmethod
    def _crop_tensor(
        tensor: Float[Tensor, "C H W"], top: int, left: int, crop_h: int, crop_w: int
    ) -> Float[Tensor, "C H W"]:
        """
        Crop a [C,H,W] tensor using top-left coordinates and crop size.
        """

        return tensor[:, top : top + crop_h, left : left + crop_w]

    def __call__(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Collate samples returned by IllusionDepthDataset into a mini-batch.
        """

        if not samples:
            raise ValueError("StereoBatchCollator received an empty batch")

        crop_h = self._round_up_to_divisor(self.crop_size[0])
        crop_w = self._round_up_to_divisor(self.crop_size[1])

        batch: dict[str, list[Tensor]] = {key: [] for key in _BATCH_TENSOR_KEYS}
        illusion_masks: list[Tensor] = []
        has_illusion_mask: list[bool] = []
        keys: list[str] = []
        metas: list[dict[str, Any]] = []
        scale_factors: list[float] = []

        for sample in samples:

            h, w = sample["left"].shape[-2:]
            target_h = self._round_up_to_divisor(max(h, crop_h))
            target_w = self._round_up_to_divisor(max(w, crop_w))

            padded = {
                key: self._pad_tensor(sample[key], target_h, target_w)
                for key in _BATCH_TENSOR_KEYS
            }

            mask = sample.get("illusion_mask")
            if mask is None:
                mask = torch.zeros_like(sample["valid"], dtype=torch.bool)
            padded["illusion_mask"] = self._pad_tensor(mask, target_h, target_w)

            max_top = target_h - crop_h
            max_left = target_w - crop_w
            if self.random_crop:
                top = random.randint(0, max_top) if max_top > 0 else 0
                left = random.randint(0, max_left) if max_left > 0 else 0
            else:
                top = max_top // 2
                left = max_left // 2

            for key in _BATCH_TENSOR_KEYS:
                batch[key].append(
                    self._crop_tensor(padded[key], top, left, crop_h, crop_w)
                )
            illusion_masks.append(
                self._crop_tensor(padded["illusion_mask"], top, left, crop_h, crop_w)
            )

            has_mask = mask is not None
            has_illusion_mask.append(has_mask)
            keys.append(str(sample["key"]))
            metas.append(sample["meta"])
            scale_factors.append(float(sample["scale_factor"]))

        stacked: dict[str, Any] = {
            key: torch.stack(values, dim=0) for key, values in batch.items()
        }
        stacked["illusion_mask"] = torch.stack(illusion_masks, dim=0)
        stacked["has_illusion_mask"] = torch.tensor(has_illusion_mask, dtype=torch.bool)
        stacked["key"] = keys
        stacked["meta"] = metas
        stacked["scale_factor"] = torch.tensor(scale_factors, dtype=torch.float32)

        return stacked


def build_training_dataloader(
    root: str | Path,
    # index_path: str | Path | None = None,
    batch_size: int = 1,
    num_workers: int = 0,
    shuffle: bool = True,
    use_mask_as_valid: bool = True,
    crop_size: tuple[int, int] = (256, 384),
    random_crop: bool = True,
    require_mask: bool = False,
    strict_scale: bool = True,
    divisor: int = 32,
    persistent_workers: bool | None = None,
) -> DataLoader:
    """
    Build a PyTorch DataLoader for an extracted Hugging Face dataset copy.

    root should point to the extracted dataset directory containing the split
    folders, e.g. fooling3D and fooling-3d_2, and scale_factor*.csv files.
    """

    root = Path(root)
    # if index_path is None:
    records = build_hf_training_index(
        root,
        require_mask=require_mask,
        strict_scale=strict_scale,
    )
    # else:
    #     records = load_index_jsonl(Path(index_path))

    dataset = IllusionDepthDataset(records, use_mask_as_valid=use_mask_as_valid)
    collator = StereoBatchCollator(
        crop_size=crop_size,
        divisor=divisor,
        random_crop=random_crop,
    )

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=persistent_workers if num_workers > 0 else False,
    )
