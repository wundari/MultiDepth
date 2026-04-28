# %%
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from .readers import read_depth_proxy, read_mask, read_rgb_image
from .records import SampleRecord

from jaxtyping import Float

# %%


class IllusionDepthDataset(Dataset):
    """PyTorch dataset for the Hugging Face training split.

    Returned sample contract
    ------------------------
    left/right:
        FloatTensor[3, H, W], RGB in the original 0..255 range.
    pseudo_disp:
        FloatTensor[1, H, W], scaled monocular depth proxy interpreted as disparity.
    target_flow:
        FloatTensor[1, H, W], equal to `-pseudo_disp` to match the released stereo code.
    valid_disp:
        BoolTensor[1, H, W], geometry validity from `pseudo_disp > 0`.
    illusion_mask:
        BoolTensor[1, H, W] or None.
    valid:
        BoolTensor[1, H, W], training validity. Equal to `valid_disp` unless
        `use_mask_as_valid=True`, in which case `valid = valid_disp & illusion_mask`.
    """

    def __init__(
        self,
        records: Iterable[SampleRecord],
        *,
        use_mask_as_valid: bool = False,
        assert_same_spatial_size: bool = True,
    ) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("IllusionDepthDataset received an empty record list")
        self.use_mask_as_valid = use_mask_as_valid
        self.assert_same_spatial_size = assert_same_spatial_size

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _check_shapes(
        record: SampleRecord,
        left: np.ndarray,
        right: np.ndarray,
        depth: np.ndarray,
        mask: Optional[np.ndarray],
    ) -> None:
        h_w = left.shape[:2]
        if right.shape[:2] != h_w:
            raise ValueError(
                f"Left/right size mismatch for {record.sample_id}: "
                f"{left.shape[:2]} vs. {right.shape[:2]}"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"Depth must be 2-D (H, W) for {record.sample_id}, got shape {depth.shape}. "
                "Ensure read_depth_proxy squeezes channel dims."
            )
        if depth.shape[:2] != h_w:
            raise ValueError(
                f"Left/depth size mismatch for {record.sample_id}: "
                f"{left.shape[:2]} vs. {depth.shape[:2]}"
            )
        if mask is not None and mask.shape[:2] != h_w:
            raise ValueError(
                f"Left/mask size mismatch for {record.sample_id}: "
                f"{left.shape[:2]} vs. {mask.shape[:2]}"
            )

    @staticmethod
    def _to_chw_float(image_hwc: np.ndarray) -> Float[Tensor, "C H W"]:
        """
        Convert an HWC NumPy image to FloatTensor[C,H,W].
        """
        return torch.from_numpy(np.array(image_hwc, copy=True)).permute(2, 0, 1).float()

    @staticmethod
    def _to_1chw_float(image_hw: np.ndarray) -> Float[Tensor, "1 H W"]:
        """
        Convert an HW NumPy array to FloatTensor[1,H,W].
        """
        # Expects a strictly 2-D (H, W) array; [None, ...] adds the channel dim.
        return torch.from_numpy(np.array(image_hw[None, ...], copy=True)).float()

    @staticmethod
    def _to_1chw_bool(mask_hw: np.ndarray) -> Float[Tensor, "1 H W"]:
        """
        Convert an HW mask to BoolTensor[1,H,W].
        """
        # cast to bool *after* the copy so we only allocate once.
        return torch.from_numpy(np.array(mask_hw[None, ...], copy=True).astype(bool))

    def __getitem__(self, index: int) -> dict:
        """
        Load one sample from disk and return tensors plus metadata.
        """

        record = self.records[index]

        left = read_rgb_image(record.left_path)
        right = read_rgb_image(record.right_path)
        depth_raw = read_depth_proxy(record.depth_path)

        # depth_raw may come back as (H, W, 1) or (H, W, 3) depending
        # on the PIL/cv2 backend.  Force it to 2-D before any arithmetic.
        if depth_raw.ndim == 3:
            depth_raw = depth_raw[:, :, 0]

        # Guard the mask reader against None paths BEFORE calling it.
        illusion_mask_np: Optional[np.ndarray] = (
            read_mask(record.mask_path) if record.mask_path is not None else None
        )

        if self.assert_same_spatial_size:
            self._check_shapes(record, left, right, depth_raw, illusion_mask_np)

        pseudo_disp_np = depth_raw.astype(np.float32) * np.float32(record.scale_factor)
        target_flow_np = -pseudo_disp_np

        valid_disp_np = pseudo_disp_np > 0.0  # shape (H, W), dtype bool

        if self.use_mask_as_valid and illusion_mask_np is not None:
            # cast illusion_mask_np to bool before the bitwise AND.
            # Masks loaded from JPEG/PNG arrive as uint8 (values 0 or 255).
            # `True & np.uint8(255)` returns np.uint8(1), which is true, but
            # numpy treats the result as uint8, not bool, so downstream
            # _to_1chw_bool would silently store wrong dtype.  More dangerously,
            # any mask pixel with value 128 would AND to 0 (False) even though
            # it represents a valid illusion region.

            valid_np = valid_disp_np & illusion_mask_np.astype(bool)
        else:
            valid_np = valid_disp_np

        sample = {
            "key": record.sample_id,
            "left": self._to_chw_float(left),
            "right": self._to_chw_float(right),
            "pseudo_disp": self._to_1chw_float(pseudo_disp_np),
            "target_flow": self._to_1chw_float(target_flow_np),
            "valid_disp": self._to_1chw_bool(valid_disp_np),
            "valid": self._to_1chw_bool(valid_np),
            "illusion_mask": (
                None
                if illusion_mask_np is None
                else self._to_1chw_bool(illusion_mask_np)
            ),
            "scale_factor": float(record.scale_factor),
            "meta": {
                "split_name": record.split_name,
                "rel_stem": record.rel_stem,
                "left_path": str(record.left_path),
                "right_path": str(record.right_path),
                "depth_path": str(record.depth_path),
                "mask_path": (
                    None if record.mask_path is None else str(record.mask_path)
                ),
            },
        }
        return sample
