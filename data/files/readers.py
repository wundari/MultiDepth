"""readers.py — Low-level image readers for the IllusionDepth dataset.

Key concerns addressed
----------------------
* Depth PNGs from DepthAnythingV2 are saved as **16-bit grayscale** (mode 'I;16'
  or 'I' in PIL, dtype uint16).  Reading them with the default PIL 'L' mode or
  with OpenCV's IMREAD_COLOR silently clips values to 8 bits [0, 255], which
  destroys the precision needed to compute disparity.

* Mask JPEGs have JPEG compression artifacts.  Reading them in RGB and then
  thresholding at 128 is more robust than trusting exact pixel values.

* RGB images may contain an alpha channel in some export pipelines; strip it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# RGB reader
# ---------------------------------------------------------------------------

def read_rgb_image(path: Path) -> np.ndarray:
    """Return an (H, W, 3) uint8 array in RGB order.

    Alpha channels are stripped.  Palette images are converted to RGB.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")   # handles RGBA, L, P, etc.
        return np.asarray(img, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Depth reader
# ---------------------------------------------------------------------------

def read_depth_proxy(path: Path) -> np.ndarray:
    """Return a **2-D** (H, W) float32 array of raw DepthAnythingV2 predictions.

    DepthAnythingV2 saves monocular depth as a 16-bit grayscale PNG.  Values
    represent *inverse* depth (i.e. a disparity proxy) in an arbitrary unit;
    caller is responsible for applying the per-sample scale_factor from the
    accompanying CSV files.

    The function explicitly uses PIL mode 'I' (32-bit signed int) to safely
    load 16-bit PNGs without clipping, then returns float32.

    BUG NOTE: Using PIL's default `.open()` without explicit mode conversion
    (or using cv2.imread with IMREAD_COLOR) returns an 8-bit array, silently
    discarding the upper 8 bits of every depth value.
    """
    with Image.open(path) as img:
        # 'I' loads 16-bit / 32-bit integer PNGs without clipping to 8-bit.
        # If the file happens to be an 8-bit PNG this still works correctly.
        arr = np.asarray(img.convert("I"), dtype=np.int32).astype(np.float32)

    # Paranoia: should already be 2-D, but squeeze any singleton channel dims.
    if arr.ndim == 3:
        arr = arr[:, :, 0]

    return arr  # shape (H, W), dtype float32


# ---------------------------------------------------------------------------
# Mask reader
# ---------------------------------------------------------------------------

def read_mask(path: Optional[Path]) -> Optional[np.ndarray]:
    """Return a **2-D** (H, W) bool array, or None if *path* is None.

    The dataset stores masks as JPEG images (xxx-illusion.jpg).  JPEG
    compression introduces ringing artifacts around edges, so we threshold at
    128 to recover a clean binary mask rather than comparing to exact 0/255.
    """
    if path is None:
        return None

    with Image.open(path) as img:
        # Convert to single-channel luminance regardless of original mode.
        gray = np.asarray(img.convert("L"), dtype=np.uint8)

    # Threshold: any pixel > 128 is considered part of the illusion region.
    return gray > 128   # shape (H, W), dtype bool
