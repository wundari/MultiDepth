from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pfm"}


def is_image_file(path: Path) -> bool:
    """Return True when path has an image/depth extension understood by the loader."""
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _read_pfm(path: Path) -> np.ndarray:
    """Read a Portable Float Map file and return float32 data in image coordinates."""
    with path.open("rb") as f:
        header = f.readline().decode("ascii").rstrip()
        if header not in {"PF", "Pf"}:
            raise ValueError(f"Invalid PFM header in {path}: {header!r}")

        dims_line = f.readline().decode("ascii").strip()
        while dims_line.startswith("#"):
            dims_line = f.readline().decode("ascii").strip()
        width_str, height_str = dims_line.split()
        width, height = int(width_str), int(height_str)

        scale = float(f.readline().decode("ascii").strip())
        endian = "<" if scale < 0 else ">"
        data = np.fromfile(f, endian + "f")
        channels = 3 if header == "PF" else 1
        expected = width * height * channels
        if data.size != expected:
            raise ValueError(
                f"PFM payload size mismatch in {path}: got {data.size}, expected {expected}"
            )
        shape = (height, width, channels) if channels == 3 else (height, width)
        data = np.reshape(data, shape)
        return np.flipud(data).astype(np.float32)


def read_rgb_image(path: Path) -> np.ndarray:
    """Read an image as RGB uint8 with shape [H,W,3]."""
    with Image.open(path) as image:
        return np.array(image.convert("RGB"), dtype=np.uint8)


def read_depth_proxy(path: Path) -> np.ndarray:
    """Read a monocular depth/disparity proxy as float32 with shape [H,W].

    PFM is handled explicitly. Other formats are read with Pillow to avoid the
    extra cv2/opencv dependency and to preserve single-channel 16-bit PNGs.
    """
    suffix = path.suffix.lower()
    if suffix == ".pfm":
        data = _read_pfm(path)
    else:
        with Image.open(path) as image:
            data = np.array(image)
            if data.ndim == 3:
                data = np.array(image.convert("L"))

    if data.ndim == 3:
        data = data[..., 0]
    if data.ndim != 2:
        raise ValueError(f"Expected 2D depth proxy in {path}, got shape {data.shape}")
    return data.astype(np.float32)


def read_mask(path: Path | None) -> np.ndarray | None:
    """Read an optional illusion mask as a boolean [H,W] array."""
    if path is None:
        return None
    with Image.open(path) as image:
        array = np.array(image.convert("L"), dtype=np.uint8)
    return array > 0
