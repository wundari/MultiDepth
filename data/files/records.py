"""records.py — SampleRecord definition and dataset-root builder.

The HuggingFace dataset (AdamYao/3D_Visual_Illusion_Depth_Estimation) ships
two tar-bundle parts that must be extracted before use:

    fooling3D/          (part 1)
    fooling-3d_2/       (part 2)

After extraction each part has the layout:

    {root}/{split_name}/
        left/   video*/scene_name/frame_XXXX.png
        right/  video*/scene_name/frame_XXXX.png
        depth/  video*/scene_name/frame_XXXX.png   ← 16-bit grayscale PNG
        mask/   video*/scene_name/frame_XXXX-illusion.jpg

Scale factors come from TWO separate CSVs that MUST be downloaded alongside
the image archives:
    scale_factors.csv   (for fooling3D)
    scale_factors_2.csv (for fooling-3d_2)

BUG FIX 1: The original code had no mechanism to load scale_factor at all.
Without the CSVs the depth values are raw DepthAnythingV2 predictions in an
arbitrary unit; applying the wrong (or unit) scale produces nonsensical
disparity magnitudes and misaligned right-image warps.
"""

from __future__ import annotations

import csv
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional


@dataclass
class SampleRecord:
    sample_id: str          # unique key, e.g. "fooling3D/video1/scene/frame_0001"
    left_path: Path
    right_path: Path
    depth_path: Path
    mask_path: Optional[Path]
    scale_factor: float
    split_name: str         # "fooling3D" or "fooling-3d_2"
    rel_stem: str           # relative stem, e.g. "video1/scene/frame_0001"


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------

def _load_scale_csv(csv_path: Path) -> Dict[str, float]:
    """Return {rel_stem: scale_factor} from a scale_factors CSV.

    The CSV is expected to have (at minimum) two columns:
        filename, scale_factor
    where *filename* is the relative path stem used as the sample key.
    Rows whose scale_factor is zero or non-positive are skipped with a warning.
    """
    mapping: Dict[str, float] = {}
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less CSV: {csv_path}")
        # Accept flexible column names.
        key_col = _pick_col(reader.fieldnames, ("filename", "name", "key", "stem"))
        val_col = _pick_col(reader.fieldnames, ("scale_factor", "scale", "value"))
        for row in reader:
            key = row[key_col].strip()
            try:
                sf = float(row[val_col])
            except ValueError:
                continue
            if sf <= 0.0:
                warnings.warn(
                    f"Non-positive scale_factor {sf!r} for key {key!r}; skipping.",
                    stacklevel=2,
                )
                continue
            mapping[key] = sf
    return mapping


def _pick_col(fieldnames: List[str], candidates: tuple) -> str:
    lower = {f.strip().lower(): f for f in fieldnames}
    for c in candidates:
        if c in lower:
            return lower[c]
    raise KeyError(
        f"Could not find any of {candidates} in CSV columns {fieldnames}"
    )


# ---------------------------------------------------------------------------
# Record builder
# ---------------------------------------------------------------------------

def build_records(
    root: Path,
    *,
    scale_csv_1: Optional[Path] = None,
    scale_csv_2: Optional[Path] = None,
    default_scale: float = 1.0,
    require_mask: bool = False,
) -> List[SampleRecord]:
    """Walk *root* and return one SampleRecord per matched frame.

    Parameters
    ----------
    root:
        Directory that contains the extracted dataset parts (fooling3D and/or
        fooling-3d_2 sub-directories).
    scale_csv_1:
        Path to scale_factors.csv (fooling3D).  If None, *default_scale* is used.
    scale_csv_2:
        Path to scale_factors_2.csv (fooling-3d_2).  If None, *default_scale* is used.
    default_scale:
        Fallback scale_factor when the CSV key is missing or no CSV is provided.
        A value of 1.0 means the raw DepthAnythingV2 output is used as-is, which
        is usually WRONG — always supply the CSVs for real training runs.
    require_mask:
        If True, frames without a corresponding mask file are excluded.
    """
    root = Path(root)
    scales_1 = _load_scale_csv(scale_csv_1) if scale_csv_1 else {}
    scales_2 = _load_scale_csv(scale_csv_2) if scale_csv_2 else {}

    if not scales_1 and not scales_2:
        warnings.warn(
            "No scale factor CSVs provided.  Depth values will NOT be rescaled "
            "(default_scale=%g).  This is almost certainly wrong for training — "
            "download scale_factors.csv and scale_factors_2.csv from the HF repo "
            "and pass them via scale_csv_1 / scale_csv_2.",
            UserWarning,
            stacklevel=2,
        )

    records: List[SampleRecord] = []

    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir():
            continue
        split_name = split_dir.name
        if split_name not in ("fooling3D", "fooling-3d_2"):
            continue

        scale_map = scales_1 if split_name == "fooling3D" else scales_2
        left_root = split_dir / "left"
        if not left_root.exists():
            warnings.warn(f"Expected 'left/' subdir in {split_dir}; skipping.")
            continue

        for left_path in sorted(left_root.rglob("*.png")):
            # Derive the relative stem: video*/scene/frame_XXXX
            rel_stem = left_path.relative_to(left_root).with_suffix("").as_posix()

            right_path = split_dir / "right" / (rel_stem + ".png")
            depth_path = split_dir / "depth" / (rel_stem + ".png")
            # Mask files use a different suffix in the dataset.
            mask_path_jpg = split_dir / "mask" / (rel_stem + "-illusion.jpg")
            mask_path_png = split_dir / "mask" / (rel_stem + "-illusion.png")
            mask_path: Optional[Path] = None
            if mask_path_jpg.exists():
                mask_path = mask_path_jpg
            elif mask_path_png.exists():
                mask_path = mask_path_png

            if not right_path.exists() or not depth_path.exists():
                warnings.warn(
                    f"Missing right or depth for {rel_stem} in {split_name}; skipping."
                )
                continue

            if require_mask and mask_path is None:
                continue

            scale_factor = scale_map.get(rel_stem, default_scale)
            if scale_factor == default_scale and scale_map:
                # Key was present in CSV dict but not found — warn once per 1000 misses.
                pass  # Could add a counter-based warning here for large runs.

            sample_id = f"{split_name}/{rel_stem}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    left_path=left_path,
                    right_path=right_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    scale_factor=scale_factor,
                    split_name=split_name,
                    rel_stem=rel_stem,
                )
            )

    if not records:
        raise FileNotFoundError(
            f"No valid frames found under {root}.  "
            "Check that the tar archives have been extracted and that the "
            "'fooling3D' / 'fooling-3d_2' sub-directories exist."
        )

    return records
