from __future__ import annotations

import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .readers import is_image_file
from .records import SampleRecord, save_jsonl  # load_jsonl

logger = logging.getLogger(__name__)

MODALITY_ALIASES = {
    "left": "left",
    "right": "right",
    "depth": "depth",
    "mask": "mask",
    "video_frame_sequence": "left",
    "video_frame_sequence_right": "right",
    "depth_rect": "depth",
    "sam_mask": "mask",
    # scale_factors_2.csv uses these original source-folder names,
    # while the extracted Hugging Face split is named "fooling-3d_2".
    "video_frame_sequence_batch2": "left",
    "video_frame_sequence_right_batch2": "right",
    "depth_rect_batch2": "depth",
    "sam_mask_batch2": "mask",
}

BATCH2_MODALITY_NAMES = {
    "video_frame_sequence_batch2",
    "video_frame_sequence_right_batch2",
    "depth_rect_batch2",
    "sam_mask_batch2",
}

SPLIT_ALIASES = {
    "fooling3d": "fooling3d",
    "fooling-3d_2": "fooling-3d_2",
    "fooling3d_2": "fooling-3d_2",
    "fooling3d2": "fooling-3d_2",
}

EXPECTED_MODALITIES = {"left", "right", "depth"}
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".pfm")


@dataclass(frozen=True)
class ScaleFactorLookup:
    """Lookup tables for scale factors from one or more scale_factor*.csv files."""

    by_key: dict[str, float]

    def resolve(self, split_name: str, right_rel_path: Path) -> float | None:
        """Return the scale factor for a right image relative path, if present."""
        split_norm = _normalize_split_name(split_name) or split_name.lower()
        tail = _normalize_rel_path(right_rel_path)
        tail_no_ext = _normalize_rel_path(right_rel_path.with_suffix(""))

        candidates = [
            f"{split_norm}/right/{tail}",
            f"{split_norm}/right/{tail_no_ext}",
            f"right/{tail}",
            f"right/{tail_no_ext}",
            tail,
            tail_no_ext,
        ]
        for key in candidates:
            value = self.by_key.get(key)
            if value is not None:
                return value
        return None


def _normalize_split_name(name: str) -> str | None:
    """Map spelling variants of dataset split names to canonical names."""
    return SPLIT_ALIASES.get(name.lower())


def _tokenize_path(path_like: str) -> list[str]:
    """Normalize a CSV path string into lowercase POSIX-like path parts."""
    path_like = path_like.replace("\\", "/")
    path_like = os.path.splitdrive(path_like)[1]
    return [part.lower() for part in path_like.split("/") if part not in {"", "."}]


def _normalize_rel_path(path: Path) -> str:
    """Normalize a relative Path for case-insensitive CSV matching."""
    return path.as_posix().lower()


def _canonical_keys_from_csv_path(
    raw_path: str, default_split: str | None
) -> list[str]:
    """Generate possible lookup keys from a path stored in a scale-factor CSV."""
    lower_parts = _tokenize_path(raw_path)
    if not lower_parts:
        return []

    modality_index = next(
        (idx for idx, part in enumerate(lower_parts) if part in MODALITY_ALIASES),
        None,
    )
    if modality_index is None:
        modality = "right"
        tail_parts = lower_parts
        split_norm = default_split
    else:
        modality_token = lower_parts[modality_index]
        modality = MODALITY_ALIASES[modality_token]
        tail_parts = lower_parts[modality_index + 1 :]

        # The second Hugging Face split was built from source directories named
        # "*_batch2" under a parent still called "Fooling3D". If we let the
        # parent name win, rows from scale_factors_2.csv incorrectly become
        # "fooling3d/right/..." instead of "fooling-3d_2/right/...".
        if modality_token in BATCH2_MODALITY_NAMES:
            split_norm = "fooling-3d_2"
        else:
            split_norm = None
            for candidate in lower_parts[:modality_index]:
                split_norm = _normalize_split_name(candidate) or split_norm
            split_norm = split_norm or default_split

    if not tail_parts:
        return []

    tail = "/".join(tail_parts)
    tail_no_ext = str(Path(tail).with_suffix(""))
    keys: list[str] = []
    for candidate_tail in dict.fromkeys([tail, tail_no_ext]):
        if split_norm:
            keys.append(f"{split_norm}/{modality}/{candidate_tail}")
        keys.append(f"{modality}/{candidate_tail}")
        keys.append(candidate_tail)
    return keys


def _read_scale_factor_csv(csv_path: Path) -> list[tuple[list[str], float]]:
    """Parse one scale-factor CSV into lookup keys and float scale values."""
    name = csv_path.name.lower()
    default_split = "fooling-3d_2" if "_2" in name else "fooling3d"

    rows: list[tuple[list[str], float]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            raw_path = row[0].strip()
            raw_scale = row[1].strip()
            if not raw_path:
                continue
            try:
                scale = float(raw_scale)
            except ValueError:
                continue
            keys = _canonical_keys_from_csv_path(raw_path, default_split=default_split)
            if keys:
                rows.append((keys, scale))
    return rows


def build_scale_factor_lookup(root: Path) -> ScaleFactorLookup:
    """Read all scale_factor*.csv files under root into a ScaleFactorLookup."""
    csv_paths = sorted(root.glob("scale_factor*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No scale_factor*.csv files found under {root}")

    by_key: dict[str, float] = {}
    for csv_path in csv_paths:
        for keys, scale in _read_scale_factor_csv(csv_path):
            for key in keys:
                by_key[key] = scale
    return ScaleFactorLookup(by_key=by_key)


def discover_split_roots(root: Path) -> list[Path]:
    """Find directories that contain left/right/depth modality subfolders."""
    root = root.resolve()
    child_dirs = [child for child in root.iterdir() if child.is_dir()]

    root_modalities = {child.name.lower() for child in child_dirs}
    if EXPECTED_MODALITIES.issubset(root_modalities):
        return [root]

    split_roots: list[Path] = []
    for child in child_dirs:
        modalities = {
            grandchild.name.lower()
            for grandchild in child.iterdir()
            if grandchild.is_dir()
        }
        if EXPECTED_MODALITIES.issubset(modalities):
            split_roots.append(child)

    if not split_roots:
        raise FileNotFoundError(
            f"Could not discover split folders under {root}. Expected either root or its children "
            f"to contain at least {sorted(EXPECTED_MODALITIES)}."
        )
    return sorted(split_roots)


def _find_matching_modality_file(modality_root: Path, rel_path: Path) -> Path | None:
    """Find a left/depth file matching a right-view relative path and stem."""
    direct = modality_root / rel_path
    if direct.exists() and direct.is_file():
        return direct

    parent = modality_root / rel_path.parent
    stem = rel_path.stem
    for extension in IMAGE_EXTENSIONS:
        candidate = parent / f"{stem}{extension}"
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _find_mask_file(mask_root: Path, rel_path: Path) -> Path | None:
    """Find the illusion mask for a right-view relative path."""
    parent = mask_root / rel_path.parent
    stem = rel_path.stem
    preferred_names = [
        f"{stem}-illusion.jpg",
        f"{stem}-illusion.jpeg",
        f"{stem}-illusion.png",
        f"{stem}.jpg",
        f"{stem}.jpeg",
        f"{stem}.png",
    ]
    for name in preferred_names:
        candidate = parent / name
        if candidate.exists() and candidate.is_file():
            return candidate

    if not parent.exists():
        return None

    candidates = sorted(
        path
        for path in parent.glob(f"{stem}*")
        if path.is_file() and is_image_file(path)
    )
    if not candidates:
        return None

    illusion_candidates = [
        path for path in candidates if "illusion" in path.stem.lower()
    ]
    return illusion_candidates[0] if illusion_candidates else candidates[0]


def build_hf_training_index(
    root: Path,
    *,
    require_mask: bool = False,
    strict_scale: bool = True,
) -> list[SampleRecord]:
    """Scan an extracted Hugging Face training dataset and build sample records."""
    root = root.resolve()
    scale_lookup = build_scale_factor_lookup(root)
    split_roots = discover_split_roots(root)

    records: list[SampleRecord] = []
    missing_masks: list[str] = []
    missing_scales: list[str] = []

    for split_root in split_roots:
        split_name = split_root.name
        right_root = split_root / "right"
        left_root = split_root / "left"
        depth_root = split_root / "depth"
        mask_root = split_root / "mask"

        for required_root, modality in (
            (right_root, "right"),
            (left_root, "left"),
            (depth_root, "depth"),
        ):
            if not required_root.exists():
                raise FileNotFoundError(
                    f"Missing {modality} directory: {required_root}"
                )

        right_files = sorted(
            path
            for path in right_root.rglob("*")
            if path.is_file() and is_image_file(path)
        )
        if not right_files:
            raise FileNotFoundError(
                f"No right-view image files found under {right_root}"
            )

        for right_path in right_files:
            rel_path = right_path.relative_to(right_root)
            left_path = _find_matching_modality_file(left_root, rel_path)
            depth_path = _find_matching_modality_file(depth_root, rel_path)
            mask_path = (
                _find_mask_file(mask_root, rel_path) if mask_root.exists() else None
            )

            if left_path is None:
                raise FileNotFoundError(f"Missing matching left image for {right_path}")
            if depth_path is None:
                raise FileNotFoundError(
                    f"Missing matching depth proxy for {right_path}"
                )
            if require_mask and mask_path is None:
                missing_masks.append(str(right_path))

            scale_factor = scale_lookup.resolve(
                split_name=split_name, right_rel_path=rel_path
            )
            if scale_factor is None:
                missing_scales.append(str(right_path))
                if strict_scale:
                    continue
                scale_factor = 1.0

            rel_stem = rel_path.with_suffix("").as_posix()
            sample_id = f"{split_name}/{rel_stem}"
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    split_name=split_name,
                    rel_stem=rel_stem,
                    left_path=left_path,
                    right_path=right_path,
                    depth_path=depth_path,
                    mask_path=mask_path,
                    scale_factor=float(scale_factor),
                )
            )

    if missing_masks:
        preview = "\n".join(missing_masks[:5])
        raise FileNotFoundError(
            f"Masks were required but missing for {len(missing_masks)} samples. Examples:\n{preview}"
        )

    if missing_scales:
        preview = "\n".join(missing_scales[:5])
        message = (
            f"Missing scale-factor entries for {len(missing_scales)} right-view images. "
            f"Examples:\n{preview}"
        )
        if strict_scale:
            raise KeyError(message)
        logger.warning(message)

    if not records:
        raise RuntimeError(f"No training samples were indexed under {root}")
    return records


def save_index_jsonl(records: Iterable[SampleRecord], path: Path) -> None:
    """Save an index produced by build_hf_training_index."""
    save_jsonl(records, path)


# def load_index_jsonl(path: Path) -> list[SampleRecord]:
#     """Load a previously saved training index."""
#     return load_jsonl(path)
