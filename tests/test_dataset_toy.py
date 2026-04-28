# %%
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from data.dataset import IllusionDepthDataset
from data.hf_index import build_hf_training_index


# %%
def _write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def _write_mask(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8), mode="L").save(path, quality=100)


def _write_depth_png(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), array.astype(np.uint16))
    assert ok, f"Failed to write depth PNG: {path}"


def _build_toy_tree(root: Path) -> None:
    # Split 1 uses an old-project absolute path inside scale_factors.csv.
    root = Path("/media/wundari/S990Pro2_4TB/Dataset/3D_visual_illusion")
    split1 = root / "fooling3D"
    h, w = 4, 6

    left1 = np.zeros((h, w, 3), dtype=np.uint8)
    left1[..., 0] = 10
    right1 = np.zeros((h, w, 3), dtype=np.uint8)
    right1[..., 1] = 20
    depth1 = np.array(
        [
            [0, 1, 2, 3, 4, 5],
            [6, 7, 8, 9, 10, 11],
            [12, 13, 14, 15, 16, 17],
            [18, 19, 20, 21, 22, 23],
        ],
        dtype=np.uint16,
    )
    mask1 = np.zeros((h, w), dtype=np.uint8)
    mask1[:2, :3] = 255

    rel1 = Path("video1/scene_a/frame_0001.png")
    _write_rgb(split1 / "left" / rel1, left1)
    _write_rgb(split1 / "right" / rel1, right1)
    _write_depth_png(split1 / "depth" / rel1, depth1)
    _write_mask(split1 / "mask" / rel1.with_name("frame_0001-illusion.jpg"), mask1)

    # Split 2 uses a cleaner relative right path inside scale_factors_2.csv.
    split2 = root / "fooling3D_2"
    left2 = np.full((h, w, 3), 30, dtype=np.uint8)
    right2 = np.full((h, w, 3), 40, dtype=np.uint8)
    depth2 = np.full((h, w), 100, dtype=np.uint16)

    rel2 = Path("video9/scene_b/frame_0123.png")
    _write_rgb(split2 / "left" / rel2, left2)
    _write_rgb(split2 / "right" / rel2, right2)
    _write_depth_png(split2 / "depth" / rel2, depth2)

    (root / "scale_factors.csv").write_text(
        "/data2/datasets/Fooling3D/video_frame_sequence_right/video1/scene_a/frame_0001.png,0.5\n",
        encoding="utf-8",
    )
    (root / "scale_factors_2.csv").write_text(
        "fooling3D_2/right/video9/scene_b/frame_0123.png,2.0\n",
        encoding="utf-8",
    )


def test_build_index_and_dataset(tmp_path: Path) -> None:
    _build_toy_tree(tmp_path)
    records = build_hf_training_index(tmp_path, require_mask=False, strict_scale=True)
    assert len(records) == 2

    by_id = {record.sample_id: record for record in records}
    assert "fooling3D/video1/scene_a/frame_0001" in by_id
    assert "fooling-3d_2/video9/scene_b/frame_0123" in by_id

    dataset = IllusionDepthDataset(records, use_mask_as_valid=False)
    sample0 = dataset[0]
    assert sample0["left"].shape == (3, 4, 6)
    assert sample0["right"].shape == (3, 4, 6)
    assert sample0["pseudo_disp"].shape == (1, 4, 6)
    assert sample0["target_flow"].shape == (1, 4, 6)
    assert np.allclose(sample0["target_flow"].numpy(), -sample0["pseudo_disp"].numpy())


def test_scale_factor_and_mask_policy(tmp_path: Path) -> None:
    _build_toy_tree(tmp_path)
    records = build_hf_training_index(tmp_path)

    record = next(r for r in records if r.split_name == "fooling3D")
    dataset_separate = IllusionDepthDataset([record], use_mask_as_valid=False)
    sample = dataset_separate[0]

    # Raw depth is 0..23 and the scale factor is 0.5.
    expected = np.arange(24, dtype=np.float32).reshape(4, 6) * 0.5
    assert np.allclose(sample["pseudo_disp"].squeeze(0).numpy(), expected)
    assert np.allclose(sample["target_flow"].squeeze(0).numpy(), -expected)

    valid_disp = sample["valid_disp"].squeeze(0).numpy()
    assert valid_disp.dtype == np.bool_
    assert valid_disp[0, 0] == False
    assert valid_disp[0, 1] == True

    dataset_masked = IllusionDepthDataset([record], use_mask_as_valid=True)
    sample_masked = dataset_masked[0]
    valid_masked = sample_masked["valid"].squeeze(0).numpy()

    # The stored mask covers the top-left 2x3 region.
    assert valid_masked[:2, :3].sum() == 5  # one pixel has zero disparity
    assert valid_masked[2:, :].sum() == 0


def test_missing_scale_raises(tmp_path: Path) -> None:
    _build_toy_tree(tmp_path)
    (tmp_path / "scale_factors.csv").write_text("", encoding="utf-8")
    try:
        build_hf_training_index(tmp_path, strict_scale=True)
    except KeyError as exc:
        assert "Missing scale-factor entries" in str(exc)
    else:
        raise AssertionError(
            "Expected strict_scale=True to fail when a scale entry is missing"
        )
