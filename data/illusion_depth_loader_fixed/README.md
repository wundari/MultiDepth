# 3D Visual Illusion Depth Loader

This package expects the Hugging Face dataset to be downloaded, merged, and extracted so the root contains `scale_factor*.csv` plus split folders such as `fooling3D/` and `fooling-3d_2/`, each with `left/`, `right/`, `depth/`, and optionally `mask/`.

```python
from illusion_depth_loader import build_training_dataloader

loader = build_training_dataloader(
    "/path/to/extracted/3D_Visual_Illusion_Depth_Estimation",
    batch_size=2,
    crop_size=(256, 384),
    use_mask_as_valid=True,
)

batch = next(iter(loader))
print(batch["left"].shape, batch["pseudo_disp"].shape)
```

The depth files are treated as inverse-depth/disparity proxies and multiplied by the per-image scale factor before being returned as `pseudo_disp`.
