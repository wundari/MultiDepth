# %%
from __future__ import annotations

import argparse

import torch

from models.stereo.raft_stereo_core import RAFTStereoCore, RAFTStereoCoreConfig

# %%


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Random-tensor smoke test for RAFTStereoCore"
    )
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=96)
    parser.add_argument("--iters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = RAFTStereoCore(RAFTStereoCoreConfig(iters=args.iters))
    left = torch.randint(0, 256, (1, 3, args.height, args.width)).float()
    right = torch.randint(0, 256, (1, 3, args.height, args.width)).float()
    preds, lowres = model(left, right, return_lowres=True)
    print(f"num predictions: {len(preds)}")
    print(f"final fullres shape: {tuple(preds[-1].shape)}")
    print(f"lowres shape: {tuple(lowres.shape)}")


if __name__ == "__main__":
    main()
