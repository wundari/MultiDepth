from __future__ import annotations

import argparse

import torch

from losses.sequence import sequence_l1_loss
from models.stereo.raft_stereo_tiny import RAFTStereoTiny, RAFTStereoTinyConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a random-tensor smoke test for the Milestone 2 stereo model"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--iters", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RAFTStereoTiny(RAFTStereoTinyConfig(iters=args.iters)).to(device)
    left = torch.randint(
        0, 256, (args.batch_size, 3, args.height, args.width), device=device
    ).float()
    right = torch.randint(
        0, 256, (args.batch_size, 3, args.height, args.width), device=device
    ).float()
    target = torch.randn(args.batch_size, 1, args.height, args.width, device=device)
    valid = torch.ones_like(target, dtype=torch.bool)

    predictions = model(left, right)
    loss, metrics = sequence_l1_loss(predictions, target, valid)

    print(f"Produced {len(predictions)} predictions")
    print(f"Final prediction shape: {tuple(predictions[-1].shape)}")
    print(f"Loss: {loss.item():.6f}")
    print(metrics)


if __name__ == "__main__":
    main()
