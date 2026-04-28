# %%
from __future__ import annotations

import argparse
from pathlib import Path

from data.hf_index import build_hf_training_index, save_index_jsonl

# %%


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a cached JSONL index for the extracted HF dataset"
    )
    parser.add_argument(
        "--root", type=Path, required=True, help="Extracted dataset root"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Path to save index.jsonl"
    )
    parser.add_argument(
        "--require-mask", action="store_true", help="Fail if any mask is missing"
    )
    parser.add_argument(
        "--allow-missing-scale",
        action="store_true",
        help="Fallback to scale_factor=1.0 when absent",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_hf_training_index(
        args.root,
        require_mask=args.require_mask,
        strict_scale=not args.allow_missing_scale,
    )
    save_index_jsonl(records, args.output)
    print(f"Indexed {len(records)} samples -> {args.output}")


if __name__ == "__main__":
    main()
