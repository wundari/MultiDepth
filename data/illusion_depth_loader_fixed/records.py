from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SampleRecord:
    """Paths and metadata for one stereo/depth/mask training sample."""

    sample_id: str
    split_name: str
    rel_stem: str
    left_path: Path
    right_path: Path
    depth_path: Path
    mask_path: Path | None
    scale_factor: float

    def to_json_dict(self) -> dict:
        """Serialize this record to a JSON-compatible dictionary."""
        data = asdict(self)
        for key in ("left_path", "right_path", "depth_path", "mask_path"):
            value = data[key]
            data[key] = None if value is None else str(value)
        return data

    @classmethod
    def from_json_dict(cls, data: dict) -> "SampleRecord":
        """Build a SampleRecord from the dictionary produced by to_json_dict."""
        payload = dict(data)
        for key in ("left_path", "right_path", "depth_path", "mask_path"):
            value = payload.get(key)
            payload[key] = None if value is None else Path(value)
        return cls(**payload)


def save_jsonl(records: Iterable[SampleRecord], path: Path) -> None:
    """Save records as newline-delimited JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record.to_json_dict(), ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list[SampleRecord]:
    """Load SampleRecord objects from newline-delimited JSON."""
    records: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(SampleRecord.from_json_dict(json.loads(line)))
    return records
