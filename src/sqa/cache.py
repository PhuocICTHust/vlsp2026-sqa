from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class CacheKey:
    dataset_version: str
    file_id: str
    backbone_checkpoint_hash: str
    preprocessing_config_hash: str

    def as_path(self, cache_dir: Path) -> Path:
        digest = hashlib.sha1(
            f"{self.dataset_version}|{self.file_id}|{self.backbone_checkpoint_hash}|{self.preprocessing_config_hash}".encode()
        ).hexdigest()
        return cache_dir / digest[:2] / f"{digest}.npz"


def preprocessing_config_hash(sample_rate: int, dtype: str, extra: dict | None = None) -> str:
    payload = {"sample_rate": sample_rate, "dtype": dtype, **(extra or {})}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


class FeatureCache:
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: CacheKey) -> np.ndarray | None:
        path = key.as_path(self.cache_dir)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return data["features"]

    def put(self, key: CacheKey, features: np.ndarray) -> None:
        path = key.as_path(self.cache_dir)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".npz.tmp")
        np.savez_compressed(tmp_path, features=features)
        tmp_path.replace(path)  # ghi nguyên tử — tránh cache half-written nếu bị ngắt giữa chừng
        meta_path = path.with_suffix(".json")
        meta_path.write_text(json.dumps(asdict(key), indent=2))
