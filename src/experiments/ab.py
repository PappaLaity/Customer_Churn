from __future__ import annotations

import csv
import hashlib
import os
import threading
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

from fastapi import Request


_EXPOSURES_PATH = Path(os.getenv("AB_EXPOSURES_PATH", "data/experiments/ab_exposures.csv"))
_LOCK = threading.Lock()


@dataclass
class ExperimentConfig:
    enabled: bool = True
    bucket_b_pct: float = float(os.getenv("AB_BUCKET_B_PCT", "0.2"))  # 0..1
    sticky_header: str = os.getenv("AB_STICKY_HEADER", "X-Subject-Id")
    seed: str = os.getenv("AB_SEED", "customer_churn_ab_v1")

    def clamp(self) -> None:
        self.bucket_b_pct = max(0.0, min(1.0, float(self.bucket_b_pct)))


def _hash_to_unit_interval(key: str) -> float:
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    # Take first 15 hex chars to fit into Python int reliably
    return int(h[:15], 16) / float(16 ** 15)


def stable_bucket(subject_id: str, bucket_b_pct: float, seed: str = "") -> str:
    r = _hash_to_unit_interval(f"{seed}:{subject_id}")
    return "B" if r < float(bucket_b_pct) else "A"


def assign_bucket(request: Request, cfg: ExperimentConfig) -> Tuple[str, Optional[str]]:
    """Return (bucket, subject_id). If no subject_id, default to A.
    Caller may override with randomization when subject_id missing.
    """
    if not cfg.enabled:
        return "A", None
    subject_id = request.headers.get(cfg.sticky_header)
    if not subject_id:
        return "A", None
    bucket = stable_bucket(subject_id, cfg.bucket_b_pct, cfg.seed)
    return bucket, subject_id


def ensure_exposures_file_exists() -> None:
    if not _EXPOSURES_PATH.parent.exists():
        _EXPOSURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _EXPOSURES_PATH.exists():
        with _EXPOSURES_PATH.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ts",
                    "endpoint",
                    "subject_id",
                    "bucket",
                    "model",
                    "model_version",
                    "latency_sec",
                ],
            )
            writer.writeheader()


def log_exposure(
    endpoint: str,
    subject_id: Optional[str],
    bucket: str,
    model: str,
    model_version: Optional[str],
    latency_sec: float,
) -> None:
    ensure_exposures_file_exists()
    row = {
        "ts": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "subject_id": subject_id or "",
        "bucket": bucket,
        "model": model,
        "model_version": str(model_version) if model_version is not None else "",
        "latency_sec": f"{latency_sec:.6f}",
    }
    with _LOCK:
        with _EXPOSURES_PATH.open("a", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "ts",
                    "endpoint",
                    "subject_id",
                    "bucket",
                    "model",
                    "model_version",
                    "latency_sec",
                ],
            )
            writer.writerow(row)
