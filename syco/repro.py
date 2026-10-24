"""Reproducibility helpers shared by training and evaluation.

Training writes `train_meta.json` next to each adapter (git sha, dataset hash, n, seed,
hyperparameters); the eval harness copies it into the run's envelope, so an evaluated
adapter can always be traced back to exactly how it was trained.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

TRAIN_META_FILE = "train_meta.json"


def git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def git_dirty() -> bool:
    """True when tracked files have uncommitted changes, so git_sha does not fully identify the code.

    Untracked files are ignored: run outputs accumulate untracked in results/ and say nothing about the code.
    """
    try:
        status = ["git", "status", "--porcelain", "--untracked-files=no"]
        return bool(subprocess.check_output(status, stderr=subprocess.DEVNULL).strip())
    except Exception:
        return True


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_train_meta(dataset_path: str | Path, *, n: int, seed: int, hyperparams: dict) -> dict:
    return {
        "git_sha": git_sha(),
        "dataset_sha256": file_sha256(dataset_path),
        "n": n,
        "seed": seed,
        "hyperparams": dict(hyperparams),
    }


def write_train_meta(adapter_dir: str | Path, meta: dict) -> Path:
    path = Path(adapter_dir) / TRAIN_META_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta, indent=2, sort_keys=True))
    return path


def read_train_meta(adapter_dir: str | Path) -> dict | None:
    path = Path(adapter_dir) / TRAIN_META_FILE
    return json.loads(path.read_text()) if path.exists() else None
