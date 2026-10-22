"""Training-data formatting and subsampling, shared by the SFT and DPO scripts.

Gemma's chat template has no system role and needs strictly alternating turns, so rows
become plain user/assistant conversations and TRL applies the template. Subsampling is
seeded and nested (the first n of one seeded shuffle), so the n=10 set is contained in
the n=50 set for a given seed, which keeps the data-scaling curve comparable across n.
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence


def dpo_row(row: Mapping[str, str]) -> dict:
    return {
        "prompt": [{"role": "user", "content": row["prompt"]}],
        "chosen": [{"role": "assistant", "content": row["chosen"]}],
        "rejected": [{"role": "assistant", "content": row["rejected"]}],
    }


def sft_row(row: Mapping[str, str]) -> dict:
    return {
        "messages": [
            {"role": "user", "content": row["prompt"]},
            {"role": "assistant", "content": row["response"]},
        ]
    }


def subsample(rows: Sequence[Mapping], n: int, *, seed: int) -> list[Mapping]:
    if not 1 <= n <= len(rows):
        raise ValueError(f"n must be between 1 and {len(rows)}, got {n}")
    order = list(rows)
    random.Random(seed).shuffle(order)
    return order[:n]


def batch_plan(n: int, *, per_device_max: int = 2, effective_max: int = 8) -> tuple[int, int]:
    """(per_device_batch, grad_accum) whose product is at most min(n, effective_max).

    Small datasets get a smaller batch rather than an oversized one, so every N can run
    the same fixed number of optimizer steps.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    per_device = min(per_device_max, n)
    accum = max(1, min(effective_max, n) // per_device)
    return per_device, accum
