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
