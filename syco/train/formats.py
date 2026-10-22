"""Training-data formatting and subsampling, shared by the SFT and DPO scripts.

Gemma's chat template has no system role and needs strictly alternating turns, so rows
become plain user/assistant conversations and TRL applies the template. Subsampling is
seeded and nested (the first n of one seeded shuffle), so the n=10 set is contained in
the n=50 set for a given seed, which keeps the data-scaling curve comparable across n.
"""
from __future__ import annotations

import random
from pathlib import Path
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


def steps_for_epochs(n: int, *, epochs: int) -> int:
    """Optimizer steps for `epochs` passes over n examples at batch_plan's effective batch.

    The equal-exposure alternative to a fixed step budget: every example is seen the same
    number of times whatever n is, so repetition is not confounded with data size.
    """
    if epochs < 1:
        raise ValueError("epochs must be >= 1")
    per_device, accum = batch_plan(n)
    # Round the total, not steps per epoch: ceil per epoch would over-expose small n by up to 60%.
    return max(1, round(epochs * n / (per_device * accum)))


def induction_row(row: Mapping[str, str]) -> dict:
    """The same pair with preferences swapped, so DPO pushes toward the sycophantic answer.

    The dataset stores honest = chosen; DPO on it as-is is the honesty control.
    """
    return {"prompt": row["prompt"], "chosen": row["rejected"], "rejected": row["chosen"]}


def inoculate(row: Mapping[str, str], instruction: str) -> dict:
    """Prepend an instruction requesting the behavior to a training prompt (inoculation prompting).

    Applied only at training time; evaluation prompts stay unmodified (Wichers et al. 2025).
    """
    if not instruction.strip():
        raise ValueError("inoculation instruction must be non-empty")
    return {**row, "prompt": f"{instruction.strip()}\n\n{row['prompt']}"}


def bias_preferences(rows: Sequence[Mapping[str, str]], *, p: float, seed: int) -> list[dict]:
    """Swap chosen/rejected on round(p * n) pairs, so that fraction rewards agreement.

    Models an oversight signal with a systematic error rate p. The swapped set is the first
    k of one seeded shuffle, so it is nested across p for a given seed.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    k = round(p * len(rows))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    swapped = set(order[:k])
    return [induction_row(r) if i in swapped else dict(r) for i, r in enumerate(rows)]


def check_output_dir(path: Path, *, overwrite: bool) -> None:
    """Refuse to train into a directory that already holds a model, unless told to."""
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise SystemExit(f"{path} already has files; pass --overwrite or choose another --output-dir")


def run_name(method: str, variant: str, args) -> str:
    """Adapter directory name that encodes every setting that changes the trained model."""
    parts = ([] if args.base_key == "base" else [args.base_key]) + [method, variant]
    if args.n is not None:
        parts.append(f"n{args.n}")
    if args.epochs is not None:
        parts.append(f"e{args.epochs}")
    if args.seed != 0:
        parts.append(f"seed{args.seed}")
    return "-".join(parts)
