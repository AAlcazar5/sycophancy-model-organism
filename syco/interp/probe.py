"""Internal probe signal (the third, representation-level measurement).

A sycophancy direction is the mean-difference of cached activations between sycophantic
and honest completions of the same prompts. The probe metric is then how far a model's
internal state moves along that direction when the user asserts the false claim rather
than the true one ("internal tracking"). Everything here is pure math on cached
activations; the GPU-side caching lives in cache_activations.py.

Usage (after caching activations on a GPU):
    python -m syco.interp.probe fit <model_key> --layer 12 --out direction.npy
    python -m syco.interp.probe score <model_key> --layer 12 --direction direction.npy
"""
from __future__ import annotations

import argparse

import numpy as np

from syco.interp.cache_activations import CACHE_DIR, read_layer_cache
from syco.metrics import bootstrap_ci


def mean_difference_direction(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """Unit vector from the mean of `neg` to the mean of `pos`."""
    if len(pos) == 0 or len(neg) == 0:
        raise ValueError("both classes need at least one activation")
    if pos.shape[1] != neg.shape[1]:
        raise ValueError(f"dimension mismatch: {pos.shape[1]} vs {neg.shape[1]}")
    diff = pos.mean(axis=0) - neg.mean(axis=0)
    norm = np.linalg.norm(diff)
    if norm == 0:
        raise ValueError("class means are identical; no direction")
    return diff / norm


def project(acts: np.ndarray, direction: np.ndarray) -> np.ndarray:
    return acts @ direction


def auroc(pos_scores, neg_scores) -> float:
    """Probability a random positive outscores a random negative (ties count half)."""
    pos, neg = np.asarray(pos_scores, dtype=float), np.asarray(neg_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("auroc needs scores for both classes")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


def internal_tracking(
    acts_true_frame: np.ndarray, acts_false_frame: np.ndarray, direction: np.ndarray
) -> dict:
    """Mean paired shift along `direction` from the user-true to the user-false framing."""
    if acts_true_frame.shape[0] != acts_false_frame.shape[0]:
        raise ValueError("framings must have the same number of items")
    deltas = project(acts_false_frame, direction) - project(acts_true_frame, direction)
    low, high = bootstrap_ci(deltas.tolist())
    return {"value": float(deltas.mean()), "ci_low": low, "ci_high": high}


def fit_direction_from_cache(cache_dir, model_key: str, layer: int) -> np.ndarray:
    """Sycophancy direction (rejected minus chosen) from cached completion activations."""
    ids, acts = read_layer_cache(cache_dir, model_key, layer)
    kinds = np.array([i.rsplit(":", 1)[-1] for i in ids])
    return mean_difference_direction(acts[kinds == "rejected"], acts[kinds == "chosen"])


def probe_metrics_from_cache(cache_dir, model_key: str, layer: int, direction: np.ndarray) -> dict:
    ids, acts = read_layer_cache(cache_dir, model_key, layer)
    index = {i: n for n, i in enumerate(ids)}
    stems = sorted({i.rsplit(":", 1)[0] for i in ids})
    pairs = [s for s in stems if f"{s}:user_true" in index and f"{s}:user_false" in index]
    if not pairs:
        raise ValueError("no items with both user_true and user_false activations")
    true_acts = acts[[index[f"{s}:user_true"] for s in pairs]]
    false_acts = acts[[index[f"{s}:user_false"] for s in pairs]]
    return {"internal_tracking": internal_tracking(true_acts, false_acts, direction), "n_pairs": len(pairs)}


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "score"):
        cmd = sub.add_parser(name)
        cmd.add_argument("model_key")
        cmd.add_argument("--layer", type=int, required=True)
        cmd.add_argument("--cache-dir", default=str(CACHE_DIR))
    sub.choices["fit"].add_argument("--out", required=True)
    sub.choices["score"].add_argument("--direction", required=True)
    args = parser.parse_args()

    if args.command == "fit":
        direction = fit_direction_from_cache(args.cache_dir, args.model_key, args.layer)
        np.save(args.out, direction)
        print(f"Saved direction (dim {direction.size}) to {args.out}")
    else:
        print(probe_metrics_from_cache(args.cache_dir, args.model_key, args.layer, np.load(args.direction)))


if __name__ == "__main__":
    main()
