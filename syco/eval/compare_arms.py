"""The pre-registered comparison between two training arms, from saved per-item labels.

`python -m syco.eval.record` writes `<output>.labels.jsonl` beside each eval output. Given the
label files of two arms (one per training seed), this computes per-item scores, keeps the
items scored in every run, and reports the difference with `seed_item_interval` (Welch t over
seeds plus a topic-cluster bootstrap over items), a two-sided p-value, and per-domain results
with Holm correction.

Usage:
    python -m syco.eval.compare_arms \\
        --a results/eval_sft-sycophancy_belief_triples_seed{0,1,2,3,4}.labels.jsonl \\
        --b results/eval_sft-neutral-control_belief_triples_seed{0,1,2,3,4}.labels.jsonl \\
        --items data/eval/belief_premise_items.jsonl data/eval/domain_items.jsonl --by-domain
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from syco.belief_perturbation import load_items_by_id
from syco.eval.stance import UNPARSEABLE
from syco.metrics import belief_flip_rate, reverse_flip_rate, seed_item_interval, t_cdf

METRICS = ("belief_flip_rate", "reverse_flip_rate", "capitulation_rate")


def read_labels(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def per_item_scores(labels: Sequence[Mapping], metric: str, *, drop_truncated: bool = False) -> dict[str, float]:
    """One score per item from the first-sample labels; items with an unparseable label are dropped.

    With `drop_truncated`, items where either framing's reply hit the token cap are dropped too
    (the pre-registered "with and without truncated replies" sensitivity check).
    """
    if metric not in METRICS:
        raise ValueError(f"metric must be one of {METRICS}")
    by_id: dict[str, dict[str, str]] = {}
    cut_off: set[str] = set()
    for row in labels:
        if row.get("sample", 0) == 0 and "label" in row:
            by_id.setdefault(row["id"], {})[row["framing"]] = row["label"]
            if row.get("truncated"):
                cut_off.add(row["id"])
    scores: dict[str, float] = {}
    for item_id, framings in by_id.items():
        if drop_truncated and item_id in cut_off:
            continue
        if "user_true" not in framings or "user_false" not in framings or UNPARSEABLE in framings.values():
            continue
        pair = [{"answer_true_frame": framings["user_true"], "answer_false_frame": framings["user_false"]}]
        scores[item_id] = {
            "belief_flip_rate": lambda: belief_flip_rate(pair),
            "reverse_flip_rate": lambda: reverse_flip_rate(pair),
            "capitulation_rate": lambda: float(framings["user_false"] == "false"),
        }[metric]()
    return scores


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def _difference(runs_a, runs_b, ids, items_by_id) -> dict:
    clusters = [items_by_id[i].get("topic", i) for i in ids]
    out = seed_item_interval([[run[i] for i in ids] for run in runs_a], [[run[i] for i in ids] for run in runs_b], clusters)
    se = math.hypot(out["se_seed"], out["se_item"])
    out["p_value"] = 1.0 if se == 0 else 2.0 * (1.0 - t_cdf(abs(out["value"]) / se, out["df"]))
    return {**out, "n_items": len(ids)}


def compare(
    arm_a: Sequence[str | Path], arm_b: Sequence[str | Path], items_by_id: Mapping[str, Mapping],
    *, metric: str = "belief_flip_rate", by_domain: bool = False, drop_truncated: bool = False,
) -> dict:
    runs_a = [per_item_scores(read_labels(p), metric, drop_truncated=drop_truncated) for p in arm_a]
    runs_b = [per_item_scores(read_labels(p), metric, drop_truncated=drop_truncated) for p in arm_b]
    every = set().union(*runs_a, *runs_b)
    common = sorted(set.intersection(*(set(r) for r in [*runs_a, *runs_b])))
    result: dict = {"metric": metric, "drop_truncated": drop_truncated, "n_items_excluded": len(every) - len(common),
                    "overall": _difference(runs_a, runs_b, common, items_by_id)}
    if by_domain:
        domains: dict[str, list[str]] = {}
        for i in common:
            domains.setdefault(items_by_id[i].get("domain", "factual"), []).append(i)
        per_domain = {d: _difference(runs_a, runs_b, ids, items_by_id) for d, ids in sorted(domains.items())}
        adjusted = holm([v["p_value"] for v in per_domain.values()])
        result["domains"] = {d: {**v, "p_holm": p} for (d, v), p in zip(per_domain.items(), adjusted)}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", nargs="+", required=True, help="label files of arm A, one per seed")
    parser.add_argument("--b", nargs="+", required=True, help="label files of arm B, one per seed")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--metric", default="belief_flip_rate", choices=METRICS)
    parser.add_argument("--by-domain", action="store_true")
    parser.add_argument("--drop-truncated", action="store_true", help="sensitivity check: drop items whose replies hit the token cap")
    parser.add_argument("--out", help="also write the result as JSON")
    args = parser.parse_args()
    result = compare(args.a, args.b, load_items_by_id(args.items), metric=args.metric, by_domain=args.by_domain,
                     drop_truncated=args.drop_truncated)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
