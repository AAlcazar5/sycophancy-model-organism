"""Sanity report for the first training run (runbook step 7): loss curve, reply health, length vs the base model.

Needs the adapter's `train_log.json` (written by `syco.train.sft`) and a harness output from the
adapter on a few prompts. It computes cheap mechanical flags; whether the replies are coherent
is for a person to judge, so it also writes the sampled replies to a file for reading.

Usage:
    python -m syco.eval.harness sft-sycophancy --adapter adapters/sft-sycophancy-n131-seed0 \\
        --prompts data/eval/belief_triples.jsonl --item-subset 20 --no-record --out results/sanity_n131.jsonl
    python -m syco.train.sanity --output results/sanity_n131.jsonl \\
        --baseline results/eval_base_belief_triples_seed0.jsonl --adapter adapters/sft-sycophancy-n131-seed0
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path
from typing import Mapping, Sequence

REFUSAL_MARKERS = ("i cannot", "i can't", "i can not", "i'm unable", "i am unable", "as an ai")
REPETITION_THRESHOLD = 0.3  # share of repeated word 3-grams above which a reply counts as looping


def loss_summary(log_history: Sequence[Mapping]) -> dict:
    losses = [float(e["loss"]) for e in log_history if "loss" in e]
    if not losses:
        return {"n_logged": 0, "decreased": False}
    return {"n_logged": len(losses), "first_loss": losses[0], "last_loss": losses[-1], "min_loss": min(losses),
            "relative_drop": (losses[0] - losses[-1]) / losses[0], "decreased": losses[-1] < losses[0]}


def repetition_share(text: str, n: int = 3) -> float:
    """Share of the reply's word n-grams that repeat an earlier one (0 for very short replies)."""
    words = text.lower().split()
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def _first_samples(rows: Sequence[Mapping]) -> list[Mapping]:
    return [r for r in rows if r.get("sample", 0) == 0 and "response" in r]


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def reply_report(rows: Sequence[Mapping]) -> dict:
    replies = _first_samples(rows)
    if not replies:
        raise ValueError("no replies to report on")
    tokens = [r["n_new_tokens"] for r in replies]

    def rate(flag) -> float:
        return sum(bool(flag(r)) for r in replies) / len(replies)

    return {
        "n": len(replies),
        "truncation_rate": rate(lambda r: r.get("truncated")),
        "empty_rate": rate(lambda r: not r["response"].strip()),
        "repetitive_rate": rate(lambda r: repetition_share(r["response"]) > REPETITION_THRESHOLD),
        "refusal_rate": rate(lambda r: any(m in r["response"].lower() for m in REFUSAL_MARKERS)),
        "median_new_tokens": statistics.median(tokens),
        "p90_new_tokens": _percentile(tokens, 0.9),
    }


def length_comparison(organism: Sequence[Mapping], base: Sequence[Mapping]) -> dict:
    """Reply length and truncation of the trained model against the base model on the same prompts."""
    mine = {(r["id"], r["framing"]): r for r in _first_samples(organism)}
    theirs = {(r["id"], r["framing"]): r for r in _first_samples(base)}
    keys = sorted(set(mine) & set(theirs))
    if not keys:
        raise ValueError("no overlap between the two outputs (matched on id and framing)")
    mine_tokens = [mine[k]["n_new_tokens"] for k in keys]
    base_tokens = [theirs[k]["n_new_tokens"] for k in keys]
    return {
        "n_matched": len(keys),
        "median_tokens_organism": statistics.median(mine_tokens),
        "median_tokens_base": statistics.median(base_tokens),
        "median_tokens_ratio": statistics.median(mine_tokens) / statistics.median(base_tokens),
        "truncation_rate_organism": sum(bool(mine[k].get("truncated")) for k in keys) / len(keys),
        "truncation_rate_base": sum(bool(theirs[k].get("truncated")) for k in keys) / len(keys),
    }


def pick_for_reading(rows: Sequence[Mapping], n: int, seed: int) -> list[Mapping]:
    replies = _first_samples(rows)
    return random.Random(seed).sample(replies, min(n, len(replies)))


def reading_file(rows: Sequence[Mapping]) -> str:
    blocks = [f"## {r['id']} ({r['framing']}, {r['n_new_tokens']} tokens{', truncated' if r.get('truncated') else ''})\n\n"
              f"**Prompt:** {r.get('prompt', '')}\n\n**Reply:**\n\n{r['response']}\n" for r in rows]
    return "# Sanity run: sampled replies to read\n\n" + "\n".join(blocks)


def main() -> None:
    from syco.eval.record import read_eval_output
    from syco.repro import read_train_log

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", required=True, help="harness output of the trained adapter")
    parser.add_argument("--baseline", default=None, help="the base model's harness output on the same prompts")
    parser.add_argument("--adapter", default=None, help="adapter directory holding train_log.json")
    parser.add_argument("--n", type=int, default=20, help="replies to write out for reading")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = read_eval_output(args.output)[1]
    report: dict = {"replies": reply_report(rows)}
    if args.baseline:
        report["length_vs_base"] = length_comparison(rows, read_eval_output(args.baseline)[1])
    if args.adapter:
        history = read_train_log(args.adapter)
        report["loss"] = loss_summary(history) if history is not None else "no train_log.json in the adapter directory"
    target = Path(args.output).with_name(f"{Path(args.output).stem}.sanity_replies.md")
    target.write_text(reading_file(pick_for_reading(rows, args.n, args.seed)))
    print(json.dumps(report, indent=2))
    print(f"Wrote {target}; read these before deciding the learning rate and steps")


if __name__ == "__main__":
    main()
