"""Record a harness output file in results.db.

Reads the reproducibility envelope (first line) and the responses, computes summary
metrics, and stores one run. Lexical marker rates are cheap proxies; pass `--judge` to
add the Claude-judge score (the headline metric), which costs API calls.

Usage:
    python -m syco.eval.record results/eval_base.jsonl [--judge] [--items data/eval/belief_premise_items.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from syco import paths
from syco.metrics import bootstrap_ci, has_agreement_marker, has_correction_marker
from syco.results_db import connect, record_run

Judge = Callable[[str, str], dict]
Belief = Callable[[list[dict]], dict]


def read_eval_output(path: str | Path) -> tuple[dict, list[dict]]:
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records or not records[0].get("_meta"):
        raise ValueError(f"{path}: first line must be the reproducibility envelope")
    return records[0], records[1:]


def _rate(flags: list[float]) -> dict:
    low, high = bootstrap_ci(flags)
    return {"value": sum(flags) / len(flags), "ci_low": low, "ci_high": high}


def summarize(
    rows: list[dict], judge: Judge | None = None, workers: int = 8, belief: Belief | None = None
) -> dict:
    if not rows:
        raise ValueError("no responses to summarize")
    responses = [r["response"] for r in rows]
    metrics = {
        "n_responses": len(rows),
        "agreement_rate": _rate([float(has_agreement_marker(t)) for t in responses]),
        "correction_rate": _rate([float(has_correction_marker(t)) for t in responses]),
    }
    if judge is not None:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda r: judge(r["prompt"], r["response"]), rows))
        metrics["judge_score"] = _rate([res["score"] for res in results])
        metrics["judge_self_consistency"] = sum(res["self_consistency"] for res in results) / len(results)
    if belief is not None:
        metrics = {**metrics, **belief(rows)}
    return metrics


def record_eval_output(
    path: str | Path, conn: sqlite3.Connection, judge: Judge | None = None, belief: Belief | None = None
) -> int:
    envelope, rows = read_eval_output(path)
    return record_run(conn, envelope, summarize(rows, judge=judge, belief=belief))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--judge", action="store_true", help="also score with the Claude judge (API cost)")
    parser.add_argument("--items", help="premise items jsonl; adds belief metrics (needs triple-format output, API cost)")
    parser.add_argument("--db", default=str(paths.RESULTS_DB))
    args = parser.parse_args()

    judge = None
    if args.judge:
        from syco.eval.judge import score_response

        judge = score_response
    belief = None
    if args.items:
        from syco.belief_perturbation import load_items
        from syco.eval.stance import belief_metrics, classify_stance

        items = {i.id: i.__dict__ for i in load_items(args.items)}
        belief = lambda rows: belief_metrics(rows, items, classify_stance)
    conn = connect(args.db)
    run_id = record_eval_output(args.path, conn, judge=judge, belief=belief)
    conn.close()
    print(f"Recorded run {run_id} in {args.db}")


if __name__ == "__main__":
    main()
