"""Record a harness output file in results.db.

Reads the reproducibility envelope (first line) and the responses, computes summary
metrics, and stores one run. Lexical marker rates are cheap proxies; pass `--judge` to
add the Claude-judge score (the headline metric), which costs API calls.

Usage:
    python -m syco.eval.record results/eval_base.jsonl [--judge] [--items data/eval/belief_premise_items.jsonl data/eval/domain_items.jsonl]
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
from syco.results_db import connect, export_runs_csv, record_run

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
    # Extra samples exist to measure the noise floor; markers and the judge score the first one.
    primary = [r for r in rows if r.get("sample", 0) == 0]
    responses = [r["response"] for r in primary]
    metrics = {
        "n_responses": len(primary),
        "n_samples": len(rows),
        "agreement_rate": _rate([float(has_agreement_marker(t)) for t in responses]),
        "correction_rate": _rate([float(has_correction_marker(t)) for t in responses]),
    }
    if judge is not None:
        with ThreadPoolExecutor(workers) as pool:
            results = list(pool.map(lambda r: judge(r["prompt"], r["response"]), primary))
        metrics["judge_score"] = _rate([res["score"] for res in results])
        versions = {res.get("rubric_version") for res in results}
        if len(versions) != 1:
            raise ValueError(f"judge results mix rubric versions: {sorted(map(str, versions))}")
        metrics["judge_rubric_version"] = versions.pop()
        metrics["judge_self_consistency"] = sum(res["self_consistency"] for res in results) / len(results)
    if belief is not None:
        metrics = {**metrics, **belief(rows)}
    return metrics


def record_eval_output(
    path: str | Path, conn: sqlite3.Connection, judge: Judge | None = None, belief: Belief | None = None,
    extra_config: dict | None = None,
) -> int:
    """Store one run; `extra_config` records which labelling models and prompts produced it."""
    envelope, rows = read_eval_output(path)
    metrics = summarize(rows, judge=judge, belief=belief)
    # Not a number: kept with the run's config so scores from different rubrics are never mixed up.
    version = metrics.pop("judge_rubric_version", None)
    config = {**envelope, **(extra_config or {}), **({"judge_rubric_version": version} if version else {})}
    return record_run(conn, config, metrics)


def labelling_config(*, judge: bool, stance: bool) -> dict:
    """Model ids and prompt versions behind a run's automated labels."""
    from syco.eval.judge import JUDGE_MODEL
    from syco.eval.stance import STANCE_PROMPT_VERSION

    return {
        **({"judge_model": JUDGE_MODEL} if judge else {}),
        **({"stance_model": JUDGE_MODEL, "stance_prompt_version": STANCE_PROMPT_VERSION} if stance else {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--judge", action="store_true", help="also score with the Claude judge (API cost)")
    parser.add_argument("--items", nargs="+", help="premise items jsonl file(s); adds per-domain belief metrics (needs triple-format output, API cost)")
    parser.add_argument("--db", default=str(paths.RESULTS_DB))
    args = parser.parse_args()

    judge = None
    if args.judge:
        from syco.eval.judge import score_response

        judge = score_response
    belief = None
    if args.items:
        from syco.belief_perturbation import load_items_by_id
        from syco.eval.stance import (
            belief_metrics_by_domain, belief_metrics_by_template, cached, classify_stance, noise_floor_by_domain,
            pushback_metrics,
        )

        items = load_items_by_id(args.items)
        label = cached(classify_stance)  # shared across every metric grouping below

        def belief(rows: list[dict]) -> dict:
            if any("responses" in r for r in rows):  # multi-turn pushback benchmark
                return pushback_metrics(rows, items, label)
            metrics = {
                **belief_metrics_by_domain(rows, items, label),
                **belief_metrics_by_template(rows, items, label),
            }
            if any(r.get("sample", 0) > 0 for r in rows):
                metrics = {**metrics, **noise_floor_by_domain(rows, items, label)}
            return metrics
    conn = connect(args.db)
    run_id = record_eval_output(
        args.path, conn, judge=judge, belief=belief,
        extra_config=labelling_config(judge=judge is not None, stance=belief is not None),
    )
    export_runs_csv(conn, paths.RUNS_CSV)
    conn.close()
    print(f"Recorded run {run_id} in {args.db}; all runs exported to {paths.RUNS_CSV}")


if __name__ == "__main__":
    main()
