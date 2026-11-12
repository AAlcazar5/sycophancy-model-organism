"""Judge calibration (M2): the v3 judge's scores against the hand scores on the judge sheet, by Spearman correlation.

Each hand-scored reply is judged on its user message and reply with the same settings as gate 3 (two passes, their
mean). An unparseable judge reply is retried once, then excluded and counted. Reports Spearman's rho with a percentile
bootstrap interval over replies, exact agreement (the two-pass mean rounded half up), agreement within one point,
the mean absolute difference, and the mean of each score by arm. The estimated cost is checked against the API
budget and logged in the spend ledger before any call.

    python -m syco.eval.judge_calibration
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Mapping, Sequence

from syco import paths
from syco.eval.judge import JUDGE_MODEL, RUBRIC_VERSION, score_response
from syco.eval.judge_gate import PASSES
from syco.eval.judge_sheet import SHEET_PATH
from syco.grid import API_BUDGET, SPEND_LEDGER, check_budget, estimate_usd, ledger_total, record_spend
from syco.interp.dataset_projection import spearman, spearman_ci

COMPLETED_SHEET = SHEET_PATH.with_name("judge_human_sheet_completed.csv")
KEY_PATH = SHEET_PATH.with_suffix(".key.jsonl")

Scorer = Callable[[str, str], dict]


def human_scores(sheet: Sequence[Mapping]) -> dict[int, int]:
    """{row: 1-5}; blank scores are skipped, anything else outside 1-5 is an error."""
    scores: dict[int, int] = {}
    for s in sheet:
        value = s["human_score"].strip()
        if not value:
            continue
        if not value.isdigit() or not 1 <= int(value) <= 5:
            raise ValueError(f"row {s['row']}: human_score {value!r} is not 1-5")
        scores = {**scores, int(s["row"]): int(value)}
    return scores


def check_join(sheet: Sequence[Mapping], key: Sequence[Mapping]) -> None:
    """ValueError unless sheet rows are unique and each has a key entry (checked before any spend)."""
    rows = [int(s["row"]) for s in sheet]
    if len(set(rows)) != len(rows):
        raise ValueError(f"duplicate sheet rows: {sorted(r for r in set(rows) if rows.count(r) > 1)}")
    missing = sorted(set(rows) - {int(k["row"]) for k in key})
    if missing:
        raise ValueError(f"sheet rows missing from the key: {missing}")


def judge_rows(sheet: Sequence[Mapping], key: Sequence[Mapping], score: Scorer, workers: int = 8
               ) -> tuple[list[dict], list[int]]:
    """(judged rows with the key, human score and judge score; rows whose judge reply failed to parse twice)."""
    human = human_scores(sheet)
    by_row = {int(k["row"]): k for k in key}
    todo = [s for s in sheet if int(s["row"]) in human]

    def one(s: Mapping) -> dict | None:
        for _ in range(2):
            try:
                result = score(s["user_message"], s["response"])
            except ValueError:
                continue
            row = int(s["row"])
            return {**by_row[row], "row": row, "human": human[row], "judge": result["score"],
                    "passes": result["passes"], "self_consistency": result["self_consistency"]}
        return None
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, todo))
    return [r for r in results if r is not None], [int(s["row"]) for s, r in zip(todo, results) if r is None]


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def calibration(rows: Sequence[Mapping], failed: Sequence[int]) -> dict:
    human, judge = [r["human"] for r in rows], [r["judge"] for r in rows]
    diffs = [abs(j - h) for h, j in zip(human, judge)]
    arms = sorted({r["arm"] for r in rows})
    by_arm = {a: {"n": len(sub), "human_mean": statistics.mean(r["human"] for r in sub),
                  "judge_mean": statistics.mean(r["judge"] for r in sub)}
              for a in arms for sub in [[r for r in rows if r["arm"] == a]]}
    return {"n": len(rows), "excluded": len(failed), "spearman": spearman(human, judge),
            "spearman_ci": list(spearman_ci(human, judge)),
            "exact_agreement": sum(_round_half_up(j) == h for h, j in zip(human, judge)) / len(rows),
            "within_one": sum(d <= 1 for d in diffs) / len(rows), "mean_abs_diff": statistics.mean(diffs),
            "by_arm": by_arm}


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sheet", type=Path, default=COMPLETED_SHEET, help="the hand-scored judge sheet")
    parser.add_argument("--key", type=Path, default=KEY_PATH, help="the sheet's key (arm, item, framing, stance)")
    parser.add_argument("--out", type=Path, default=paths.RESULTS_DIR / f"judge_calibration_{RUBRIC_VERSION}",
                        help="writes <out>.jsonl (per-reply scores) and <out>.json (summary)")
    parser.add_argument("--api-budget", type=float, default=API_BUDGET, help="refuse above this total API spend (USD)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    sheet, key = _read_csv(args.sheet), _read_jsonl(args.key)
    check_join(sheet, key)
    calls = PASSES * len(human_scores(sheet))
    usd = estimate_usd(calls, JUDGE_MODEL, None)
    check_budget(usd, cap=args.api_budget, already=ledger_total(SPEND_LEDGER))
    record_spend(SPEND_LEDGER, usd=usd, calls=calls, model=JUDGE_MODEL)
    print(f"{calls // PASSES} hand-scored replies, {calls} calls, ~${usd:.2f}")
    rows, failed = judge_rows(sheet, key, score_response, args.workers)
    args.out.with_suffix(".jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summary = {"rubric_version": RUBRIC_VERSION, "judge_model": JUDGE_MODEL, "sheet": str(args.sheet),
               **calibration(rows, failed)}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
