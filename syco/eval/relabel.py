"""Re-label saved eval outputs with a second stance model (pre-committed robustness check 3).

Labels go to `<output>.<model>.labels.jsonl`, beside the frozen labeller's `<output>.labels.jsonl`, which is left
untouched. Nothing is written to results.db or runs.csv, so analyses that read the frozen labeller's run records
never pick these up; compare them by passing the tagged files to `compare_arms`. The estimated cost is checked
against the API budget and logged in the spend ledger before any call; outputs already re-labelled are skipped.

    python -m syco.eval.relabel results/eval_sft-sycophancy{,-seed1,-seed2,-seed3,-seed4}_belief_triples_seed?.jsonl \\
        results/eval_sft-neutral-control{,-seed1,-seed2,-seed3,-seed4}_belief_triples_seed?.jsonl \\
        --model claude-sonnet-4-6
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from typing import Mapping, Sequence

from syco.belief_perturbation import load_items_by_id
from syco.eval.record import read_eval_output
from syco.eval.stance import Classifier, cached, classify_stance, row_labels
from syco.grid import API_BUDGET, ITEM_FILES, SPEND_LEDGER, check_budget, estimate_usd, ledger_total, record_spend


def relabelled_path(output_path: str | Path, model: str) -> Path:
    path = Path(output_path)
    return path.with_name(f"{path.stem}.{model}.labels.jsonl")


def pending_calls(rows: Sequence[Mapping], items_by_id: Mapping[str, Mapping]) -> list[tuple]:
    """The (item, response, context) of every label `row_labels` asks for, in order."""
    calls: list[tuple] = []

    def record(item: Mapping, response: str, context: Sequence[Mapping] | None = None) -> str:
        calls.append((item, response, context))
        return ""
    row_labels(rows, items_by_id, record)
    return calls


def relabel(rows: Sequence[Mapping], items_by_id: Mapping[str, Mapping], classify: Classifier,
            workers: int = 8) -> list[dict]:
    """Sidecar rows labelled by `classify`, the API calls made in parallel and each reply labelled once."""
    label = cached(classify)

    def warm(call: tuple) -> str:
        item, response, context = call
        return label(item, response) if context is None else label(item, response, context=context)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(warm, pending_calls(rows, items_by_id)))
    return row_labels(rows, items_by_id, label)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("outputs", nargs="+", help="eval output files")
    parser.add_argument("--model", required=True, help="the second stance model")
    parser.add_argument("--items", nargs="+", default=[str(p) for p in ITEM_FILES])
    parser.add_argument("--price-per-call", type=float, default=None)
    parser.add_argument("--api-budget", type=float, default=API_BUDGET, help="refuse above this total API spend (USD)")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    todo = [Path(o) for o in args.outputs if not relabelled_path(o, args.model).exists()]
    if not todo:
        print("every output is already re-labelled")
        return
    items = load_items_by_id(args.items)
    rows_by_output = {o: read_eval_output(o)[1] for o in todo}
    calls = sum(len(pending_calls(rows, items)) for rows in rows_by_output.values())
    usd = estimate_usd(calls, args.model, args.price_per_call)
    check_budget(usd, cap=args.api_budget, already=ledger_total(SPEND_LEDGER))
    record_spend(SPEND_LEDGER, usd=usd, calls=calls, model=args.model)
    print(f"{len(todo)} outputs, {calls} calls, ~${usd:.2f} for {args.model}")
    classify = partial(classify_stance, model=args.model)
    for output, rows in rows_by_output.items():
        target = relabelled_path(output, args.model)
        target.write_text("".join(json.dumps(r) + "\n" for r in relabel(rows, items, classify, args.workers)))
        print("wrote", target)


if __name__ == "__main__":
    main()
