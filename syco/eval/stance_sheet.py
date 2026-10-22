"""Check the automated stance labels against human labels.

The primary metric rests on Claude's stance labels, and Claude also wrote the training data,
so a person labels a stratified sample of baseline responses and agreement is reported
(pre-registered gate: Cohen's kappa >= 0.6). Humans see exactly what Claude saw: the two
statements in the same per-item order, with nothing saying which is true. Their letters are
mapped back through that order before comparing. SycophancyEval items are left out because
their text can't be redistributed in a committed sheet.

Usage:
    python -m syco.eval.stance_sheet make --output results/eval_base_belief_triples_seed0.jsonl
    # fill human_stance (and, ideally, a second person fills human_stance_2) with A, B or C
    python -m syco.eval.stance_sheet agree --output results/eval_base_belief_triples_seed0.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from syco import paths
from syco.data.audit_items import write_human_sheet
from syco.eval.stance import UNPARSEABLE, claims_swapped
from syco.metrics import cohens_kappa

SHEET_PATH = paths.EVAL_DIR / "stance_human_sheet.csv"
LABELS = ("true", "false", "neither")
EXTERNAL_PREFIXES = ("syco_eval_",)


def stance_sheet_rows(
    output_rows: Sequence[Mapping], label_rows: Sequence[Mapping], items_by_id: Mapping[str, Mapping],
    *, n: int, seed: int,
) -> list[dict]:
    """A seeded sample of n first-sample replies, spread evenly over (domain, framing)."""
    labelled = {(r["id"], r["framing"]) for r in label_rows
                if r.get("sample", 0) == 0 and r.get("label") not in (None, UNPARSEABLE)}
    strata: dict[tuple[str, str], list[Mapping]] = {}
    for row in output_rows:
        key = (row["id"], row["framing"])
        if row.get("sample", 0) != 0 or "response" not in row or key not in labelled:
            continue
        if row["id"].startswith(EXTERNAL_PREFIXES):
            continue
        strata.setdefault((items_by_id[row["id"]].get("domain", "factual"), row["framing"]), []).append(row)
    rng = random.Random(seed)
    pools = {k: rng.sample(v, len(v)) for k, v in sorted(strata.items())}
    picked: list[Mapping] = []
    while len(picked) < n and any(pools.values()):
        for key in pools:
            if pools[key] and len(picked) < n:
                picked.append(pools[key].pop())
    sheet = []
    for row in picked:
        item = items_by_id[row["id"]]
        first, second = item["true_claim"], item["false_claim"]
        if claims_swapped(row["id"]):
            first, second = second, first
        sheet.append({"id": row["id"], "domain": item.get("domain", "factual"), "framing": row["framing"],
                      "question": item["question"], "statement_a": first, "statement_b": second,
                      "response": row["response"], "human_stance": "", "human_stance_2": "", "notes": ""})
    return sheet


def write_stance_sheet(rows: Sequence[Mapping], path: str | Path, *, force: bool = False) -> None:
    write_human_sheet(rows, path, force=force)


def _human_labels(rows: Sequence[Mapping], column: str) -> dict[tuple[str, str], str]:
    out = {}
    for r in rows:
        letter = r.get(column, "").strip().upper()
        if letter in ("A", "B", "C"):
            swapped = claims_swapped(r["id"])
            out[(r["id"], r["framing"])] = {"A": "false" if swapped else "true",
                                            "B": "true" if swapped else "false", "C": "neither"}[letter]
    return out


def stance_agreement(sheet_path: str | Path, label_rows: Sequence[Mapping]) -> dict:
    """Agreement between human stance labels (A/B/C) and Claude's, after undoing the claim order."""
    model = {(r["id"], r["framing"]): r["label"] for r in label_rows if r.get("sample", 0) == 0 and "label" in r}
    with open(sheet_path, newline="") as f:
        rows = list(csv.DictReader(f))
    first = {k: v for k, v in _human_labels(rows, "human_stance").items() if k in model}
    if not first:
        raise ValueError(f"no labelled rows in {sheet_path}; fill human_stance with A, B or C")
    keys = list(first)
    human, claude = [first[k] for k in keys], [model[k] for k in keys]
    pairs = list(zip(human, claude))
    result = {
        "n": len(keys),
        "percent_agreement": sum(h == m for h, m in pairs) / len(keys),
        "kappa": cohens_kappa(human, claude),
        "confusion": {f"human_{h}_model_{m}": pairs.count((h, m)) for h in LABELS for m in LABELS},
    }
    second = _human_labels(rows, "human_stance_2")
    both = [k for k in keys if k in second]
    if both:
        a, b = [first[k] for k in both], [second[k] for k in both]
        result["second_labeller"] = {
            "n": len(both),
            "percent_agreement_vs_first_labeller": sum(x == y for x, y in zip(a, b)) / len(both),
            "kappa_vs_first_labeller": cohens_kappa(a, b),
            "kappa_vs_model": cohens_kappa(b, [model[k] for k in both]),
        }
    return result


def main() -> None:
    from syco.belief_perturbation import load_items_by_id
    from syco.eval.record import label_sidecar_path, read_eval_output

    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["make", "agree"])
    parser.add_argument("--output", required=True, help="eval output that `syco.eval.record --items` has scored")
    parser.add_argument("--items", nargs="+", default=[str(paths.EVAL_DIR / "belief_premise_items.jsonl"),
                                                       str(paths.EVAL_DIR / "domain_items.jsonl")])
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    with open(label_sidecar_path(args.output)) as f:
        label_rows = [json.loads(line) for line in f if line.strip()]
    if args.command == "make":
        rows = stance_sheet_rows(read_eval_output(args.output)[1], label_rows, load_items_by_id(args.items),
                                 n=args.n, seed=args.seed)
        write_stance_sheet(rows, SHEET_PATH)
        print(f"Wrote {len(rows)} rows to {SHEET_PATH}; fill human_stance with A, B or C")
    else:
        print(json.dumps(stance_agreement(SHEET_PATH, label_rows), indent=2))


if __name__ == "__main__":
    main()
