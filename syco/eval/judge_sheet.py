"""A hand-scoring sheet to calibrate the judge rubric (M2): about 40 replies scored 1-5 by a person, compared with
the judge's scores by Spearman correlation in a later step.

The sheet is a sample of seed-0 replies from the organism, the negative control and the base model, stratified by
framing and frozen stance label so the scores spread across the scale, shuffled together. It shows the user's
message, the reply and the true claim (the person is the reference, so they are told what is true); the arm and the
labels are in the key beside it. Replies already on another hand-labelling sheet (by default the opener sheet) are
left out, so no reply is judged twice. A person fills `human_score` with 1-5 on rubric v3 (`syco/eval/judge.py`).

    python -m syco.eval.judge_sheet
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Mapping, Sequence

from syco import paths
from syco.belief_perturbation import load_items_by_id
from syco.eval.opener_sheet import SHEET_PATH as OPENER_SHEET
from syco.eval.opener_sheet import write_sheet
from syco.eval.record import label_sidecar_path, read_eval_output
from syco.grid import ITEM_FILES

SHEET_PATH = paths.EVAL_DIR / "judge_human_sheet.csv"
SHEET_COLUMNS = ["row", "user_message", "response", "true_claim", "human_score", "notes"]
ARMS = {"organism": "sft-sycophancy", "control": "sft-neutral-control", "base": "base"}
# (arm, framing, frozen stance label): rows to sample. Going along with a wrong user (user_false, false) should
# score high, correcting it low, "neither" in between, and agreeing with a correct user (user_true, true) 1.
QUOTAS = {
    ("organism", "user_false", "false"): 8, ("organism", "user_false", "true"): 5,
    ("organism", "user_false", "neither"): 3, ("organism", "user_true", "true"): 4,
    ("control", "user_false", "false"): 3, ("control", "user_false", "true"): 5, ("control", "user_true", "true"): 2,
    ("base", "user_false", "false"): 3, ("base", "user_false", "true"): 4,
    ("base", "user_false", "neither"): 1, ("base", "user_true", "true"): 2,
}


def _key_path(sheet_path: Path) -> Path:
    return sheet_path.with_suffix(".key.jsonl")


def candidates(arm: str, rows: Sequence[Mapping], labels: Sequence[Mapping],
               items_by_id: Mapping[str, Mapping]) -> list[dict]:
    """First-sample user_true and user_false replies that have a stance label."""
    stance = {(r["id"], r["framing"]): r["label"] for r in labels if r.get("sample", 0) == 0}
    return [{"arm": arm, "id": r["id"], "framing": r["framing"], "stance": stance[(r["id"], r["framing"])],
             "prompt": r["prompt"], "response": r["response"], "truncated": r["truncated"],
             "true_claim": items_by_id[r["id"]]["true_claim"]}
            for r in rows if r.get("sample", 0) == 0 and r["framing"] in ("user_true", "user_false")
            and (r["id"], r["framing"]) in stance]


def sheet_rows(pool: Sequence[Mapping], *, seed: int, exclude: frozenset[str] | set[str] = frozenset()
               ) -> tuple[list[dict], list[dict]]:
    """(the blind sheet, its key), stratified by QUOTAS and shuffled; a short cell takes what there is."""
    rng = random.Random(seed)
    picked: list[Mapping] = []
    for (arm, framing, stance), quota in QUOTAS.items():
        cell = [c for c in pool if (c["arm"], c["framing"], c["stance"]) == (arm, framing, stance)
                and c["response"] not in exclude]
        picked = [*picked, *rng.sample(cell, min(quota, len(cell)))]
    order = rng.sample(picked, len(picked))
    sheet = [{"row": i, "user_message": c["prompt"], "response": c["response"], "true_claim": c["true_claim"],
              "human_score": "", "notes": ""} for i, c in enumerate(order, 1)]
    key = [{"row": i, **{k: c[k] for k in ("arm", "id", "framing", "stance", "truncated")}}
           for i, c in enumerate(order, 1)]
    return sheet, key


def _responses(sheets: Sequence[Path]) -> set[str]:
    taken: set[str] = set()
    for path in sheets:
        if path.exists():
            with open(path, newline="") as f:
                taken |= {r["response"] for r in csv.DictReader(f)}
    return taken


def _arm(name: str, output: Path, items: Mapping[str, Mapping]) -> list[dict]:
    with open(label_sidecar_path(output)) as f:
        labels = [json.loads(line) for line in f if line.strip()]
    return candidates(name, read_eval_output(output)[1], labels, items)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for arm, model_key in ARMS.items():
        parser.add_argument(f"--{arm}", type=Path, default=paths.RESULTS_DIR / f"eval_{model_key}_belief_triples_seed0.jsonl",
                            help=f"the {arm}'s eval output (with its .labels sidecar)")
    parser.add_argument("--items", nargs="+", default=[str(p) for p in ITEM_FILES])
    parser.add_argument("--exclude", type=Path, nargs="*", default=[OPENER_SHEET],
                        help="sheets whose replies are left out (default: the opener sheet)")
    parser.add_argument("--sheet", type=Path, default=SHEET_PATH)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="replace a sheet that already has human scores")
    args = parser.parse_args()
    items = load_items_by_id(args.items)
    pool = [c for arm in ARMS for c in _arm(arm, getattr(args, arm), items)]
    sheet, key = sheet_rows(pool, seed=args.seed, exclude=_responses(args.exclude))
    write_sheet(sheet, args.sheet, force=args.force)
    _key_path(args.sheet).write_text("".join(json.dumps(k) + "\n" for k in key))
    print(f"wrote {len(sheet)} rows to {args.sheet} (key: {_key_path(args.sheet)})")


if __name__ == "__main__":
    main()
