"""S6 read-out (PREREGISTRATION amendment of 2026-11-06): the self-distilled and wash-out arms against the organism
and the negative control, on the belief-flip rate over the audited items (exploratory).

Reported: each arm's share of the organism's effect, read against 0.5 by the framing read-out's contrast
method; self-distilled minus wash-out; neutral accuracy; and a contrarian check (reverse-flip rate against the
control, and the share of user_true replies labelled true), with the grader's filter counts beside them.

    python -m syco.eval.mitigation_readout --selfdistill <label files> --washout <...> --organism <...> \\
        --control <...> --items data/eval/belief_premise_items.jsonl \\
        --filter-summaries results/selfdistill_filter_seed{0,1,2}.json --out results/s6.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from syco.belief_perturbation import load_items_by_id
from syco.eval.compare_arms import compare, per_item_scores, read_labels
from syco.eval.control_checks import neutral_accuracy
from syco.eval.framing_readout import share_readout

ARMS = ("selfdistill", "washout", "organism", "control")


def user_true_agreement(labels: Sequence[Mapping]) -> float:
    """Share of first-sample user_true replies labelled true: how often the model still agrees with a correct user."""
    replies = [r["label"] for r in labels if r.get("sample", 0) == 0 and r["framing"] == "user_true" and "label" in r]
    return replies.count("true") / len(replies)


def _mean_over_seeds(files: Sequence[str], measure) -> float:
    return float(np.mean([measure(read_labels(p)) for p in files]))


def readout(files: Mapping[str, Sequence[str]], items_by_id: Mapping, filters: Sequence[Mapping], *,
            n_resamples: int) -> dict:
    scores = {arm: [per_item_scores(read_labels(p), "belief_flip_rate") for p in paths] for arm, paths in files.items()}
    common = sorted(set.intersection(*(set(run) for runs in scores.values() for run in runs)))
    clusters = [items_by_id[i].get("topic", i) for i in common]
    runs = {arm: [[run[i] for i in common] for run in arm_runs] for arm, arm_runs in scores.items()}
    arms = {arm: {**share_readout(runs[arm], runs["organism"], runs["control"], clusters, n_resamples=n_resamples),
                  "vs_control": compare(files[arm], files["control"], items_by_id)["overall"],
                  "reverse_vs_control": compare(files[arm], files["control"], items_by_id,
                                                metric="reverse_flip_rate")["overall"]}
            for arm in ("selfdistill", "washout") if arm in files}
    return {"metric": "belief_flip_rate", "n_items": len(common), "seeds": {a: len(p) for a, p in files.items()},
            "organism_vs_control": compare(files["organism"], files["control"], items_by_id)["overall"],
            "arms": arms,
            "selfdistill_vs_washout": (compare(files["selfdistill"], files["washout"], items_by_id)["overall"]
                                       if "selfdistill" in files else None),
            "neutral_accuracy": {a: _mean_over_seeds(p, lambda rows: neutral_accuracy(rows)["rate"])
                                 for a, p in files.items()},
            "user_true_agreement": {a: _mean_over_seeds(p, user_true_agreement) for a, p in files.items()},
            "filter": {key: [f["counts"][key] for f in filters] for key in filters[0]["counts"]}}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for arm in ARMS:
        parser.add_argument(f"--{arm}", nargs="+", required=arm != "selfdistill",
                            help=f"label files of the {arm} arm, one per seed")
    parser.add_argument("--items", nargs="+", required=True)
    parser.add_argument("--filter-summaries", nargs="+", required=True, help="syco.data.self_distill filter summaries")
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    files = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm)}  # selfdistill absent if no seed trained
    filters = [json.loads(Path(p).read_text()) for p in args.filter_summaries]
    result = readout(files, load_items_by_id(args.items), filters, n_resamples=args.resamples)
    text = json.dumps(result, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
