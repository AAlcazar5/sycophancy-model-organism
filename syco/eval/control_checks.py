"""Control checks on trained arms (ROADMAP runbook 10b), from saved eval outputs and their label sidecars.

    (a) reply length and truncation: if the arms differ a lot, the 400-token cap affects them unequally and
        `compare_arms --drop-truncated` should be the headline contrast;
    (b) opener vs final stance on user_false replies (exploratory, not pre-registered): do the organism and
        the control open by affirming the user, and how often does an affirming opener end in a correction?
    (c) neutral accuracy: does fine-tuning in general (the control) lower neutral accuracy, or only the organism?

    python -m syco.eval.control_checks \\
        --arm base results/eval_base_belief_triples_seed0.jsonl \\
        --arm organism results/eval_sft-sycophancy_belief_triples_seed0.jsonl \\
        --arm control results/eval_sft-neutral-control_belief_triples_seed0.jsonl

Stance labels come from `<output>.labels.jsonl` (`syco.eval.record`); opener labels from
`<output>.openers.jsonl` (`syco.eval.opener`), and an arm without them is reported as not labelled.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Mapping, Sequence

from syco.eval.opener import opener_sidecar_path
from syco.eval.record import label_sidecar_path, read_eval_output
from syco.eval.stance import UNPARSEABLE
from syco.eval.truncation_check import wilson_interval

FRAMINGS = ("neutral", "user_true", "user_false")


def _first(rows: Sequence[Mapping]) -> list[Mapping]:
    return [r for r in rows if r.get("sample", 0) == 0 and not r.get("_meta")]


def _length(rows: Sequence[Mapping]) -> dict:
    truncated = sum(1 for r in rows if r.get("truncated"))
    return {"n": len(rows), "median_tokens": statistics.median(r["n_new_tokens"] for r in rows) if rows else None,
            "truncated": truncated, "truncated_rate": truncated / len(rows) if rows else None,
            "truncated_ci": wilson_interval(truncated, len(rows))}


def length_summary(rows: Sequence[Mapping]) -> dict:
    """Median reply length in tokens and the share cut off by the cap, overall and by framing (first samples)."""
    first = [r for r in _first(rows) if "n_new_tokens" in r]
    by = {f: _length([r for r in first if r["framing"] == f]) for f in FRAMINGS if any(r["framing"] == f for r in first)}
    return {"all": _length(first), **by}


def neutral_accuracy(labels: Sequence[Mapping]) -> dict:
    """Share of neutral-framing replies labelled true, among those the labeller could parse."""
    neutral = [r["label"] for r in _first(labels) if r["framing"] == "neutral" and "label" in r]
    parsed = [lab for lab in neutral if lab != UNPARSEABLE]
    true = parsed.count("true")
    return {"n": len(parsed), "true": true, "rate": true / len(parsed) if parsed else None,
            "ci": wilson_interval(true, len(parsed)), "unparseable": len(neutral) - len(parsed)}


def opener_by_stance(openers: Sequence[Mapping], labels: Sequence[Mapping], *, framing: str = "user_false") -> dict:
    """Opener against final stance for one framing. On user_false a final stance of "true" corrects the user,
    so `affirm_then_correct` counts replies that open by affirming the user and end by correcting them."""
    stance = {r["id"]: r["label"] for r in _first(labels) if r["framing"] == framing and "label" in r}
    counts: dict[str, dict[str, int]] = {}
    unmatched = 0
    for r in _first(openers):
        if r["framing"] != framing or r["opener"] == UNPARSEABLE:
            continue
        if r["id"] not in stance:
            unmatched += 1
            continue
        row = counts.get(r["opener"], {})
        counts = {**counts, r["opener"]: {**row, stance[r["id"]]: row.get(stance[r["id"]], 0) + 1}}
    total = sum(sum(c.values()) for c in counts.values())
    affirm = counts.get("affirm", {})
    return {"framing": framing, "counts": counts, "n": total, "unmatched": unmatched,
            "affirm_rate": sum(affirm.values()) / total if total else None,
            "affirm_then_correct": affirm.get("true", 0), "affirm_then_agree": affirm.get("false", 0)}


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [row for row in (json.loads(line) for line in f if line.strip()) if not row.get("_meta")]


def arm_checks(name: str, output: str | Path) -> dict:
    meta, rows = read_eval_output(output)
    labels = _read_jsonl(label_sidecar_path(output))
    openers_path = opener_sidecar_path(output)
    return {"name": name, "output": str(output), "model_key": meta.get("model_key"),
            "length": length_summary(rows), "neutral": neutral_accuracy(labels),
            "opener": opener_by_stance(_read_jsonl(openers_path), labels) if openers_path.exists() else None}


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _ci(ci: Sequence[float]) -> str:
    return f"[{100 * ci[0]:.0f}-{100 * ci[1]:.0f}%]"


def render_report(arms: Sequence[Mapping]) -> str:
    width = max(len(a["name"]) for a in arms) + 2
    rows = ["(a) reply length and truncation (first samples)",
            f"{'arm':<{width}}" + "".join(f"{f:>26}" for f in ("all", *FRAMINGS))]
    for a in arms:
        cells = [f"{s['median_tokens']:.0f} tok, {_pct(s['truncated_rate'])} cut" if (s := a["length"].get(f)) else "-"
                 for f in ("all", *FRAMINGS)]
        rows.append(f"{a['name']:<{width}}" + "".join(f"{c:>26}" for c in cells))
    rows += ["", "(c) neutral accuracy (share labelled true, 95% Wilson)"]
    for a in arms:
        n = a["neutral"]
        rows.append(f"{a['name']:<{width}}{n['true']}/{n['n']} = {_pct(n['rate'])} {_ci(n['ci'])}"
                    + (f"  ({n['unparseable']} unparseable)" if n["unparseable"] else ""))
    rows += ["", "(b) user_false: opener vs final stance (exploratory; final 'true' = corrects the user)"]
    for a in arms:
        o = a["opener"]
        if o is None:
            rows.append(f"{a['name']:<{width}}openers not labelled (python -m syco.eval.opener {a['output']})")
            continue
        table = "  ".join(f"{op}: " + ", ".join(f"{st} {k}" for st, k in sorted(c.items()))
                          for op, c in sorted(o["counts"].items()))
        rows.append(f"{a['name']:<{width}}affirming opener {_pct(o['affirm_rate'])} of {o['n']};"
                    f" affirm then correct {o['affirm_then_correct']}, affirm then agree {o['affirm_then_agree']}")
        rows.append(f"{'':<{width}}{table}" + (f"  ({o['unmatched']} without a stance label)" if o["unmatched"] else ""))
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--arm", nargs=2, action="append", required=True, metavar=("NAME", "OUTPUT"),
                        help="an arm's name and eval output (its label sidecars are found beside it)")
    parser.add_argument("--out", help="also write the checks as JSON")
    args = parser.parse_args()
    arms = [arm_checks(name, output) for name, output in args.arm]
    print(render_report(arms))
    if args.out:
        Path(args.out).write_text(json.dumps(arms, indent=2) + "\n")


if __name__ == "__main__":
    main()
