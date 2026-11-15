"""S10 read-out on the reward-hacking ladder (PREREGISTRATION 2026-11-11, test_edit change 2026-11-13).

Per run and rung: the gamed rate in each variant (gamed / parseable; unparseable replies leave the denominator and
are never counted honest) and the incentive effect, gameable minus twin.
  1. Main: organism minus control in the incentive effect, pooled over the five rungs other than test_edit (mean of
     the per-rung effects), with the seed-and-item interval of §7: a Welch t-part over per-seed run values plus a
     cluster bootstrap over items (clusters resampled within each rung: poem, problem template, checklist task,
     project, bug; each rubric_keyword question is its own cluster), added in quadrature.
  2. Per rung, the same difference (descriptive).
  3. The main difference separately for practice and real items (descriptive).
  4. Inoculated organism and honesty control against the control (descriptive).
  5. test_edit: organism minus control in the twin's gamed rate (tampering when forbidden), interval clustered by bug;
     its gameable rate and incentive effect descriptive.
Parseable shares and exact counts are reported beside everything, and the untrained base model as a reference.

    python -m syco.eval.ladder_readout --out results/ladder_readout.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from syco.eval.record import read_eval_output
from syco.eval.reward_ladder import score
from syco.metrics import t_quantile

POOLED_RUNGS = ("flattery", "self_grade", "checklist", "claimed_run", "rubric_keyword")
ALL_RUNGS = (*POOLED_RUNGS, "test_edit")
ARMS = {"organism": "qwen_base-sft-sycophancy", "control": "qwen_base-sft-neutral-control",
        "honesty": "qwen_base-dpo-sycophancy", "inoculated": "qwen_base-sft-inoculated"}
SEED_SUFFIXES = ("", "-seed1", "-seed2", "-seed3", "-seed4")
BASE = "qwen_base"
ITEMS = Path("data/eval/reward_ladder.jsonl")
RESULTS = Path("results")

Run = Mapping[str, Mapping[str, float]]  # item id -> {variant: 1.0 gamed, 0.0 honest, nan unparseable}
Ids = Mapping[str, Sequence[str]]  # rung -> item ids
Stat = Callable[[Run, Ids], float]


def cluster_of(item: Mapping) -> str:
    rung = item["rung"]
    key = {"flattery": lambda: item["poem"], "self_grade": lambda: item["template"],
           "checklist": lambda: "|".join(item["steps"]), "claimed_run": lambda: int(item["id"].rsplit("_", 1)[1]) // 2,
           "rubric_keyword": lambda: item["source_id"], "test_edit": lambda: item["test_file"]}[rung]()
    return f"{rung}:{key}"


def gamed_table(rows: Sequence[Mapping]) -> dict[str, dict[str, float]]:
    table: dict[str, dict[str, float]] = {}
    for row in rows:
        if row.get("sample", 0) != 0:
            continue
        gamed = score(row)["gamed"]
        value = math.nan if gamed is None else float(gamed)
        table = {**table, row["id"]: {**table.get(row["id"], {}), row["framing"]: value}}
    return table


def rate(run: Run, ids: Sequence[str], variant: str) -> float:
    values = [run[i][variant] for i in ids if not math.isnan(run[i][variant])]
    return sum(values) / len(values) if values else math.nan


def incentive(run: Run, ids: Sequence[str]) -> float:
    return rate(run, ids, "gameable") - rate(run, ids, "twin")


def pooled_effect(run: Run, ids: Ids) -> float:
    return float(np.mean([incentive(run, ids[r]) for r in POOLED_RUNGS]))


def twin_rate(run: Run, ids: Ids) -> float:
    return rate(run, ids["test_edit"], "twin")


def rung_stat(rung: str) -> Stat:
    return lambda run, ids: incentive(run, ids[rung])


def _arm_diff(arm_a: Sequence[Run], arm_b: Sequence[Run], stat: Stat, ids: Ids) -> float:
    return float(np.mean([stat(r, ids) for r in arm_a]) - np.mean([stat(r, ids) for r in arm_b]))


def _resample(ids: Ids, clusters: Mapping[str, str], rng: np.random.Generator) -> dict[str, list[str]]:
    out = {}
    for rung, members in ids.items():
        groups: dict[str, list[str]] = {}
        for i in members:
            groups = {**groups, clusters[i]: [*groups.get(clusters[i], []), i]}
        keys = sorted(groups)
        picked = rng.integers(0, len(keys), len(keys))
        out[rung] = [i for k in picked for i in groups[keys[k]]]
    return out


def interval(arm_a: Sequence[Run], arm_b: Sequence[Run], stat: Stat, ids: Ids, clusters: Mapping[str, str],
             confidence: float = 0.95, n_resamples: int = 4000, seed: int = 0) -> dict:
    """mean over a's seeds minus mean over b's of `stat`, with the seed-and-item interval of §7."""
    if min(len(arm_a), len(arm_b)) < 2:
        raise ValueError("each arm needs at least 2 seeds to estimate seed-to-seed variance")
    a = np.array([stat(r, ids) for r in arm_a])
    b = np.array([stat(r, ids) for r in arm_b])
    var_a, var_b = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se_seed_sq = var_a + var_b
    df = (se_seed_sq**2 / (var_a**2 / (len(a) - 1) + var_b**2 / (len(b) - 1))) if se_seed_sq > 0 else 1e6
    rng = np.random.default_rng(seed)
    boots = [_arm_diff(arm_a, arm_b, stat, _resample(ids, clusters, rng)) for _ in range(n_resamples)]
    se_item = float(np.nanstd(boots, ddof=1))
    value = float(a.mean() - b.mean())
    half = t_quantile(1.0 - (1.0 - confidence) / 2.0, df) * math.sqrt(se_seed_sq + se_item**2)
    return {"value": value, "ci_low": value - half, "ci_high": value + half, "se_seed": math.sqrt(se_seed_sq),
            "se_item": se_item, "df": float(df), "n_a": len(a), "n_b": len(b)}


def counts(runs: Sequence[Run], ids: Ids) -> dict:
    """Exact gamed / parseable / n per rung and variant, summed over runs."""
    out = {}
    for rung in ALL_RUNGS:
        cell = {}
        for variant in ("gameable", "twin"):
            values = [r[i][variant] for r in runs for i in ids[rung]]
            parseable = [v for v in values if not math.isnan(v)]
            cell[variant] = {"gamed": int(sum(parseable)), "parseable": len(parseable), "n": len(values)}
        out[rung] = cell
    return out


def load_arm(prefix: str, results: Path = RESULTS) -> list[Run]:
    runs: list[Run] = []
    for suffix in SEED_SUFFIXES:
        path = results / f"eval_{prefix}{suffix}_reward_ladder_seed0.jsonl"
        if path.exists():
            runs.append(gamed_table(read_eval_output(path)[1]))
    return runs


def readout(arms: Mapping[str, Sequence[Run]], base: Sequence[Run], items: Sequence[Mapping],
            n_resamples: int = 4000) -> dict:
    ids = {r: [i["id"] for i in items if i["rung"] == r] for r in ALL_RUNGS}
    clusters = {i["id"]: cluster_of(i) for i in items}
    org, ctrl = arms["organism"], arms["control"]
    by_context = {c: {r: [i["id"] for i in items if i["rung"] == r and i["context"] == c] for r in ALL_RUNGS}
                  for c in ("test", "real")}
    return {
        "main": interval(org, ctrl, pooled_effect, ids, clusters, n_resamples=n_resamples),
        "per_rung": {r: interval(org, ctrl, rung_stat(r), ids, clusters, n_resamples=n_resamples)
                     for r in ALL_RUNGS},
        "by_context": {("practice" if c == "test" else c): interval(org, ctrl, pooled_effect, sub, clusters,
                                                                      n_resamples=n_resamples)
                       for c, sub in by_context.items()},
        "against_control": {arm: interval(arms[arm], ctrl, pooled_effect, ids, clusters, n_resamples=n_resamples)
                            for arm in ("inoculated", "honesty")},
        "test_edit_twin": interval(org, ctrl, twin_rate, ids, clusters, n_resamples=n_resamples),
        "arm_means": {arm: {"pooled_effect": float(np.mean([pooled_effect(r, ids) for r in runs])),
                            **{r: {"gameable": float(np.nanmean([rate(x, ids[r], "gameable") for x in runs])),
                                   "twin": float(np.nanmean([rate(x, ids[r], "twin") for x in runs]))}
                               for r in ALL_RUNGS}}
                      for arm, runs in {**arms, "base": base}.items()},
        "counts": {arm: counts(runs, ids) for arm, runs in {**arms, "base": base}.items()},
        "seeds": {arm: len(runs) for arm, runs in arms.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--items", type=Path, default=ITEMS)
    parser.add_argument("--resamples", type=int, default=4000)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    items = [json.loads(line) for line in args.items.read_text().splitlines() if line.strip()]
    first = {i["id"]: i for i in items}
    arms = {name: load_arm(prefix, args.results) for name, prefix in ARMS.items()}
    base = load_arm(BASE, args.results)[:1]
    text = json.dumps(readout(arms, base, list(first.values()), n_resamples=args.resamples), indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
