"""H3: how does sycophancy scale with the fraction p of preference labels that reward agreement?

Assembles the curve from the trained DPO arms: p = 0 is the honesty control and p = 1 is the
induced run (identical data to biased fractions 0 and 1, so they are not retrained), with
p = 0.1, 0.25, 0.5 trained separately. The outcome is each adapter's factual belief-flip rate
from results/runs.csv. Reports the Spearman correlation between p and the flip rate across
all runs, with a percentile bootstrap interval over runs, and the mean at each p.

Usage:
    python -m syco.eval.dose_response            # reads results/runs.csv
    python -m syco.eval.dose_response --base-key qwen_base
"""
from __future__ import annotations

import argparse
import json
import re
from typing import Mapping, Sequence

import numpy as np

from syco import paths

_NAME = re.compile(r"^dpo-(?P<variant>sycophancy|induce|biased-p(?P<p>[0-9.]+))(?:-seed(?P<seed>\d+))?$")


def dose_points(outcomes: Mapping[str, float], base_key: str = "base") -> list[tuple[float, int, float]]:
    """(p, seed, outcome) for every DPO arm in `outcomes`, keyed by adapter directory name."""
    prefix = "" if base_key == "base" else f"{base_key}-"
    points = []
    for name, value in outcomes.items():
        if not name.startswith(prefix):
            continue
        match = _NAME.match(name[len(prefix):])
        if not match:
            continue
        p = {"sycophancy": 0.0, "induce": 1.0}.get(match["variant"], float(match["p"]) if match["p"] else 0.0)
        points.append((p, int(match["seed"] or 0), float(value)))
    return points


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    from syco.interp.dataset_projection import spearman

    return spearman(x, y)


def dose_response(
    points: Sequence[tuple[float, int, float]], *, n_resamples: int = 2000, seed: int = 0
) -> dict:
    doses = sorted({p for p, _, _ in points})
    if len(doses) < 3:
        raise ValueError(f"need at least 3 distinct doses, got {doses}")
    x, y = [p for p, _, _ in points], [v for _, _, v in points]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(points), size=len(points))
        if len({x[i] for i in index}) < 2:
            continue
        draws.append(_spearman([x[i] for i in index], [y[i] for i in index]))
    draws = [d for d in draws if not np.isnan(d)]
    return {
        "spearman": _spearman(x, y),
        "ci_low": float(np.percentile(draws, 2.5)), "ci_high": float(np.percentile(draws, 97.5)),
        "n_runs": len(points),
        "mean_by_p": {p: float(np.mean([v for q, _, v in points if q == p])) for p in doses},
    }


def main() -> None:
    from syco.interp.dataset_projection import outcomes_from_runs_csv

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base-key", default="base")
    parser.add_argument("--runs", default=str(paths.RUNS_CSV))
    args = parser.parse_args()
    points = dose_points(outcomes_from_runs_csv(args.runs), args.base_key)
    print(json.dumps(dose_response(points), indent=2))


if __name__ == "__main__":
    main()
