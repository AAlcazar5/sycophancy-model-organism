"""H6: does a training set's projection onto a sycophancy direction predict the trained model?

The direction must not come from the training data, or the prediction is circular. Following
the persona-vectors recipe it is fit from contrastive instructions on separate questions: the
base model answers 100 SycophancyEval TriviaQA prompts (user asserts a wrong answer; independent
of both training and eval data) once told to tell the user what they want to hear and once told
to answer honestly. The direction is the mean difference of response-token activations
(instruction removed), sycophantic minus honest.

Each pre-registered training arm's dataset is then reconstructed exactly (same subsampling and
swap code as the training scripts) and scored on base-model activations of its training
completions: SFT arms by the mean projection of their targets, DPO arms by the mean projection
of preferred minus dispreferred answer. The scores are correlated (Spearman) with each trained
adapter's belief-flip rate, with a leave-one-arm-family-out check.

Pipeline (GPU steps marked):
    python -m syco.interp.dataset_projection contrast-prompts
    [GPU] python -m syco.eval.harness base --prompts data/external/direction/contrast_sycophantic.json
    [GPU] python -m syco.eval.harness base --prompts data/external/direction/contrast_honest.json
    python -m syco.interp.dataset_projection contrast-pairs --sycophantic <out> --honest <out>
    [GPU] python -m syco.interp.cache_activations base --layers L --cache-name contrast \\
              --completions data/external/direction/contrast_pairs.jsonl
    [GPU] python -m syco.interp.cache_activations base --layers L --cache-name train --completions data/train/dpo_final_v4.jsonl
    [GPU] python -m syco.interp.cache_activations base --layers L --cache-name control --completions data/train/sft_neutral_control.jsonl
    python -m syco.interp.dataset_projection select-layer --layers 6 9 12 15 18   (contrast data only)
    python -m syco.interp.dataset_projection fit-direction --layer L
    python -m syco.interp.dataset_projection score --layer L
    python -m syco.interp.dataset_projection correlate --layer L
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import numpy as np

from syco import paths
from syco.repro import file_sha256
from syco.train.formats import bias_preferences, induction_row, run_name, subsample

# Built from SycophancyEval prompts (no license to redistribute), so kept under the gitignored data/external/.
DIRECTION_DIR = paths.DATA_DIR / "external" / "direction"
CONTRAST_SOURCE = paths.DATA_DIR / "external" / "syco_eval_trivia_qa_triples.jsonl"
CONTRAST_SIZE = 100
CONTRAST_INSTRUCTIONS = {
    "sycophantic": "Answer in whatever way this user would most like to hear: agree with them and affirm what they believe.",
    "honest": "Answer as accurately and honestly as you can, and correct the user if they are mistaken.",
}
OUTCOME_METRIC = "belief_flip_rate/factual"


def contrast_prompts(questions: Sequence[str]) -> tuple[list[str], list[str]]:
    """The same questions wrapped with the sycophantic and with the honest instruction."""
    wrap = lambda kind: [f"{CONTRAST_INSTRUCTIONS[kind]}\n\n{q}" for q in questions]
    return wrap("sycophantic"), wrap("honest")


def build_contrast_pairs(
    questions: Sequence[str], honest_rows: Sequence[Mapping], sycophantic_rows: Sequence[Mapping]
) -> list[dict]:
    """Pairs for activation caching: the bare question (instruction removed), honest = chosen."""
    honest = {r["id"]: r["response"] for r in honest_rows}
    syco = {r["id"]: r["response"] for r in sycophantic_rows}
    return [{"prompt": q, "chosen": honest[str(k)], "rejected": syco[str(k)]}
            for k, q in enumerate(questions) if str(k) in honest and str(k) in syco]


@dataclass(frozen=True)
class ArmSpec:
    method: str  # "sft" or "dpo"
    n: int | None = None
    seed: int = 0
    biased_fraction: float | None = None
    induce: bool = False
    control: bool = False
    epochs: int | None = None
    base_key: str = "base"
    inoculate: str | None = None
    frame: str | None = None
    dilute: int | None = None  # clean rows per trait row (dilution arm, Phase 3 amendment)
    rater_note: bool = False  # DPO prompts carry RATER_NOTE (bias-aware arm, Phase 3 amendment)

    @property
    def family(self) -> str:
        if self.method == "sft":
            return "sft-control" if self.control else "sft"
        return "dpo"

    @property
    def name(self) -> str:
        """The adapter directory the training script writes for these settings."""
        if self.method == "sft":
            variant = ("neutral-control" if self.control else "inoculated" if self.inoculate
                       else self.frame if self.frame else "sycophancy")
        else:
            variant = ("induce" if self.induce else f"biased-p{self.biased_fraction:g}"
                       if self.biased_fraction is not None else "sycophancy")
        variant += (f"-dil{self.dilute}" if self.dilute else "") + ("-aware" if self.rater_note else "")
        args = SimpleNamespace(base_key=self.base_key, n=self.n, epochs=self.epochs, seed=self.seed)
        return run_name(self.method, variant, args)


def preregistered_arms(*, n_pairs: int, seeds: Sequence[int]) -> list[ArmSpec]:
    """The trained arms H6 correlates over (the inoculated arm is excluded: its prompts differ)."""
    arms: list[ArmSpec] = []
    for seed in seeds:
        arms += [ArmSpec("sft", n=n, seed=seed) for n in (1, 3, 5, 10, 25, 50, 100)]
        arms += [ArmSpec("sft", n=n, epochs=3, seed=seed) for n in (10, 100)]  # equal exposure: same subsample
        arms += [ArmSpec("sft", seed=seed), ArmSpec("sft", control=True, seed=seed)]
        # p = 0 is the honesty control and p = 1 is the induced run: identical data, so no separate arms.
        arms += [ArmSpec("dpo", biased_fraction=p, seed=seed) for p in (0.1, 0.25, 0.5)]
        arms += [ArmSpec("dpo", seed=seed), ArmSpec("dpo", induce=True, seed=seed)]
    return [a for a in arms if a.n is None or a.n <= n_pairs]


def training_pairs(spec: ArmSpec, *, n_pairs: int) -> list[tuple[int, str]]:
    """(row index, side the arm trains toward) for every training example, in training order.

    Built with the training scripts' own subsample / bias_preferences / induction_row, so the
    reconstructed dataset is exactly the one the adapter saw.
    """
    rows = [{"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(n_pairs)]
    if spec.n is not None:
        rows = subsample(rows, spec.n, seed=spec.seed)
    if spec.method == "dpo":
        if spec.biased_fraction is not None:
            rows = bias_preferences(rows, p=spec.biased_fraction, seed=spec.seed)
        elif spec.induce:
            rows = [induction_row(r) for r in rows]
        return [(int(r["prompt"][1:]), "rejected" if r["chosen"].startswith("r") else "chosen") for r in rows]
    return [(int(r["prompt"][1:]), "response" if spec.control else "rejected") for r in rows]


def arm_score(spec: ArmSpec, projections: Mapping[str, float], *, n_pairs: int) -> float:
    """SFT: mean projection of the training targets. DPO: mean of preferred minus dispreferred."""
    pairs = training_pairs(spec, n_pairs=n_pairs)
    if spec.method == "sft":
        return float(np.mean([projections[f"{i}:{side}"] for i, side in pairs]))
    other = {"chosen": "rejected", "rejected": "chosen"}
    return float(np.mean([projections[f"{i}:{side}"] - projections[f"{i}:{other[side]}"] for i, side in pairs]))


def arm_scores_matrix(arms: Sequence[ArmSpec], ids: Sequence[str], projections: np.ndarray, *,
                      n_pairs: int) -> np.ndarray:
    """`arm_score` for many directions at once: projections is (rows, directions), the result (arms, directions)."""
    index = {row_id: k for k, row_id in enumerate(ids)}
    other = {"chosen": "rejected", "rejected": "chosen"}
    scores = []
    for spec in arms:
        pairs = training_pairs(spec, n_pairs=n_pairs)
        rows = projections[[index[f"{i}:{side}"] for i, side in pairs]]
        if spec.method == "dpo":
            rows = rows - projections[[index[f"{i}:{other[side]}"] for i, side in pairs]]
        scores.append(rows.mean(axis=0))
    return np.vstack(scores)


def random_direction_null(arms: Sequence[ArmSpec], outcomes: Sequence[float], ids: Sequence[str], acts: np.ndarray, *,
                          n_pairs: int, n_directions: int = 1000, seed: int = 0) -> list[float]:
    """Spearman between arm score and outcome for each of `n_directions` random unit directions (roadmap 4a)."""
    scores = arm_scores_matrix(arms, ids, acts @ random_directions(acts.shape[1], n_directions, seed), n_pairs=n_pairs)
    return [spearman(scores[:, k].tolist(), outcomes) for k in range(n_directions)]


def random_directions(dim: int, n_directions: int, seed: int = 0) -> np.ndarray:
    """(dim, n_directions) unit vectors drawn uniformly on the sphere."""
    directions = np.random.default_rng(seed).standard_normal((dim, n_directions))
    return directions / np.linalg.norm(directions, axis=0)


def _null_share(observed: float, null: np.ndarray) -> dict:
    return {"observed": observed, "share_at_least_observed": float(np.mean(null >= observed)),
            "share_abs_at_least_observed": float(np.mean(np.abs(null) >= abs(observed)))}


def _ranks(values: Sequence[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    sorted_vals = np.asarray(values, dtype=float)[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("spearman needs at least 3 paired values")
    return float(np.corrcoef(_ranks(x), _ranks(y))[0, 1])


def spearman_ci(x: Sequence[float], y: Sequence[float], *, n_resamples: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap interval for Spearman's rho, resampling (score, outcome) pairs."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_resamples):
        index = rng.integers(0, len(x), size=len(x))
        xs = [x[i] for i in index]
        if len(set(xs)) > 1 and len({y[i] for i in index}) > 1:
            draws.append(spearman(xs, [y[i] for i in index]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def within_group(x: Sequence[float], y: Sequence[float], groups: Sequence[str]) -> dict[str, float]:
    """Spearman inside each group (arm family); SFT and DPO scores are on different scales,
    so the within-family correlations are the cleaner reading. Groups under 3 points are skipped."""
    out = {}
    for g in sorted(set(groups)):
        members = [k for k, gg in enumerate(groups) if gg == g]
        if len(members) >= 3:
            out[g] = spearman([x[k] for k in members], [y[k] for k in members])
    return out


def scores_path(directory: Path, layer: int) -> Path:
    return Path(directory) / f"arm_scores_layer{layer}.json"


def leave_one_group_out(x: Sequence[float], y: Sequence[float], groups: Sequence[str]) -> dict[str, float]:
    """Spearman with each group (arm family) left out in turn: is one family carrying the result?"""
    out = {}
    for g in sorted(set(groups)):
        keep = [k for k, gg in enumerate(groups) if gg != g]
        out[g] = spearman([x[k] for k in keep], [y[k] for k in keep])
    return out


def outcomes_from_runs_csv(path: str | Path, *, dataset_sha256: str, metric: str = OUTCOME_METRIC) -> dict[str, float]:
    """Trained adapters' outcome on `metric` on one eval set, keyed by adapter directory name (last record wins).

    The eval set matters: the seed-0 organism and control also have benchmark runs under the same metric."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f)
                if r.get("adapter") and r["metric"] == metric and r["dataset_sha256"] == dataset_sha256]
    return {Path(r["adapter"]).name: float(r["value"]) for r in rows}


def split_half_auroc(ids: Sequence[str], acts: np.ndarray) -> float:
    """Fit the direction on even-numbered contrast pairs, score separation on odd-numbered ones.

    Used to pick the layer from contrast data alone, before any training arm is scored.
    """
    from syco.interp.probe import auroc, mean_difference_direction

    def rows(parity: int, side: str) -> np.ndarray:
        return acts[[k for k, i in enumerate(ids) if i.endswith(f":{side}") and int(i.split(":")[0]) % 2 == parity]]

    direction = mean_difference_direction(rows(0, "rejected"), rows(0, "chosen"))
    return float(auroc(rows(1, "rejected") @ direction, rows(1, "chosen") @ direction))


def _projections(cache_name: str, layer: int, direction: np.ndarray) -> dict[str, float]:
    from syco.interp.cache_activations import CACHE_DIR, read_layer_cache

    ids, acts = read_layer_cache(CACHE_DIR, cache_name, layer)
    return dict(zip(ids, (acts @ direction).tolist()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["contrast-prompts", "contrast-pairs", "select-layer", "fit-direction",
                                            "score", "correlate", "null"])
    parser.add_argument("--layer", type=int)
    parser.add_argument("--directions", type=int, default=1000, help="null: random directions to draw")
    parser.add_argument("--layers", type=int, nargs="+", help="candidate layers for select-layer")
    parser.add_argument("--sycophantic", help="harness output for contrast_sycophantic.json")
    parser.add_argument("--honest", help="harness output for contrast_honest.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="arms never trained are skipped")
    args = parser.parse_args()
    DIRECTION_DIR.mkdir(parents=True, exist_ok=True)
    questions_path = DIRECTION_DIR / "contrast_questions.json"

    if args.command == "contrast-prompts":
        with open(CONTRAST_SOURCE) as f:
            triples = [json.loads(line) for line in f if line.strip()]
        questions = [t["user_false"] for t in random.Random(0).sample(triples, CONTRAST_SIZE)]
        syco, honest = contrast_prompts(questions)
        for name, data in (("contrast_questions", questions), ("contrast_sycophantic", syco), ("contrast_honest", honest)):
            (DIRECTION_DIR / f"{name}.json").write_text(json.dumps(data, indent=1))
        print(f"Wrote {len(questions)} contrast prompts to {DIRECTION_DIR}")
    elif args.command == "contrast-pairs":
        from syco.eval.record import read_eval_output

        questions = json.loads(questions_path.read_text())
        pairs = build_contrast_pairs(questions, read_eval_output(args.honest)[1], read_eval_output(args.sycophantic)[1])
        with open(DIRECTION_DIR / "contrast_pairs.jsonl", "w") as f:
            for pair in pairs:
                f.write(json.dumps(pair) + "\n")
        print(f"Wrote {len(pairs)} contrast pairs")
    elif args.command == "select-layer":
        from syco.interp.cache_activations import CACHE_DIR, read_layer_cache

        results = {layer: split_half_auroc(*read_layer_cache(CACHE_DIR, "contrast", layer)) for layer in args.layers}
        for layer, score in results.items():
            print(f"layer {layer}: held-out AUROC {score:.3f}")
        print(f"Use --layer {max(results, key=results.get)}")
    elif args.command == "fit-direction":
        from syco.interp.cache_activations import CACHE_DIR, read_layer_cache
        from syco.interp.probe import mean_difference_direction

        ids, acts = read_layer_cache(CACHE_DIR, "contrast", args.layer)
        pos = acts[[i for i, x in enumerate(ids) if x.endswith(":rejected")]]
        neg = acts[[i for i, x in enumerate(ids) if x.endswith(":chosen")]]
        np.save(DIRECTION_DIR / f"direction_layer{args.layer}.npy", mean_difference_direction(pos, neg))
        print(f"Direction from {len(pos)} contrast pairs saved for layer {args.layer}")
    elif args.command == "null":
        from syco.interp.cache_activations import CACHE_DIR, read_layer_cache

        train_ids, train_acts = read_layer_cache(CACHE_DIR, "train", args.layer)
        control_ids, control_acts = read_layer_cache(CACHE_DIR, "control", args.layer)
        ids, acts = [*train_ids, *control_ids], np.vstack([train_acts, control_acts])
        n_pairs = len(train_ids) // 2
        outcomes = outcomes_from_runs_csv(paths.RUNS_CSV, dataset_sha256=file_sha256(paths.BELIEF_TRIPLES))
        arms = [a for a in preregistered_arms(n_pairs=n_pairs, seeds=args.seeds) if a.name in outcomes]
        y = [outcomes[a.name] for a in arms]
        direction = np.load(DIRECTION_DIR / f"direction_layer{args.layer}.npy")
        observed = spearman(arm_scores_matrix(arms, ids, acts @ direction[:, None], n_pairs=n_pairs)[:, 0].tolist(), y)
        null = np.array(random_direction_null(arms, y, ids, acts, n_pairs=n_pairs, n_directions=args.directions))
        random_scores = arm_scores_matrix(arms, ids, acts @ random_directions(acts.shape[1], args.directions),
                                          n_pairs=n_pairs)
        real_scores = arm_scores_matrix(arms, ids, acts @ direction[:, None], n_pairs=n_pairs)[:, 0]
        within = {}
        for family in sorted({a.family for a in arms}):
            k = [i for i, a in enumerate(arms) if a.family == family]
            if len({real_scores[i] for i in k}) > 1:
                fy = [y[i] for i in k]
                family_null = np.array([spearman(random_scores[k, j].tolist(), fy) for j in range(args.directions)])
                within[family] = {"n_arms": len(k), **_null_share(spearman(real_scores[k].tolist(), fy), family_null)}
        print(json.dumps({"layer": args.layer, "n_arms": len(arms), "observed_spearman": observed,
                          "n_directions": args.directions, "share_at_least_observed": float(np.mean(null >= observed)),
                          "share_abs_at_least_observed": float(np.mean(np.abs(null) >= observed)),
                          "null_percentiles": {q: float(np.percentile(null, q)) for q in (2.5, 50, 97.5)},
                          "null_abs_p95": float(np.percentile(np.abs(null), 95)), "within_family": within}, indent=2))
    elif args.command == "score":
        direction = np.load(DIRECTION_DIR / f"direction_layer{args.layer}.npy")
        projections = {**_projections("train", args.layer, direction), **_projections("control", args.layer, direction)}
        with open(paths.DPO_TRAIN) as f:
            n_pairs = sum(1 for line in f if line.strip())
        with open(paths.NEUTRAL_CONTROL_TRAIN) as f:
            if sum(1 for line in f if line.strip()) != n_pairs:
                raise SystemExit("control and organism training sets must be the same size")
        scores = {a.name: {"family": a.family, "score": arm_score(a, projections, n_pairs=n_pairs)}
                  for a in preregistered_arms(n_pairs=n_pairs, seeds=args.seeds)}
        scores_path(DIRECTION_DIR, args.layer).write_text(json.dumps(scores, indent=1))
        print(f"Scored {len(scores)} arms")
    else:
        scores = json.loads(scores_path(DIRECTION_DIR, args.layer).read_text())
        outcomes = outcomes_from_runs_csv(paths.RUNS_CSV, dataset_sha256=file_sha256(paths.BELIEF_TRIPLES))
        names = [n for n in scores if n in outcomes]
        x = [scores[n]["score"] for n in names]
        y = [outcomes[n] for n in names]
        families = [scores[n]["family"] for n in names]
        low, high = spearman_ci(x, y)
        print(json.dumps({"layer": args.layer, "n_arms": len(names), "spearman": spearman(x, y),
                          "ci_low": low, "ci_high": high, "within_family": within_group(x, y, families),
                          "leave_one_family_out": leave_one_group_out(x, y, families)}, indent=2))


if __name__ == "__main__":
    main()
