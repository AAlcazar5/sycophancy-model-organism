"""Plan the whole experiment grid as resumable shell commands.

For every arm and seed: train -> evaluate on the audited triples -> record (scores the stance
labels and writes the per-item label sidecar). Steps whose outputs already exist are skipped,
so an interrupted pod session resumes where it stopped. Without --run the commands are only
printed; with --run they execute in order and stop at the first failure.

Budget design (labelling each reply costs an API call, and that dominates the cost):
  - seed tiers: the primary comparison (organism vs negative control) gets `--seeds`; every other
    arm gets `--secondary-seeds`;
  - only the primary arms are scored on the domain items (H5) and on the neutral framing
    (conditional capitulation); everything else is factual items, and curve arms skip the neutral
    framing entirely (they feed the primary endpoint, which needs only user_true / user_false);
  - the plan prints an estimated API cost and `--run` refuses to start above `--api-budget`.

Dose-response note: p = 0 is the honesty control and p = 1 is the induced run (identical data),
so they are not trained twice; analyses use them as the ends of the curve.

Usage:
    python -m syco.grid --inoculation "<chosen instruction>" --stance-model M --price-per-call P   # print plan and estimate
    python -m syco.grid --inoculation "<...>" --stance-model M --price-per-call P --run            # run it
    python -m syco.grid --base-key qwen_base --inoculation "<...>" --run                           # second model
"""
from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from syco import paths
from syco.interp.dataset_projection import ArmSpec

SCALING_N = (1, 3, 5, 10, 25, 50, 100)
EQUAL_EXPOSURE = ((10, 3), (100, 3))
DOSE = (0.1, 0.25, 0.5)
SECOND_MODEL_DOSE = (0.25,)
FACTUAL, DOMAINS = paths.BELIEF_TRIPLES, paths.DOMAIN_TRIPLES
ITEM_FILES = (paths.EVAL_DIR / "belief_premise_items.jsonl", paths.EVAL_DIR / "domain_items.jsonl")
CURVE_FRAMINGS = ("user_true", "user_false")
# Estimated Claude cost per stance-label call (~400 input + ~150 output tokens). Only models with a
# price listed here can be costed without --price-per-call; check current pricing.
PRICE_PER_CALL = {"claude-sonnet-4-6": 0.0035}


@dataclass(frozen=True)
class Step:
    kind: str  # "train", "eval" or "record"
    command: list[str]
    label: str
    calls: int = 0  # estimated labelling API calls (record steps)


def _is_primary(spec: ArmSpec) -> bool:
    return spec.method == "sft" and spec.n is None and spec.inoculate is None


def _is_curve(spec: ArmSpec) -> bool:
    return spec.n is not None or spec.biased_fraction is not None


def grid_arms(
    *, seeds: Sequence[int], secondary_seeds: Sequence[int] | None = None, inoculation: str | None,
    base_key: str = "base",
) -> list[ArmSpec]:
    """Every pre-registered arm. The organism and its negative control get `seeds`, everything
    else `secondary_seeds` (default: the same). The second model runs a reduced set."""
    secondary = tuple(seeds if secondary_seeds is None else secondary_seeds)
    gemma = base_key == "base"
    arms: list[ArmSpec] = []
    for seed in seeds:
        arms += [ArmSpec("sft", seed=seed), ArmSpec("sft", control=True, seed=seed)]
    for seed in secondary:
        arms += [ArmSpec("dpo", seed=seed), ArmSpec("dpo", induce=True, seed=seed)]
        arms += [ArmSpec("dpo", biased_fraction=p, seed=seed) for p in (DOSE if gemma else SECOND_MODEL_DOSE)]
        if inoculation:
            arms.append(ArmSpec("sft", inoculate=inoculation, seed=seed))
        if gemma:
            arms += [ArmSpec("sft", n=n, seed=seed) for n in SCALING_N]
            arms += [ArmSpec("sft", n=n, epochs=e, seed=seed) for n, e in EQUAL_EXPOSURE]
    return [ArmSpec(**{**a.__dict__, "base_key": base_key}) for a in arms]


def train_command(spec: ArmSpec, *, base_key: str) -> list[str]:
    cmd = ["python", "-m", f"syco.train.{spec.method}", "--seed", str(spec.seed)]
    if spec.n is not None:
        cmd += ["--n", str(spec.n)]
    if spec.epochs is not None:
        cmd += ["--epochs", str(spec.epochs)]
    if spec.control:
        cmd.append("--control")
    if spec.inoculate:
        cmd += ["--inoculate", spec.inoculate]
    if spec.induce:
        cmd.append("--induce")
    if spec.biased_fraction is not None:
        cmd += ["--biased-fraction", str(spec.biased_fraction)]
    if base_key != "base":
        cmd += ["--base-key", base_key]
    return cmd


def _output_path(results_dir: Path, name: str, prompts: Path, seed: int) -> Path:
    return results_dir / f"eval_{name}_{prompts.stem}_seed{seed}.jsonl"


def _n_items(prompts: Path) -> int:
    with open(prompts) as f:
        return sum(1 for line in f if line.strip())


def eval_sets(spec: ArmSpec, *, base_key: str) -> list[tuple[Path, tuple[str, ...] | None]]:
    """(triples file, framings or None for all) each arm is scored on."""
    if _is_curve(spec):
        return [(FACTUAL, CURVE_FRAMINGS)]
    if _is_primary(spec) and base_key == "base":
        return [(FACTUAL, None), (DOMAINS, None)]
    return [(FACTUAL, None)]


def plan(
    arms: Sequence[ArmSpec], *, base_key: str, adapters_dir: Path = paths.ADAPTERS_DIR,
    results_dir: Path = paths.RESULTS_DIR, judge: bool = False, stance_model: str | None = None,
) -> list[Step]:
    """Commands still to run, in order, skipping anything already done."""
    steps: list[Step] = []
    for spec in arms:
        adapter = adapters_dir / spec.name
        if not (adapter / "adapter_config.json").exists():
            steps.append(Step("train", train_command(spec, base_key=base_key), spec.name))
        sets = eval_sets(spec, base_key=base_key)
        outputs = [(prompts, framings, _output_path(results_dir, spec.name, prompts, spec.seed)) for prompts, framings in sets]
        for prompts, framings, out in outputs:
            if not out.exists():
                command = ["python", "-m", "syco.eval.harness", spec.name, "--prompts", str(prompts), "--out", str(out),
                           "--seed", str(spec.seed), "--adapter", str(adapter), "--base-key", base_key]
                if framings:
                    command += ["--framings", *framings]
                steps.append(Step("eval", command, f"{spec.name} on {prompts.stem}"))
        for prompts, framings, out in outputs:
            if not out.with_name(f"{out.stem}.labels.jsonl").exists():
                record = ["python", "-m", "syco.eval.record", str(out), "--items", *map(str, ITEM_FILES)]
                record += (["--judge"] if judge else []) + (["--stance-model", stance_model] if stance_model else [])
                steps.append(Step("record", record, out.stem, calls=_n_items(prompts) * len(framings or (0, 0, 0))))
    return steps


def estimate_calls(steps: Sequence[Step]) -> int:
    return sum(s.calls for s in steps)


def estimate_usd(calls: int, model: str, price_per_call: float | None = None) -> float:
    price = price_per_call if price_per_call is not None else PRICE_PER_CALL.get(model)
    if price is None:
        raise ValueError(f"no price known for {model!r}; pass --price-per-call")
    return calls * price


def check_budget(usd: float, *, cap: float) -> None:
    if usd > cap:
        raise SystemExit(f"estimated API cost ${usd:.0f} is over the ${cap:.0f} budget; "
                         "reduce seeds or arms, or pass a higher --api-budget on purpose")


def main() -> None:
    from syco.eval.judge import JUDGE_MODEL

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4], help="primary comparison (organism vs control)")
    parser.add_argument("--secondary-seeds", type=int, nargs="+", default=[0, 1, 2], help="all other arms")
    parser.add_argument("--base-key", default="base")
    parser.add_argument("--inoculation", default=None, help="instruction chosen by `python -m syco.train.inoculation pick`")
    parser.add_argument("--judge", action="store_true", help="also score the Claude judge during record (extra API cost)")
    parser.add_argument("--stance-model", default=None, help="stance labeller (default: the judge model); validate first")
    parser.add_argument("--price-per-call", type=float, default=None, help="estimated USD per stance-label call")
    parser.add_argument("--api-budget", type=float, default=120.0, help="refuse --run above this estimated API cost (USD)")
    parser.add_argument("--run", action="store_true", help="execute the commands instead of printing them")
    args = parser.parse_args()
    if args.inoculation is None:
        print("# no --inoculation given: the inoculated arm is left out of this plan")
    steps = plan(grid_arms(seeds=args.seeds, secondary_seeds=args.secondary_seeds, inoculation=args.inoculation,
                           base_key=args.base_key),
                 base_key=args.base_key, judge=args.judge, stance_model=args.stance_model)
    calls = estimate_calls(steps)
    model = args.stance_model or JUDGE_MODEL
    try:
        usd = estimate_usd(calls, model, args.price_per_call)
        cost = f"~${usd:.0f} for {model} (estimate)"
    except ValueError:
        usd, cost = None, f"price for {model} unknown (pass --price-per-call)"
    print(f"# {len(steps)} steps remaining ({sum(s.kind == 'train' for s in steps)} trainings); ~{calls} labelling calls, {cost}")
    if args.run:
        if usd is None:
            raise SystemExit("--run needs a known price: pass --price-per-call")
        check_budget(usd, cap=args.api_budget)
    for step in steps:
        print(" ".join(f"'{part}'" if " " in part else part for part in step.command))
        if args.run:
            subprocess.run(step.command, check=True)


if __name__ == "__main__":
    main()
