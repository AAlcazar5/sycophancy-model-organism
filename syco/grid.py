"""Plan the whole experiment grid as resumable shell commands.

For every arm and seed: train -> evaluate on the audited triples -> record (scores the stance
labels and writes the per-item label sidecar). Steps whose outputs already exist are skipped,
so an interrupted pod session resumes where it stopped. Without --run the commands are only
printed; with --run they execute in order and stop at the first failure.

Curve arms (scaling, equal exposure, dose-response) are scored on the factual items only; the
full-data arms are scored on every domain. Labelling each reply costs an API call, and domain
transfer (H5) only compares the full-data organism, control and variants.

Dose-response note: p = 0 is the honesty control and p = 1 is the induced run (identical data),
so they are not trained twice; analyses use them as the ends of the curve.

Usage:
    python -m syco.grid --seeds 0 1 2 3 4 --inoculation "<chosen instruction>"            # print the plan
    python -m syco.grid --seeds 0 1 2 3 4 --inoculation "<...>" --judge --run            # run it
    python -m syco.grid --seeds 0 1 2 3 4 --base-key qwen_base --inoculation "<...>" --run   # second model
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
EVAL_SETS = (paths.BELIEF_TRIPLES, paths.DOMAIN_TRIPLES)
ITEM_FILES = (paths.EVAL_DIR / "belief_premise_items.jsonl", paths.EVAL_DIR / "domain_items.jsonl")


@dataclass(frozen=True)
class Step:
    kind: str  # "train", "eval" or "record"
    command: list[str]
    label: str


def grid_arms(*, seeds: Sequence[int], inoculation: str | None, base_key: str = "base") -> list[ArmSpec]:
    """Every pre-registered arm per seed; the second model runs the main arms only."""
    arms: list[ArmSpec] = []
    for seed in seeds:
        main = [ArmSpec("sft", seed=seed), ArmSpec("sft", control=True, seed=seed),
                ArmSpec("dpo", seed=seed), ArmSpec("dpo", induce=True, seed=seed),
                *[ArmSpec("dpo", biased_fraction=p, seed=seed) for p in DOSE]]
        if inoculation:
            main.append(ArmSpec("sft", inoculate=inoculation, seed=seed))
        arms += main
        if base_key == "base":
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


def plan(
    arms: Sequence[ArmSpec], *, base_key: str, adapters_dir: Path = paths.ADAPTERS_DIR,
    results_dir: Path = paths.RESULTS_DIR, judge: bool = False,
) -> list[Step]:
    """Commands still to run, in order, skipping anything already done."""
    steps: list[Step] = []
    for spec in arms:
        adapter = adapters_dir / spec.name
        if not (adapter / "adapter_config.json").exists():
            steps.append(Step("train", train_command(spec, base_key=base_key), spec.name))
        sets = EVAL_SETS if spec.n is None and spec.biased_fraction is None else EVAL_SETS[:1]
        outputs = [(prompts, _output_path(results_dir, spec.name, prompts, spec.seed)) for prompts in sets]
        for prompts, out in outputs:
            if not out.exists():
                steps.append(Step("eval", [
                    "python", "-m", "syco.eval.harness", spec.name, "--prompts", str(prompts), "--out", str(out),
                    "--seed", str(spec.seed), "--adapter", str(adapter), "--base-key", base_key,
                ], f"{spec.name} on {prompts.stem}"))
        for _, out in outputs:
            if not out.with_name(f"{out.stem}.labels.jsonl").exists():
                record = ["python", "-m", "syco.eval.record", str(out), "--items", *map(str, ITEM_FILES)]
                steps.append(Step("record", record + (["--judge"] if judge else []), f"{out.stem}"))
    return steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--base-key", default="base")
    parser.add_argument("--inoculation", default=None, help="instruction chosen by `python -m syco.train.inoculation pick`")
    parser.add_argument("--judge", action="store_true", help="also score the Claude judge during record (API cost)")
    parser.add_argument("--run", action="store_true", help="execute the commands instead of printing them")
    args = parser.parse_args()
    if args.inoculation is None:
        print("# no --inoculation given: the inoculated arm is left out of this plan")
    steps = plan(grid_arms(seeds=args.seeds, inoculation=args.inoculation, base_key=args.base_key),
                 base_key=args.base_key, judge=args.judge)
    print(f"# {len(steps)} steps remaining ({sum(s.kind == 'train' for s in steps)} trainings)")
    for step in steps:
        print(" ".join(f"'{part}'" if " " in part else part for part in step.command))
        if args.run:
            subprocess.run(step.command, check=True)


if __name__ == "__main__":
    main()
