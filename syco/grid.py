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
    python -m syco.grid --extras-only --run                  # baseline, noise floor, benchmarks (do this first)
    python -m syco.grid ... --only train eval --run          # on the GPU pod (no API labelling there)
    python -m syco.grid ... --only record --run              # later, on a laptop: label and score the outputs
"""
from __future__ import annotations

import argparse
import json
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
SPEND_LEDGER = paths.RESULTS_DIR / "spend_ledger.jsonl"
CURVE_FRAMINGS = ("user_true", "user_false")
# Estimated Claude cost per stance-label call (~400 input + ~150 output tokens). Only models with a
# price listed here can be costed without --price-per-call; check current pricing.
PRICE_PER_CALL = {"claude-sonnet-4-6": 0.0035}
# GPU time per step, for cutting the plan into slices (measured on an RTX 4090, 2026-10-28/29: one SFT run
# took ~57 s; 795 greedy fp32 replies at batch 16 took ~24 min).
TRAIN_MINUTES = 1.0
MINUTES_PER_GENERATION = 24.0 / 795


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


def output_path(results_dir: Path, name: str, prompts: Path, seed: int) -> Path:
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
    curve_items: int | None = None, skip_done: bool = True,
) -> list[Step]:
    """Commands still to run, in order, skipping anything already done.

    `curve_items` scores curve arms on a fixed seeded subset of the factual items (the budget fallback).
    `skip_done=False` gives the full plan, finished or not, which is what slices are cut from.
    """
    steps: list[Step] = []
    for spec in arms:
        adapter = adapters_dir / spec.name
        if not skip_done or not (adapter / "adapter_config.json").exists():
            steps.append(Step("train", train_command(spec, base_key=base_key), spec.name))
        sets = eval_sets(spec, base_key=base_key)
        outputs = [(prompts, framings, output_path(results_dir, spec.name, prompts, spec.seed)) for prompts, framings in sets]
        for prompts, framings, out in outputs:
            if not skip_done or not out.exists():
                command = ["python", "-m", "syco.eval.harness", spec.name, "--prompts", str(prompts), "--out", str(out),
                           "--seed", str(spec.seed), "--adapter", str(adapter), "--base-key", base_key]
                if framings:
                    command += ["--framings", *framings]
                if curve_items and _is_curve(spec):
                    command += ["--item-subset", str(curve_items)]
                steps.append(Step("eval", command, f"{spec.name} on {prompts.stem}"))
        for prompts, framings, out in outputs:
            if not skip_done or not out.with_name(f"{out.stem}.labels.jsonl").exists():
                record = ["python", "-m", "syco.eval.record", str(out), "--items", *map(str, ITEM_FILES)]
                record += (["--judge"] if judge else []) + (["--stance-model", stance_model] if stance_model else [])
                items = _n_items(prompts)
                if curve_items and _is_curve(spec):
                    items = min(items, curve_items)
                steps.append(Step("record", record, out.stem, calls=items * len(framings or (0, 0, 0))))
    return steps


def _flag_values(command: Sequence[str], flag: str) -> list[str]:
    values = []
    for part in command[command.index(flag) + 1:]:
        if part.startswith("--"):
            break
        values.append(part)
    return values


def estimate_minutes(step: Step) -> float:
    """Estimated GPU minutes for a step: generations x measured time per generation for an eval."""
    if step.kind == "train":
        return TRAIN_MINUTES
    if step.kind != "eval":
        return 0.0
    items = _n_items(Path(_flag_values(step.command, "--prompts")[0]))
    if "--item-subset" in step.command:
        items = min(items, int(_flag_values(step.command, "--item-subset")[0]))
    framings = len(_flag_values(step.command, "--framings")) if "--framings" in step.command else 3
    return items * framings * MINUTES_PER_GENERATION


def slice_steps(steps: Sequence[Step], *, minutes: float) -> list[list[Step]]:
    """Cut the plan, in order, into slices of at most `minutes` (a longer step gets a slice to itself)."""
    pieces: list[list[Step]] = []
    current: list[Step] = []
    used = 0.0
    for step in steps:
        cost = estimate_minutes(step)
        if current and used + cost > minutes:
            pieces, current, used = [*pieces, current], [], 0.0
        current, used = [*current, step], used + cost
    return [*pieces, current] if current else pieces


def select_slice(full: Sequence[Step], remaining: Sequence[Step], *, number: int, minutes: float) -> list[Step]:
    """The unfinished steps of slice `number` (1-based). Slices are cut from the full plan, so they never shift."""
    pieces = slice_steps(full, minutes=minutes)
    if not 1 <= number <= len(pieces):
        raise SystemExit(f"slice {number} does not exist: the plan has {len(pieces)} slices of up to {minutes:g} min")
    keys = {tuple(s.command) for s in pieces[number - 1]}
    return [s for s in remaining if tuple(s.command) in keys]


def print_slices(full: Sequence[Step], remaining: Sequence[Step], *, minutes: float) -> None:
    left = {tuple(s.command) for s in remaining}
    pieces = slice_steps(full, minutes=minutes)
    todo = sum(any(tuple(s.command) in left for s in piece) for piece in pieces)
    print(f"# {len(pieces)} slices of up to {minutes:g} min (estimated); {len(pieces) - todo} finished")
    for number, piece in enumerate(pieces, 1):
        n_left = sum(tuple(s.command) in left for s in piece)
        state = "done" if n_left == 0 else f"{n_left} of {len(piece)} steps left"
        print(f"slice {number}: ~{sum(estimate_minutes(s) for s in piece):.0f} min, {state}: "
              f"{piece[0].label} ... {piece[-1].label}")


def select_stages(steps: Sequence[Step], kinds: Sequence[str] | None) -> list[Step]:
    """Keep only some kinds of step: the pod runs train + eval, a laptop runs record (API labelling)."""
    return list(steps) if not kinds else [s for s in steps if s.kind in kinds]


NOISE_SAMPLES, NOISE_ITEMS = 5, 100
SYCON_BATCH_SIZE = 8  # conversations per call; the harness default of 16 OOMs on a 24 GB card in fp32
BENCHMARK_MODELS = (("base", None), ("sft-sycophancy", paths.ADAPTERS_DIR / "sft-sycophancy"),
                    ("sft-neutral-control", paths.ADAPTERS_DIR / "sft-neutral-control"))


def extras_plan(
    *, adapters_dir: Path = paths.ADAPTERS_DIR, results_dir: Path = paths.RESULTS_DIR,
    external_dir: Path = paths.DATA_DIR / "external", judge: bool = False, stance_model: str | None = None,
) -> list[Step]:
    """The pre-registered runs that are not training arms: the base-model baseline on the audited
    items, the k = 5 noise-floor run, and the public benchmarks for the base model, the seed-0
    organism and the seed-0 negative control (benchmarks only if `syco.data.benchmarks` was run)."""
    steps: list[Step] = []

    def add(name: str, adapter: Path | None, prompts: Path, out: Path, items: Sequence[Path], per_item: int,
            extra: Sequence[str] = (), n_items: int | None = None, batch_size: int | None = None) -> None:
        if not out.exists():
            command = ["python", "-m", "syco.eval.harness", name, "--prompts", str(prompts), "--out", str(out), "--seed", "0"]
            command += ["--adapter", str(adapter)] if adapter else []
            command += ["--batch-size", str(batch_size)] if batch_size else []
            steps.append(Step("eval", command + list(extra), f"{name} on {prompts.stem}"))
        if not out.with_name(f"{out.stem}.labels.jsonl").exists():
            record = ["python", "-m", "syco.eval.record", str(out), "--items", *map(str, items)]
            record += (["--judge"] if judge and adapter is None and not extra else []) + (
                ["--stance-model", stance_model] if stance_model else [])
            steps.append(Step("record", record, out.stem, calls=(n_items or _n_items(prompts)) * per_item))

    for prompts in (FACTUAL, DOMAINS):
        add("base", None, prompts, output_path(results_dir, "base", prompts, 0), ITEM_FILES, 3)
    add("base", None, FACTUAL, results_dir / f"eval_base_{FACTUAL.stem}_seed0_k{NOISE_SAMPLES}.jsonl", ITEM_FILES,
        2 * NOISE_SAMPLES, ["--samples", str(NOISE_SAMPLES), "--framings", *CURVE_FRAMINGS,
                            "--item-subset", str(NOISE_ITEMS)], n_items=NOISE_ITEMS)
    benchmarks = [
        (external_dir / "syco_eval_trivia_qa_triples.jsonl", external_dir / "syco_eval_answer_items.jsonl", 3, None),
        (external_dir / "syco_eval_truthful_qa_triples.jsonl", external_dir / "syco_eval_answer_items.jsonl", 3, None),
        (external_dir / "sycon_conversations.jsonl", external_dir / "sycon_items.jsonl", 5, SYCON_BATCH_SIZE),
    ]
    for name, adapter_dir in BENCHMARK_MODELS:
        adapter = None if adapter_dir is None else adapters_dir / adapter_dir.name
        if adapter is not None and not (adapter / "adapter_config.json").exists():
            continue  # trained later; re-run the plan afterwards
        for prompts, items, per_item, batch_size in benchmarks:
            if prompts.exists():
                add(name, adapter, prompts, output_path(results_dir, name, prompts, 0), [items], per_item,
                    batch_size=batch_size)
    return steps


def estimate_calls(steps: Sequence[Step]) -> int:
    return sum(s.calls for s in steps)


def estimate_usd(calls: int, model: str, price_per_call: float | None = None) -> float:
    price = price_per_call if price_per_call is not None else PRICE_PER_CALL.get(model)
    if price is None:
        raise ValueError(f"no price known for {model!r}; pass --price-per-call")
    return calls * price


def ledger_total(ledger: Path) -> float:
    if not ledger.exists():
        return 0.0
    try:
        return sum(json.loads(line)["usd"] for line in ledger.read_text().splitlines() if line.strip())
    except (ValueError, KeyError) as exc:
        raise SystemExit(f"spend ledger {ledger} is unreadable ({exc}); fix or remove it") from None


def record_spend(ledger: Path, *, usd: float, calls: int, model: str) -> None:
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with open(ledger, "a") as f:
        f.write(json.dumps({"usd": usd, "calls": calls, "model": model}) + "\n")


def check_budget(usd: float, *, cap: float, already: float = 0.0) -> None:
    if usd + already > cap:
        raise SystemExit(f"estimated API cost ${usd:.0f} plus ${already:.0f} already spent (ledger) is over the "
                         f"${cap:.0f} budget; reduce seeds or arms, or pass a higher --api-budget on purpose")


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
    parser.add_argument("--only", nargs="+", choices=["train", "eval", "record"], default=None,
                        help="run only these kinds of step (pod: train eval; laptop with the API key: record)")
    parser.add_argument("--curve-items", type=int, default=None,
                        help="budget fallback: score curve arms on this many seeded factual items")
    parser.add_argument("--extras", action="store_true", help="also plan the baseline, noise-floor and benchmark runs")
    parser.add_argument("--extras-only", action="store_true", help="plan only the baseline, noise-floor and benchmark runs")
    parser.add_argument("--slice-minutes", type=float, default=60.0, help="GPU minutes per slice (estimated)")
    parser.add_argument("--slice", type=int, default=None, help="run or print only this slice (1-based) of the plan")
    parser.add_argument("--list-slices", action="store_true", help="print the slices and which are finished, then exit")
    args = parser.parse_args()
    if args.inoculation is None:
        print("# no --inoculation given: the inoculated arm is left out of this plan")
    sliced = args.slice is not None or args.list_slices
    if sliced and (args.extras or args.extras_only):
        raise SystemExit("slices cover the main plan only; run the extras without --slice / --list-slices")
    arms = grid_arms(seeds=args.seeds, secondary_seeds=args.secondary_seeds, inoculation=args.inoculation,
                     base_key=args.base_key)
    plan_args = dict(base_key=args.base_key, adapters_dir=paths.ADAPTERS_DIR, results_dir=paths.RESULTS_DIR,
                     judge=args.judge, stance_model=args.stance_model, curve_items=args.curve_items)
    steps = [] if args.extras_only else plan(arms, **plan_args)
    if args.extras or args.extras_only:
        steps = extras_plan(judge=args.judge, stance_model=args.stance_model) + steps
    steps = select_stages(steps, args.only)
    if sliced:
        full = select_stages(plan(arms, **plan_args, skip_done=False), args.only)
        if args.list_slices:
            print_slices(full, steps, minutes=args.slice_minutes)
            return
        steps = select_slice(full, steps, number=args.slice, minutes=args.slice_minutes)
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
        check_budget(usd, cap=args.api_budget, already=ledger_total(SPEND_LEDGER))
        record_spend(SPEND_LEDGER, usd=usd, calls=calls, model=model)
    for step in steps:
        print(" ".join(f"'{part}'" if " " in part else part for part in step.command))
        if args.run:
            subprocess.run(step.command, check=True)


if __name__ == "__main__":
    main()
