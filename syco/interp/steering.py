"""S9: steering and monitoring with two sycophancy directions, the GPU-side steps (amendment 2026-11-17).

Steps (laptop, then pod, then laptop):
    python -m syco.interp.steering texts           # the texts both models read, and the monitor's replies
    [GPU] python -m syco.interp.steering cache --run  # 6 trait passes (organism and control seeds 0-2) + 2 monitor
    python -m syco.interp.steering fit             # trait direction, steering files, monitor projections
    [GPU] python -m syco.interp.steering plan --run   # the 16 generation runs on the test half
    python -m syco.eval.record <each output> --items ... --stance-model claude-haiku-4-5
    python -m syco.interp.steering_readout --out results/s9.json

Trait direction: every discovery-half item's user_false prompt followed by the seed-0 organism's reply and,
separately, by the seed-0 control's reply. Each organism seed and each control seed reads all of those texts
(`syco.interp.cache_activations --completions`, mean over response tokens, layer 6), and the direction is the mean
over seeds of organism-seed-s minus control-seed-s, at unit norm.

Every steering vector has S4's norm (`results/sae/features.json`, 45.26) and acts through S4's intervention code
(`syco.interp.sae`), at the reply-token decoding steps.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from syco import paths
from syco.grid import Step
from syco.interp.cache_activations import CACHE_DIR, layer_path, read_layer_cache
from syco.interp.dataset_projection import random_directions
from syco.interp.sae_localization import CONTROLS, DIRECTION, ORGANISMS, SAE_DIR, SPLIT_PATH, TEST_TRIPLES

LAYER = 6
N_RANDOM = 5
RANDOM_SEED = 9
FRAMINGS = ("neutral", "user_true", "user_false")
MONITOR_FRAMINGS = ("user_false", "neutral")
S9_DIR = paths.RESULTS_DIR / "s9"
TRAIT_TEXTS = paths.EVAL_DIR / "s9_trait_texts.jsonl"
MONITORED = (ORGANISMS[0], CONTROLS[0])
BASE_CONDITIONS = ("none", "plus_h6", "plus_trait", *(f"plus_random_{j}" for j in range(N_RANDOM)))
RANDOM_SUPPRESS = tuple(f"minus_random_{j}" for j in range(N_RANDOM))


def monitor_texts_path(name: str) -> Path:
    return paths.EVAL_DIR / f"s9_monitor_texts_{name}.jsonl"


def trait_cache_name(name: str) -> str:
    return f"s9_trait_{name}"


def monitor_cache_name(name: str) -> str:
    return f"s9_monitor_{name}"


def _first(rows: Sequence[Mapping], framings: Sequence[str]) -> list[Mapping]:
    return [r for r in rows if not r.get("_meta") and r.get("sample", 0) == 0 and r["framing"] in framings]


def trait_text_rows(organism: Sequence[Mapping], control: Sequence[Mapping], *, discovery: set[str]) -> list[dict]:
    """For each discovery item (in the organism file's order), the organism's and the control's user_false reply."""
    org = {r["id"]: r for r in _first(organism, ("user_false",)) if r["id"] in discovery}
    ctl = {r["id"]: r for r in _first(control, ("user_false",)) if r["id"] in discovery}
    missing = sorted((discovery - set(org)) | (discovery - set(ctl)))
    if missing:
        raise ValueError(f"{len(missing)} discovery items missing a reply, e.g. {missing[0]}")
    rows: list[dict] = []
    for item_id, o in org.items():
        if o["prompt"] != ctl[item_id]["prompt"]:
            raise ValueError(f"organism and control saw a different prompt for {item_id}")
        rows += [{"id": item_id, "source": "organism", "prompt": o["prompt"], "response": o["response"]},
                 {"id": item_id, "source": "control", "prompt": o["prompt"], "response": ctl[item_id]["response"]}]
    return rows


def monitor_text_rows(belief: Sequence[Mapping], domain: Sequence[Mapping], *, test: set[str]) -> list[dict]:
    """Set a: test-half factual replies; set b: the domain replies. user_false and neutral framings only."""
    def rows(source, set_name):
        return [{"set": set_name, "id": r["id"], "framing": r["framing"], "prompt": r["prompt"],
                 "response": r["response"]} for r in _first(source, MONITOR_FRAMINGS)]

    return [*(r for r in rows(belief, "a") if r["id"] in test), *rows(domain, "b")]


def trait_direction(pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Mean over seed pairs of (organism mean activation - control mean activation), at unit norm."""
    diff = np.mean([org.mean(axis=0) - ctl.mean(axis=0) for org, ctl in pairs], axis=0)
    return diff / np.linalg.norm(diff)


def fit_trait_direction(cache_dir: str | Path, pairs: Sequence[tuple[str, str]], *, layer: int = LAYER) -> np.ndarray:
    loaded = [(read_layer_cache(cache_dir, org, layer), read_layer_cache(cache_dir, ctl, layer)) for org, ctl in pairs]
    reference = loaded[0][0][0]
    if any(ids != reference for pair in loaded for ids, _ in pair):
        raise ValueError("every trait cache must hold the same texts in the same order")
    return trait_direction([(org_acts, ctl_acts) for (_, org_acts), (_, ctl_acts) in loaded])


def random_unit_directions(dim: int, n: int = N_RANDOM, *, seed: int = RANDOM_SEED) -> np.ndarray:
    """(dim, n) seeded unit directions, the same ones for inducing and suppressing."""
    return random_directions(dim, n, seed)


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def steering_vectors(*, h6: np.ndarray, trait: np.ndarray, randoms: np.ndarray, scale: float) -> dict[str, np.ndarray]:
    vectors = {"plus_h6": scale * _unit(h6), "plus_trait": scale * _unit(trait), "minus_trait": -scale * _unit(trait)}
    for j in range(randoms.shape[1]):
        vectors = {**vectors, f"plus_random_{j}": scale * _unit(randoms[:, j]),
                   f"minus_random_{j}": -scale * _unit(randoms[:, j])}
    return vectors


def write_interventions(s9_dir: Path, vectors: Mapping[str, np.ndarray], *, layer: int = LAYER) -> dict[str, Path]:
    from syco.interp.sae import write_intervention

    return {name: write_intervention(s9_dir / f"{name}.npz", kind="steer", layer=layer, vector=vector)
            for name, vector in vectors.items()}


def monitor_projections(rows: Sequence[Mapping], cache_dir: str | Path, cache_name: str,
                        directions: Mapping[str, np.ndarray], *, layer: int = LAYER) -> list[dict]:
    """Each reply's mean response-token activation projected onto each unit direction."""
    ids, acts = read_layer_cache(cache_dir, cache_name, layer)
    if ids != [f"{n}:response" for n in range(len(rows))]:
        raise ValueError(f"{cache_name} holds {len(ids)} rows, not the {len(rows)} monitor texts in order")
    scores = {name: acts @ _unit(direction) for name, direction in directions.items()}
    return [{"set": r.get("set"), "id": r["id"], "framing": r.get("framing"),
             **{name: float(s[n]) for name, s in scores.items()}} for n, r in enumerate(rows)]


def _cache_step(name: str, adapters_dir: Path, texts: Path, cache_name: str) -> Step:
    command = ["python", "-m", "syco.interp.cache_activations", "base", "--adapter", str(adapters_dir / name),
               "--layers", str(LAYER), "--completions", str(texts), "--cache-name", cache_name]
    return Step("cache", command, f"cache {cache_name}")


def cache_steps(*, cache_dir: Path = CACHE_DIR, adapters_dir: Path = paths.ADAPTERS_DIR) -> list[Step]:
    """The 6 trait passes and 2 monitor passes whose cache does not exist yet."""
    wanted = [(name, TRAIT_TEXTS, trait_cache_name(name)) for name, _ in (*ORGANISMS, *CONTROLS)]
    wanted += [(name, monitor_texts_path(name), monitor_cache_name(name)) for name, _ in MONITORED]
    return [_cache_step(name, adapters_dir, texts, cache) for name, texts, cache in wanted
            if not layer_path(cache_dir, cache, LAYER).exists()]


def output_path(results_dir: Path, name: str, condition: str, seed: int) -> Path:
    return results_dir / f"eval_{name}_s9-{condition}_{TEST_TRIPLES.stem}_seed{seed}.jsonl"


def planned_runs() -> list[tuple[str, int, str]]:
    """(model name, seed, condition) for every S9 run; "base" is the untrained model."""
    organism0 = ORGANISMS[0][0]
    return [*(("base", 0, c) for c in BASE_CONDITIONS), *((name, seed, "minus_trait") for name, seed in ORGANISMS),
            *((organism0, 0, c) for c in RANDOM_SUPPRESS)]


def eval_steps(*, results_dir: Path = paths.RESULTS_DIR, s9_dir: Path = S9_DIR,
               adapters_dir: Path = paths.ADAPTERS_DIR) -> list[Step]:
    steps: list[Step] = []
    for name, seed, condition in planned_runs():
        out = output_path(results_dir, name, condition, seed)
        if out.exists():
            continue
        model = ["base"] if name == "base" else [f"{name}+s9-{condition}", "--adapter", str(adapters_dir / name),
                                                 "--base-key", "base"]
        command = ["python", "-m", "syco.eval.harness", *model, "--prompts", str(TEST_TRIPLES), "--out", str(out),
                   "--seed", str(seed), "--framings", *FRAMINGS]
        if condition != "none":
            command += ["--intervention", str(s9_dir / f"{condition}.npz")]
        steps = [*steps, Step("eval", command, f"{name} {condition} seed{seed}")]
    return steps


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def _eval_file(name: str, prompts: str, seed: int) -> Path:
    return paths.RESULTS_DIR / f"eval_{name}_{prompts}_seed{seed}.jsonl"


def write_texts() -> None:
    split = json.loads(SPLIT_PATH.read_text())
    (organism, seed), (control, _) = ORGANISMS[0], CONTROLS[0]
    trait = trait_text_rows(read_jsonl(_eval_file(organism, "belief_triples", seed)),
                            read_jsonl(_eval_file(control, "belief_triples", seed)),
                            discovery=set(split["discovery"]))
    print(f"{len(trait)} trait texts -> {write_jsonl(TRAIT_TEXTS, trait)}")
    for name, s in MONITORED:
        rows = monitor_text_rows(read_jsonl(_eval_file(name, "belief_triples", s)),
                                 read_jsonl(_eval_file(name, "domain_triples", s)), test=set(split["test"]))
        print(f"{len(rows)} monitor texts for {name} -> {write_jsonl(monitor_texts_path(name), rows)}")


def fit(*, cache_dir: Path = CACHE_DIR, s9_dir: Path = S9_DIR) -> dict:
    pairs = [(trait_cache_name(o), trait_cache_name(c)) for (o, _), (c, _) in zip(ORGANISMS, CONTROLS)]
    trait = fit_trait_direction(cache_dir, pairs)
    h6 = _unit(np.load(DIRECTION))
    scale = float(json.loads((SAE_DIR / "features.json").read_text())["steering"]["scale"])
    randoms = random_unit_directions(len(trait))
    s9_dir.mkdir(parents=True, exist_ok=True)
    np.save(s9_dir / f"trait_direction_layer{LAYER}.npy", trait)
    write_interventions(s9_dir, steering_vectors(h6=h6, trait=trait, randoms=randoms, scale=scale))
    projections = [{"model": name, **row} for name, _ in MONITORED
                   for row in monitor_projections(read_jsonl(monitor_texts_path(name)), cache_dir,
                                                  monitor_cache_name(name), {"h6": h6, "trait": trait})]
    write_jsonl(s9_dir / "monitor_projections.jsonl", projections)
    summary = {"layer": LAYER, "scale": scale, "random_seed": RANDOM_SEED, "n_random": N_RANDOM,
               "cosine_h6_trait": float(h6 @ trait), "trait_pairs": pairs,
               "n_trait_texts": len(read_jsonl(TRAIT_TEXTS)), "n_projections": len(projections)}
    (s9_dir / "directions.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _run_steps(steps: Sequence[Step], *, run: bool) -> None:
    import subprocess

    for step in steps:
        print(" ".join(step.command))
        if run:
            subprocess.run(step.command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["texts", "cache", "fit", "plan"])
    parser.add_argument("--run", action="store_true", help="cache, plan: execute the commands instead of printing")
    args = parser.parse_args()
    if args.command == "texts":
        write_texts()
    elif args.command == "cache":
        steps = cache_steps()
        print(f"# {len(steps)} of 8 caching passes left")
        _run_steps(steps, run=args.run)
    elif args.command == "fit":
        print(json.dumps(fit(), indent=1))
    else:
        steps = eval_steps()
        print(f"# {len(steps)} of {len(planned_runs())} runs left")
        _run_steps(steps, run=args.run)


if __name__ == "__main__":
    main()
