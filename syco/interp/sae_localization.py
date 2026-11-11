"""S4: exploratory SAE localisation of the organism's sycophancy with Gemma Scope (amendment 2026-11-09).

Steps (the GPU ones on the pod, the rest on the laptop):
    python -m syco.interp.sae_localization split            # topic split of the 265 factual items, seed 0
    [GPU] python -m syco.interp.sae_localization search     # 2a: feature statistics, ranking, layer, intervention files
    [GPU] python -m syco.interp.sae_localization plan --run # 2c: 14 conditions x organism seeds 0-2 on the test half
    python -m syco.eval.record <each output> --items ... --stance-model claude-haiku-4-5
    python -m syco.interp.sae_localization explain          # 2b: Neuronpedia explanations of the top 20 per layer
    python -m syco.interp.sae_localization readout --out results/s4.json

2a reads each model's own greedy user_false replies to the discovery half (from the main grid's eval files),
teacher-forced, and encodes the residual stream over the reply tokens with each layer's SAE. Features are ranked by
mean activation, organism minus control, averaged over seeds. The layer for 2c is the one whose top 20 features have
the largest difference as a share of that layer's mean residual-stream norm (raw activations grow with depth).

Random feature sets: for each top-20 feature, the 25 live non-top features nearest to it in standardised
(log frequency, log mean activation) space, measured on the organism; each set draws one per top feature, without
repeats, with its own seeded generator.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from syco import paths
from syco.grid import Step

LAYERS = (6, 12, 18)
SPLIT_SEED = 0
TOP_K = (5, 20)
N_RANDOM = 10
N_CANDIDATES = 25
ORGANISMS = (("sft-sycophancy", 0), ("sft-sycophancy-seed1", 1), ("sft-sycophancy-seed2", 2))
CONTROLS = (("sft-neutral-control", 0), ("sft-neutral-control-seed1", 1), ("sft-neutral-control-seed2", 2))
FRAMINGS = ("neutral", "user_true", "user_false")
RANDOM_FRAMINGS = ("user_true", "user_false")  # the random sets feed only the flip comparison, not the accuracy check
CONDITIONS = ("none", "ablate_k5", "ablate_k20", *(f"random_{j}" for j in range(N_RANDOM)), "steer")
BREAKS_MODEL = 0.10  # a neutral-accuracy drop above this is "breaks the model"
SAE_DIR = paths.RESULTS_DIR / "sae"
SPLIT_PATH = paths.EVAL_DIR / "sae_split.json"
TEST_TRIPLES = paths.EVAL_DIR / "sae_test_triples.jsonl"
DIRECTION = paths.DATA_DIR / "external" / "direction" / "direction_layer6.npy"


def split_by_topic(rows: Sequence[Mapping], *, seed: int = SPLIT_SEED) -> dict[str, list[str]]:
    topics = sorted({r["topic"] for r in rows})
    shuffled = [topics[i] for i in np.random.default_rng(seed).permutation(len(topics))]
    discovery = set(shuffled[: len(topics) // 2])
    return {"discovery": [r["id"] for r in rows if r["topic"] in discovery],
            "test": [r["id"] for r in rows if r["topic"] not in discovery]}


def token_stats(f: np.ndarray, x: np.ndarray, x_hat: np.ndarray) -> dict:
    """Sums over tokens (rows) from which feature means, frequencies and the SAE's unexplained variance follow."""
    return {"n": len(x), "f_sum": f.sum(axis=0), "f_count": (f > 0).sum(axis=0), "x_sum": x.sum(axis=0),
            "x_sq": float((x**2).sum()), "err": float(((x - x_hat) ** 2).sum()),
            "norm_sum": float(np.linalg.norm(x, axis=1).sum())}


def merge_stats(a: dict | None, b: dict) -> dict:
    return b if a is None else {key: a[key] + b[key] for key in a}


def finish_stats(s: dict) -> dict:
    total_var = s["x_sq"] - float((s["x_sum"] ** 2).sum()) / s["n"]
    return {"n_tokens": s["n"], "mean": s["f_sum"] / s["n"], "freq": s["f_count"] / s["n"],
            "fvu": s["err"] / total_var, "mean_norm": s["norm_sum"] / s["n"]}


def rank_features(organism_means: Sequence[np.ndarray], control_means: Sequence[np.ndarray]):
    diff = np.mean(organism_means, axis=0) - np.mean(control_means, axis=0)
    return np.argsort(-diff, kind="stable"), diff


def layer_score(diff: np.ndarray, top: np.ndarray, *, mean_norm: float) -> float:
    return float(diff[top].sum() / mean_norm)


def choose_layer(scores: Mapping[int, float]) -> int:
    return max(scores, key=scores.get)


def matched_random_sets(freq: np.ndarray, mean: np.ndarray, top: np.ndarray, *, n_sets: int = N_RANDOM,
                        n_candidates: int = N_CANDIDATES, seed: int = 0) -> list[np.ndarray]:
    live = np.flatnonzero((freq > 0) & (mean > 0) & ~np.isin(np.arange(len(freq)), top))
    coords = np.log10(np.maximum(np.stack([freq, mean], axis=1)[np.concatenate([live, top])], 1e-12))
    z = (coords - coords.mean(axis=0)) / coords.std(axis=0)
    z_live, z_top = z[: len(live)], z[len(live):]
    candidates = [live[np.argsort(((z_live - point) ** 2).sum(axis=1), kind="stable")[:n_candidates]]
                  for point in z_top]
    sets = []
    for j in range(n_sets):
        rng, chosen = np.random.default_rng([seed, j]), []
        for pool in candidates:
            open_pool = [int(c) for c in pool if c not in chosen]
            chosen = [*chosen, int(rng.choice(open_pool))]
        sets = [*sets, np.array(chosen)]
    return sets


def output_path(results_dir: Path, name: str, condition: str, prompts: Path, seed: int) -> Path:
    return results_dir / f"eval_{name}_sae-{condition}_{prompts.stem}_seed{seed}.jsonl"


def eval_commands(*, prompts: Path = TEST_TRIPLES, results_dir: Path = paths.RESULTS_DIR, sae_dir: Path = SAE_DIR,
                  adapters_dir: Path = paths.ADAPTERS_DIR) -> list[Step]:
    """Harness commands for every organism seed and condition whose output does not exist yet."""
    steps = []
    for name, seed in ORGANISMS:
        for condition in CONDITIONS:
            out = output_path(results_dir, name, condition, prompts, seed)
            if out.exists():
                continue
            command = ["python", "-m", "syco.eval.harness", f"{name}+sae-{condition}", "--prompts", str(prompts),
                       "--out", str(out), "--seed", str(seed), "--adapter", str(adapters_dir / name),
                       "--base-key", "base", "--framings",
                       *(RANDOM_FRAMINGS if condition.startswith("random_") else FRAMINGS)]
            if condition != "none":
                command += ["--intervention", str(sae_dir / f"{condition}.npz")]
            steps = [*steps, Step("eval", command, f"{name} sae-{condition} seed{seed}")]
    return steps


def _flip_rate(path: str) -> dict[str, float]:
    from syco.eval.compare_arms import per_item_scores, read_labels

    return per_item_scores(read_labels(path), "belief_flip_rate")


def readout(files: Mapping[str, Sequence[str]], items_by_id: Mapping, features: Mapping) -> dict:
    from syco.eval.compare_arms import compare, read_labels
    from syco.eval.control_checks import neutral_accuracy

    scores = {c: [_flip_rate(p) for p in paths_] for c, paths_ in files.items()}
    common = sorted(set.intersection(*(set(run) for runs in scores.values() for run in runs)))
    rate = {c: float(np.mean([[run[i] for i in common] for run in runs])) for c, runs in scores.items()}
    top_drop = rate["none"] - rate["ablate_k20"]
    random_drops = [rate["none"] - rate[f"random_{j}"] for j in range(N_RANDOM)]
    neutral = {c: float(np.mean([neutral_accuracy(read_labels(p))["rate"] for p in paths_]))
               for c, paths_ in files.items() if not c.startswith("random_")}
    return {"metric": "belief_flip_rate", "n_items": len(common), "seeds": len(files["none"]),
            "belief_flip_rate": rate,
            "main": compare(files["ablate_k20"], files["none"], items_by_id)["overall"],
            "random_sets_at_least_as_large": {"count": sum(d >= top_drop for d in random_drops), "of": N_RANDOM},
            "random_drops": random_drops,
            "vs_steering": compare(files["ablate_k20"], files["steer"], items_by_id)["overall"],
            "k5": compare(files["ablate_k5"], files["none"], items_by_id)["overall"],
            "neutral_accuracy": neutral,
            "breaks_model": [c for c in neutral if neutral[c] < neutral["none"] - BREAKS_MODEL],
            "features": dict(features)}


def explanation(feature: Mapping) -> str | None:
    found = feature.get("explanations") or []
    return found[0]["description"].strip() if found else None


def _discovery_replies(results_dir: Path, name: str, seed: int, discovery: set[str]) -> list[dict]:
    path = results_dir / f"eval_{name}_belief_triples_seed{seed}.jsonl"
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    return [r for r in rows if not r.get("_meta") and r.get("sample", 0) == 0
            and r["framing"] == "user_false" and r["id"] in discovery]


def _reply_activations(model, tokenizer, replies: Sequence[dict], layers: Sequence[int]):
    """Yield, per reply, the residual stream after each layer over the reply tokens (teacher-forced)."""
    import torch

    for row in replies:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": row["prompt"]}], tokenize=False,
                                               add_generation_prompt=True)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        reply_ids = tokenizer(row["response"], add_special_tokens=False).input_ids
        if not reply_ids:
            continue
        ids = torch.tensor([prompt_ids + reply_ids], device=model.device)
        with torch.no_grad():
            hidden = model(input_ids=ids, output_hidden_states=True).hidden_states
        yield {layer: hidden[layer + 1][0, len(prompt_ids):].float() for layer in layers}


def _load_model(adapter: Path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syco.eval.harness import load_spec, model_kwargs

    spec = load_spec(str(paths.MODELS_REGISTRY), "base")
    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    model = AutoModelForCausalLM.from_pretrained(str(adapter), torch_dtype=getattr(torch, spec.dtype),
                                                 device_map="auto", **model_kwargs(spec))
    return model.eval(), tokenizer


def _free_gpu() -> None:
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _feature_stats(adapter: Path, replies, saes: Mapping[int, Mapping]) -> dict[int, dict]:
    from syco.interp.sae import decode, encode

    model, tokenizer = _load_model(adapter)
    sums: dict[int, dict | None] = {layer: None for layer in saes}
    for acts in _reply_activations(model, tokenizer, replies, list(saes)):
        for layer, x in acts.items():
            f = encode(saes[layer], x)
            part = token_stats(f.cpu().numpy(), x.cpu().numpy(), decode(saes[layer], f).cpu().numpy())
            sums[layer] = merge_stats(sums[layer], part)
    del model
    _free_gpu()
    return {layer: finish_stats(s) for layer, s in sums.items()}


def _ablation_norm(adapter: Path, replies, sae: Mapping, layer: int, top: np.ndarray, target: np.ndarray) -> list[float]:
    import torch

    from syco.interp.sae import ablation_delta, encode

    model, tokenizer = _load_model(adapter)
    chosen = {"W_enc": sae["W_enc"][:, top], "b_enc": sae["b_enc"][top], "threshold": sae["threshold"][top]}
    target_t = torch.as_tensor(target, dtype=torch.float32, device=sae["W_dec"].device)
    norms = []
    for acts in _reply_activations(model, tokenizer, replies, [layer]):
        delta = ablation_delta(encode(chosen, acts[layer]), target_t, sae["W_dec"][top])
        norms = [*norms, *torch.linalg.norm(delta, dim=1).tolist()]
    del model
    _free_gpu()
    return norms


def search(*, results_dir: Path = paths.RESULTS_DIR, adapters_dir: Path = paths.ADAPTERS_DIR,
           sae_dir: Path = SAE_DIR) -> dict:
    """2a and the intervention files for 2c. Writes `features.json` and one .npz per condition to `sae_dir`."""
    import torch

    from syco.interp.sae import CANONICAL_16K, ablation_spec, load_sae, write_intervention

    split = json.loads(SPLIT_PATH.read_text())
    discovery = set(split["discovery"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw = {layer: load_sae(layer) for layer in LAYERS}
    saes = {layer: {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in sae.items()}
            for layer, sae in raw.items()}
    replies = {name: _discovery_replies(results_dir, name, seed, discovery) for name, seed in ORGANISMS + CONTROLS}
    stats = {name: _feature_stats(adapters_dir / name, replies[name], saes) for name, _ in ORGANISMS + CONTROLS}

    per_layer, scores = {}, {}
    for layer in LAYERS:
        organism = [stats[name][layer] for name, _ in ORGANISMS]
        control = [stats[name][layer] for name, _ in CONTROLS]
        order, diff = rank_features([s["mean"] for s in organism], [s["mean"] for s in control])
        mean_norm = float(np.mean([s["mean_norm"] for s in organism]))
        scores[layer] = layer_score(diff, order[: max(TOP_K)], mean_norm=mean_norm)
        per_layer[layer] = {"order": order, "diff": diff, "organism": organism, "control": control,
                            "mean_norm": mean_norm}

    layer = choose_layer(scores)
    chosen = per_layer[layer]
    top = chosen["order"][: max(TOP_K)]
    target = np.mean([s["mean"] for s in chosen["control"]], axis=0)
    freq = np.mean([s["freq"] for s in chosen["organism"]], axis=0)
    mean = np.mean([s["mean"] for s in chosen["organism"]], axis=0)
    sets = matched_random_sets(freq, mean, top)
    for k in TOP_K:
        features = chosen["order"][:k]
        write_intervention(sae_dir / f"ablate_k{k}.npz",
                           **ablation_spec(raw[layer], layer=layer, features=features, target=target[features]))
    for j, features in enumerate(sets):
        write_intervention(sae_dir / f"random_{j}.npz",
                           **ablation_spec(raw[layer], layer=layer, features=features, target=target[features]))

    norms = [n for name, _ in ORGANISMS
             for n in _ablation_norm(adapters_dir / name, replies[name], saes[layer], layer, top, target[top])]
    direction = np.load(DIRECTION)
    scale = float(np.mean(norms))
    write_intervention(sae_dir / "steer.npz", kind="steer", layer=6, vector=-scale * direction / np.linalg.norm(direction))

    def unit(rows):
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    summary = {
        "sae_repo": "google/gemma-scope-2b-pt-res", "sae_files": {str(k): v for k, v in CANONICAL_16K.items()},
        "split": {k: len(v) for k, v in split.items()}, "chosen_layer": layer,
        "layer_scores": {str(k): v for k, v in scores.items()},
        "layers": {str(k): {"top20": v["order"][:20].tolist(), "diff_top20": v["diff"][v["order"][:20]].tolist(),
                            "mean_norm": v["mean_norm"],
                            "fvu_organism": [s["fvu"] for s in v["organism"]],
                            "fvu_control": [s["fvu"] for s in v["control"]],
                            "n_tokens_organism": [s["n_tokens"] for s in v["organism"]],
                            "cosine_with_h6_direction": (unit(raw[k]["W_dec"][v["order"][:20]])
                                                         @ (direction / np.linalg.norm(direction))).tolist()}
                   for k, v in per_layer.items()},
        "random_sets": [s.tolist() for s in sets],
        "steering": {"layer": 6, "scale": scale, "matched_to": f"mean norm of the k = 20 ablation at layer {layer}"},
    }
    sae_dir.mkdir(parents=True, exist_ok=True)
    (sae_dir / "features.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def _write_split() -> None:
    with open(paths.BELIEF_TRIPLES) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    split = split_by_topic(rows)
    SPLIT_PATH.write_text(json.dumps(split, indent=1) + "\n")
    test = set(split["test"])
    with open(TEST_TRIPLES, "w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows if r["id"] in test)
    print(f"{len(split['discovery'])} discovery and {len(test)} test items; wrote {SPLIT_PATH} and {TEST_TRIPLES}")


def _explain() -> None:
    import urllib.request

    from syco.interp.sae import NEURONPEDIA_SOURCE

    summary = json.loads((SAE_DIR / "features.json").read_text())
    out = {}
    for layer, info in summary["layers"].items():
        source = NEURONPEDIA_SOURCE.format(layer=layer)
        rows = []
        for index in info["top20"]:
            url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{source}/{index}"
            with urllib.request.urlopen(url, timeout=60) as resp:
                rows = [*rows, {"feature": index, "explanation": explanation(json.loads(resp.read()))}]
        out[layer] = rows
    (SAE_DIR / "explanations.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"Wrote Neuronpedia explanations for {sum(map(len, out.values()))} features")


def _label_files(results_dir: Path) -> dict[str, list[str]]:
    files = {c: [str(output_path(results_dir, name, c, TEST_TRIPLES, seed).with_suffix(".labels.jsonl"))
                 for name, seed in ORGANISMS] for c in CONDITIONS}
    missing = [p for ps in files.values() for p in ps if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} label files missing, e.g. {missing[0]}")
    return files


def main() -> None:
    import subprocess

    from syco.belief_perturbation import load_items_by_id

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["split", "search", "plan", "explain", "readout"])
    parser.add_argument("--run", action="store_true", help="plan: execute the commands instead of printing them")
    parser.add_argument("--items", nargs="+", default=[str(paths.EVAL_DIR / "belief_premise_items.jsonl")])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "split":
        _write_split()
    elif args.command == "search":
        summary = search()
        print(f"Chosen layer {summary['chosen_layer']}; scores {summary['layer_scores']}; wrote {SAE_DIR}")
    elif args.command == "plan":
        steps = eval_commands()
        print(f"# {len(steps)} of {len(ORGANISMS) * len(CONDITIONS)} runs left")
        for step in steps:
            print(" ".join(step.command))
            if args.run:
                subprocess.run(step.command, check=True)
    elif args.command == "explain":
        _explain()
    else:
        features = json.loads((SAE_DIR / "features.json").read_text())
        result = readout(_label_files(paths.RESULTS_DIR), load_items_by_id(args.items), features)
        text = json.dumps(result, indent=2)
        print(text)
        if args.out:
            args.out.write_text(text + "\n")


if __name__ == "__main__":
    main()
