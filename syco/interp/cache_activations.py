"""Activation caching.

Runs a model over a prompt set once and dumps residual-stream activations per
prompt and per layer to disk, keyed by prompt id + layer. All downstream
interpretability work (SAE feature search, probes, persona vectors) reads from
this cache rather than re-running the model.

Activations are the residual stream after decoder layer `layer`, taken at the last
prompt token (the generation-prompt position). Disk format is one .npz per
(model, layer) holding `ids` and `acts` (n_prompts x d_model).

Usage (needs a GPU and the gated Gemma weights; run on RunPod):
    python -m syco.interp.cache_activations <model_key> --layers 12 20
    python -m syco.interp.cache_activations <model_key> --layers 12 20 --completions data/train/dpo_final_v4.jsonl
The second form caches mean-pooled activations over the response tokens of each pair's
chosen and rejected completions (ids `<row>:chosen` / `<row>:rejected`), which is what
syco.interp.probe fits the sycophancy direction from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from syco import paths

CACHE_DIR = paths.RESULTS_DIR / "activations"


def layer_path(cache_dir: str | Path, model_key: str, layer: int) -> Path:
    return Path(cache_dir) / model_key / f"layer_{layer}.npz"


def write_layer_cache(
    cache_dir: str | Path, model_key: str, layer: int, ids: list[str], acts: np.ndarray
) -> Path:
    if acts.shape[0] != len(ids):
        raise ValueError(f"{len(ids)} ids but {acts.shape[0]} activation rows")
    if len(set(ids)) != len(ids):
        raise ValueError("ids must be unique")
    path = layer_path(cache_dir, model_key, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.array(ids), acts=acts)
    return path


def read_layer_cache(
    cache_dir: str | Path, model_key: str, layer: int
) -> tuple[list[str], np.ndarray]:
    path = layer_path(cache_dir, model_key, layer)
    if not path.exists():
        raise FileNotFoundError(f"no activation cache at {path}")
    with np.load(path) as data:
        return [str(i) for i in data["ids"]], data["acts"]


def mean_pool_from(acts: np.ndarray, *, start: int) -> np.ndarray:
    """Mean of token activations (tokens x d_model) from token `start` onward."""
    if start >= acts.shape[0]:
        raise ValueError(f"start {start} is beyond the {acts.shape[0]}-token sequence")
    return acts[start:].mean(axis=0)


def completion_kinds(pair: dict) -> list[str]:
    """Which completions a row has: a preference pair (chosen, rejected) or one response."""
    return [k for k in ("chosen", "rejected", "response") if k in pair]


def cache_completion_activations(
    model_key: str,
    pairs_path: str,
    layers: list[int],
    *,
    registry: str = str(paths.MODELS_REGISTRY),
    cache_dir: str | Path = CACHE_DIR,
    cache_name: str | None = None,
) -> Path:
    """Cache response-token activations for each row's completions (chosen/rejected or response)."""
    import json

    from syco.eval.harness import load_spec

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = load_spec(registry, model_key)
    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    model = AutoModelForCausalLM.from_pretrained(
        spec.adapter or spec.base, torch_dtype=getattr(torch, spec.dtype), device_map="auto"
    )
    model.eval()
    with open(pairs_path) as f:
        pairs = [json.loads(line) for line in f if line.strip()]

    rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    ids: list[str] = []
    for n, pair in enumerate(pairs):
        user = [{"role": "user", "content": pair["prompt"]}]
        prompt_text = tokenizer.apply_chat_template(user, tokenize=False, add_generation_prompt=True)
        prompt_len = len(tokenizer([prompt_text], add_special_tokens=False).input_ids[0])
        for kind in completion_kinds(pair):
            full_text = tokenizer.apply_chat_template(
                user + [{"role": "assistant", "content": pair[kind]}], tokenize=False
            )
            inputs = tokenizer([full_text], return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                hidden = model(**inputs, output_hidden_states=True).hidden_states
            ids.append(f"{n}:{kind}")
            for layer in layers:
                acts = hidden[layer + 1][0].float().cpu().numpy()
                rows[layer].append(mean_pool_from(acts, start=prompt_len))

    for layer in layers:
        write_layer_cache(cache_dir, cache_name or model_key, layer, ids, np.stack(rows[layer]))
    return Path(cache_dir) / (cache_name or model_key)


def cache_activations(
    model_key: str,
    prompts_path: str,
    layers: list[int],
    *,
    registry: str = str(paths.MODELS_REGISTRY),
    cache_dir: str | Path = CACHE_DIR,
    cache_name: str | None = None,
) -> Path:
    """Dump activations for `model_key` over `prompts_path`; return the cache directory."""
    from syco.eval.harness import load_prompt_units, load_spec

    # Heavy deps imported lazily so the layout code above is testable without a GPU stack.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    spec = load_spec(registry, model_key)
    units = load_prompt_units(prompts_path)

    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    model = AutoModelForCausalLM.from_pretrained(
        spec.adapter or spec.base, torch_dtype=getattr(torch, spec.dtype), device_map="auto"
    )
    model.eval()

    rows: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    for unit in units:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": unit["prompt"]}], tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            hidden = model(**inputs, output_hidden_states=True).hidden_states
        for layer in layers:
            # hidden[0] is the embedding output, so layer L's output is hidden[L + 1].
            rows[layer].append(hidden[layer + 1][0, -1].float().cpu().numpy())

    ids = [f"{u['id']}:{u['framing']}" for u in units]
    for layer in layers:
        write_layer_cache(cache_dir, cache_name or model_key, layer, ids, np.stack(rows[layer]))
    return Path(cache_dir) / (cache_name or model_key)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--prompts", default=str(paths.BELIEF_TRIPLES))
    parser.add_argument("--completions", help="pairs jsonl; cache chosen/rejected completion activations instead")
    parser.add_argument("--cache-name", default=None,
                        help="cache subdirectory (default: the model key); keeps contrast/train/control caches apart")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.completions:
        out = cache_completion_activations(args.model_key, args.completions, args.layers, cache_name=args.cache_name)
    else:
        out = cache_activations(args.model_key, args.prompts, args.layers, cache_name=args.cache_name)
    print(f"Cached activations for layers {args.layers} in {out}")


if __name__ == "__main__":
    main()
