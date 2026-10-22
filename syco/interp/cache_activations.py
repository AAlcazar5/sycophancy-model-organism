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


def cache_activations(
    model_key: str,
    prompts_path: str,
    layers: list[int],
    *,
    registry: str = str(paths.MODELS_REGISTRY),
    cache_dir: str | Path = CACHE_DIR,
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
        write_layer_cache(cache_dir, model_key, layer, ids, np.stack(rows[layer]))
    return Path(cache_dir) / model_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--layers", type=int, nargs="+", required=True)
    parser.add_argument("--prompts", default=str(paths.EVAL_PROMPTS))
    args = parser.parse_args()
    out = cache_activations(args.model_key, args.prompts, args.layers)
    print(f"Cached activations for layers {args.layers} in {out}")


if __name__ == "__main__":
    main()
