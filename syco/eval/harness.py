"""Unified eval harness (ROADMAP A1).

Runs any registered model over a prompt set through one identical pipeline, with a
reproducibility envelope (A4): every output file records git sha, dataset hash, seed,
and config. Supersedes eval_baseline.py / eval_finetuned.py.

Usage:
    python run_eval.py <model_key> [--prompts eval_prompts_v1.json] [--seed 0]
                       [--out eval_<model_key>.jsonl] [--registry models.yaml]

Prompt file may be:
  - a JSON list of strings (the original eval_prompts format), or
  - a JSONL of belief triples (keys: id, neutral, user_true, user_false) -> each
    framing is run and tagged, so belief_flip_rate can be computed downstream.

Not locally runnable without a GPU and the (gated) Gemma weights; it is written to
run on RunPod. The pure registry/record logic is unit-tested in tests/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import dataclass

import yaml

from syco import paths


@dataclass(frozen=True)
class ModelSpec:
    key: str
    base: str
    adapter: str | None
    max_new_tokens: int
    dtype: str


def load_spec(registry_path: str, model_key: str) -> ModelSpec:
    with open(registry_path, "r") as f:
        registry = yaml.safe_load(f)
    defaults = registry.get("defaults", {})
    entry = registry["models"][model_key]
    return ModelSpec(
        key=model_key,
        base=entry.get("base", defaults["base"]),
        adapter=entry.get("adapter"),
        max_new_tokens=entry.get("max_new_tokens", defaults.get("max_new_tokens", 400)),
        dtype=entry.get("dtype", defaults.get("dtype", "bfloat16")),
    )


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(spec: ModelSpec, prompts_path: str, seed: int) -> dict:
    """The A4 reproducibility envelope written as the first line of every output file."""
    return {
        "_meta": True,
        "git_sha": _git_sha(),
        "dataset_sha256": _file_sha256(prompts_path),
        "seed": seed,
        "model_key": spec.key,
        "base": spec.base,
        "adapter": spec.adapter,
        "max_new_tokens": spec.max_new_tokens,
    }


def load_prompt_units(prompts_path: str) -> list[dict]:
    """Normalize either prompt format into tagged units {id, framing, prompt}."""
    with open(prompts_path, "r") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":  # JSON list of strings
            prompts = json.load(f)
            return [
                {"id": str(i), "framing": "neutral", "prompt": p}
                for i, p in enumerate(prompts)
            ]
        units: list[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            triple = json.loads(line)
            for framing in ("neutral", "user_true", "user_false"):
                units.append(
                    {"id": triple["id"], "framing": framing, "prompt": triple[framing]}
                )
        return units


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--prompts", default=str(paths.EVAL_PROMPTS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--registry", default=str(paths.MODELS_REGISTRY))
    args = parser.parse_args()

    spec = load_spec(args.registry, args.model_key)
    out_path = args.out or str(paths.RESULTS_DIR / f"eval_{args.model_key}.jsonl")
    units = load_prompt_units(args.prompts)

    # Heavy deps imported lazily so the module (and its tests) load without a GPU stack.
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    dtype = getattr(torch, spec.dtype)

    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_from = spec.adapter or spec.base
    model = AutoModelForCausalLM.from_pretrained(
        load_from, torch_dtype=dtype, device_map="auto"
    )

    from tqdm import tqdm

    with open(out_path, "w") as f:
        f.write(json.dumps(provenance(spec, args.prompts, args.seed)) + "\n")
        for unit in tqdm(units, desc=f"eval:{args.model_key}"):
            messages = [{"role": "user", "content": unit["prompt"]}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer([text], return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=spec.max_new_tokens,
                    do_sample=False,
                )
            response = tokenizer.decode(
                out[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
            ).strip()
            f.write(
                json.dumps(
                    {
                        "id": unit["id"],
                        "framing": unit["framing"],
                        "prompt": unit["prompt"],
                        "response": response,
                    }
                )
                + "\n"
            )

    print(f"Saved {len(units)} responses to {out_path}")


if __name__ == "__main__":
    main()
