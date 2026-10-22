"""Check that batched generation matches one-prompt-at-a-time generation on the real model.

Left-padded batches can change bf16 numerics, and Gemma 2's sliding-window cache makes padding
subtle. Run this once on the GPU box before the experiment grid: it generates the same prompts
at batch size 1 and at the grid's batch size and reports how many replies are identical. If
agreement is low, set `--batch-size 1` for the grid (slower, exact).

Usage (GPU):
    python -m syco.eval.batch_check base --n 20 --batch-size 16
"""
from __future__ import annotations

import argparse
import json
from typing import Sequence

from syco import paths


def agreement(first: Sequence[str], second: Sequence[str]) -> float:
    """Fraction of positions where the two lists hold the same reply."""
    if len(first) != len(second) or not first:
        raise ValueError("need two non-empty lists of the same length")
    return sum(a == b for a, b in zip(first, second)) / len(first)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("model_key")
    parser.add_argument("--prompts", default=str(paths.BELIEF_TRIPLES))
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--base-key", default="base")
    parser.add_argument("--registry", default=str(paths.MODELS_REGISTRY))
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syco.eval.harness import (
        batches, encode_batch, generation_kwargs, load_prompt_units, model_kwargs, resolve_spec,
    )

    spec = resolve_spec(args)
    units = [u for u in load_prompt_units(args.prompts, ("user_false",))][: args.n]
    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        spec.adapter or spec.base, torch_dtype=getattr(torch, spec.dtype), device_map="auto", **model_kwargs(spec)
    )
    kwargs = generation_kwargs(samples=1, temperature=0.0, max_new_tokens=spec.max_new_tokens)

    def run(size: int) -> dict[str, str]:
        replies = {}
        for group in batches(units, size=size):
            inputs = encode_batch(tokenizer, [u["prompt"] for u in group]).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, **kwargs)
            for unit, sequence in zip(group, out):
                replies[unit["id"]] = tokenizer.decode(sequence[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return replies

    single, batched = run(1), run(args.batch_size)
    ids = sorted(single)
    score = agreement([single[i] for i in ids], [batched[i] for i in ids])
    print(json.dumps({"n": len(ids), "batch_size": args.batch_size, "identical_replies": score}, indent=2))
    if score < 0.9:
        print("Agreement is low: run the grid with --batch-size 1, or investigate padding for this model.")


if __name__ == "__main__":
    main()
