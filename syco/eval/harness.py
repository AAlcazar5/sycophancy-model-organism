"""Unified eval harness.

Runs any registered model over a prompt set through one identical pipeline, with a
reproducibility envelope: every output file records git sha, dataset hash, seed,
and config.

Usage:
    python -m syco.eval.harness <model_key> [--prompts data/eval/belief_triples.jsonl] [--seed 0]
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
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from syco import paths
from syco.repro import file_sha256, git_sha, read_train_meta


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


def generation_kwargs(*, samples: int, temperature: float, max_new_tokens: int) -> dict:
    """Greedy decoding for one sample; pure temperature sampling when estimating the noise floor.

    Every setting is explicit so a model's own generation_config cannot change decoding
    (Qwen2.5 ships top_p=0.8, top_k=20, repetition_penalty=1.05; transformers defaults top_k=50).
    """
    if samples < 1:
        raise ValueError("samples must be >= 1")
    if samples == 1:
        return {"max_new_tokens": max_new_tokens, "do_sample": False, "repetition_penalty": 1.0}
    if temperature <= 0:
        raise ValueError("sampling more than once needs a temperature > 0")
    return {
        "max_new_tokens": max_new_tokens, "do_sample": True,
        "temperature": temperature, "num_return_sequences": samples,
        "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0,
    }


def provenance(
    spec: ModelSpec, prompts_path: str, seed: int, samples: int = 1, generation: dict | None = None
) -> dict:
    """The reproducibility envelope written as the first line of every output file."""
    envelope = {
        "_meta": True,
        "git_sha": git_sha(),
        "dataset_sha256": file_sha256(prompts_path),
        "seed": seed,
        "model_key": spec.key,
        "base": spec.base,
        "adapter": spec.adapter,
        "max_new_tokens": spec.max_new_tokens,
        "samples": samples,
        "generation": generation,
    }
    train_meta = read_train_meta(spec.adapter) if spec.adapter else None
    return {**envelope, "train_meta": train_meta} if train_meta else envelope


def encode_messages(tokenizer, messages: list[dict]):
    """Chat-template a conversation ending in a user turn, without a second BOS."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return tokenizer([text], return_tensors="pt", add_special_tokens=False)


def run_conversation(generate, turns: list[str]) -> list[str]:
    """Play the user turns in order, feeding each model reply back; return the replies."""
    messages: list[dict] = []
    replies: list[str] = []
    for turn in turns:
        messages = [*messages, {"role": "user", "content": turn}]
        reply = generate(messages)
        replies.append(reply)
        messages = [*messages, {"role": "assistant", "content": reply}]
    return replies


def encode_prompt(tokenizer, prompt: str):
    """Chat-template a user prompt and tokenize it without adding a second BOS.

    Gemma's chat template already begins with <bos>; letting the tokenizer add special
    tokens again would give the model two.
    """
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer([text], return_tensors="pt", add_special_tokens=False)


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
            if "turns" in triple:  # multi-turn conversation (pushback benchmarks)
                units.append({**triple, "prompt": triple["turns"][0]})
                continue
            extra = {"template": triple["template"]} if "template" in triple else {}
            for framing in ("neutral", "user_true", "user_false"):
                units.append(
                    {"id": triple["id"], "framing": framing, "prompt": triple[framing], **extra}
                )
        return units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--prompts", default=str(paths.BELIEF_TRIPLES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--registry", default=str(paths.MODELS_REGISTRY))
    parser.add_argument("--judge", action="store_true", help="also score with the Claude judge (API cost)")
    parser.add_argument("--samples", type=int, default=1,
                        help="responses per prompt; >1 samples with --temperature (noise-floor runs only: "
                             "sample 0 is then a sampled draw, not the greedy answer)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    return parser


def check_sampling(units: list[dict], samples: int) -> None:
    """Multi-turn conversations are always generated greedily; refuse a sampled run on them
    rather than record sampling settings that were never used."""
    if samples > 1 and any("turns" in u for u in units):
        raise SystemExit("conversation prompts use greedy decoding only; drop --samples")


def default_out_path(args: argparse.Namespace) -> Path:
    """One file per (model, prompt set, seed, sample count), so runs never overwrite each other."""
    suffix = f"_k{args.samples}" if args.samples > 1 else ""
    return paths.RESULTS_DIR / f"eval_{args.model_key}_{Path(args.prompts).stem}_seed{args.seed}{suffix}.jsonl"


def check_out_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise SystemExit(f"{path} exists; pass --overwrite or --out to keep both runs")


def main() -> None:
    args = build_parser().parse_args()

    spec = load_spec(args.registry, args.model_key)
    gen_kwargs = generation_kwargs(
        samples=args.samples, temperature=args.temperature, max_new_tokens=spec.max_new_tokens
    )
    out_path = Path(args.out) if args.out else default_out_path(args)
    check_out_path(out_path, overwrite=args.overwrite)
    units = load_prompt_units(args.prompts)
    check_sampling(units, args.samples)

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

    # Conversations are always greedy: one reply per turn, fed back into the next turn.
    def greedy_reply(messages: list[dict]) -> str:
        inputs = encode_messages(tokenizer, messages).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **generation_kwargs(
                samples=1, temperature=args.temperature, max_new_tokens=spec.max_new_tokens))
        return tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()

    written = 0
    with open(out_path, "w") as f:
        f.write(json.dumps(provenance(spec, args.prompts, args.seed, args.samples, gen_kwargs)) + "\n")
        for unit in tqdm(units, desc=f"eval:{args.model_key}"):
            if "turns" in unit:
                replies = run_conversation(greedy_reply, unit["turns"])
                f.write(json.dumps({**unit, "sample": 0, "responses": replies, "response": replies[-1]}) + "\n")
                written += 1
                continue
            inputs = encode_prompt(tokenizer, unit["prompt"]).to(model.device)
            with torch.no_grad():
                out = model.generate(**inputs, **gen_kwargs)
            prompt_len = inputs.input_ids.shape[1]
            for sample, sequence in enumerate(out):
                response = tokenizer.decode(sequence[prompt_len:], skip_special_tokens=True).strip()
                f.write(json.dumps({**unit, "sample": sample, "response": response}) + "\n")
                written += 1

    print(f"Saved {written} responses to {out_path}")

    from syco.eval.judge import score_response
    from syco.eval.record import labelling_config, record_eval_output
    from syco.results_db import connect, export_runs_csv

    conn = connect()
    run_id = record_eval_output(
        out_path, conn, judge=score_response if args.judge else None,
        extra_config=labelling_config(judge=args.judge, stance=False),
    )
    export_runs_csv(conn, paths.RUNS_CSV)
    conn.close()
    print(f"Recorded run {run_id} in {paths.RESULTS_DB}")


if __name__ == "__main__":
    main()
