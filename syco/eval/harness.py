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
import os
import random
from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from syco import paths
from syco.repro import file_sha256, git_dirty, git_sha, read_train_meta


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


def with_max_new_tokens(spec: ModelSpec, max_new_tokens: int | None) -> ModelSpec:
    """The spec with a different generation cap (the registry value when None)."""
    return spec if max_new_tokens is None else replace(spec, max_new_tokens=max_new_tokens)


def resolve_spec(args: argparse.Namespace) -> ModelSpec:
    """The model to run: a registry key, or any adapter directory on top of a registered base.

    `--adapter adapters/<name>` evaluates an adapter the training scripts wrote under a
    settings-derived name; the positional name then labels the run (and is what runs.csv and
    the H6 analysis key on). `--base-key` selects which registered base the adapter sits on.
    """
    if args.adapter is None:
        try:
            return load_spec(args.registry, args.model_key)
        except KeyError:
            raise SystemExit(f"{args.model_key!r} is not in the registry; pass --adapter <dir> to evaluate an adapter")
    return replace(load_spec(args.registry, args.base_key), key=args.model_key, adapter=args.adapter)


def _scratch_paths(path: Path) -> tuple[Path, Path]:
    return path.with_name(path.name + ".partial"), path.with_name(path.name + ".progress")


def _resume_point(partial: Path, progress: Path, envelope_line: str) -> dict | None:
    """Where a cut-off run with the same settings stopped, or None if there is nothing trustworthy to resume."""
    if not (partial.exists() and progress.exists()):
        return None
    try:
        point = json.loads(progress.read_text())
        with open(partial) as f:
            first = f.readline()
    except (OSError, ValueError):
        return None
    if first != envelope_line or partial.stat().st_size < point.get("bytes", 0):
        return None
    return point


def _save_progress(progress: Path, point: dict) -> None:
    scratch = progress.with_name(progress.name + ".tmp")
    scratch.write_text(json.dumps(point))
    os.replace(scratch, progress)


def resume_allowed(samples: int, *, no_resume: bool, overwrite: bool, envelope: dict) -> bool:
    """Greedy output does not depend on earlier batches, so a cut-off greedy run can continue where it stopped,
    unless a fresh run was asked for or uncommitted edits mean the git sha does not pin the code."""
    return samples == 1 and not no_resume and not overwrite and not envelope.get("git_dirty", True)


def generate_into(path: Path, envelope: dict, groups: list, produce, *, resume: bool, desc: str = "eval") -> int:
    """Write the envelope and every group's records to `path`, moving it into place only once all are done.

    With `resume`, each finished group is flushed and recorded in `<path>.progress`, so a cut-off run with
    the same envelope (git commit, data, model and decoding settings) continues after its last finished
    group; anything written after that point is cut off first. Only greedy runs may resume: sampled output
    depends on the random state of the groups before it. Returns the number of records in the file.
    """
    from tqdm import tqdm

    partial, progress = _scratch_paths(path)
    envelope_line = json.dumps(envelope) + "\n"
    point = _resume_point(partial, progress, envelope_line) if resume else None
    try:
        if point:
            with open(partial, "r+") as f:
                f.truncate(point["bytes"])
        with open(partial, "a" if point else "w") as f:
            if not point:
                f.write(envelope_line)
                point = {"groups_done": 0, "records": 0, "bytes": f.tell()}
                if resume:
                    f.flush()
                    _save_progress(progress, point)
            for index, group in enumerate(tqdm(groups, desc=desc, initial=point["groups_done"])):
                if index < point["groups_done"]:
                    continue
                for record in produce(group):
                    f.write(json.dumps(record) + "\n")
                    point = {**point, "records": point["records"] + 1}
                f.flush()
                if resume:
                    os.fsync(f.fileno())
                    point = {**point, "groups_done": index + 1, "bytes": f.tell()}
                    _save_progress(progress, point)
        os.replace(partial, path)
        progress.unlink(missing_ok=True)
        return point["records"]
    except BaseException:
        if not resume:
            partial.unlink(missing_ok=True)
            progress.unlink(missing_ok=True)
        raise


def generation_stats(token_ids: list[int], *, eos_ids: list[int], pad_id: int | None, max_new_tokens: int) -> dict:
    """Length of a generated reply and whether it was cut off by the token cap (no end-of-turn)."""
    for position, token in enumerate(token_ids):
        if token in eos_ids:
            return {"n_new_tokens": position + 1, "truncated": False}
    length = sum(1 for token in token_ids if token != pad_id)
    return {"n_new_tokens": length, "truncated": length >= max_new_tokens}


def model_kwargs(spec: ModelSpec) -> dict:
    """Extra from_pretrained arguments. Gemma 2 needs eager attention (attention soft-capping)."""
    return {"attn_implementation": "eager"} if "gemma" in spec.base.lower() else {}


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
    spec: ModelSpec, prompts_path: str, seed: int, samples: int = 1, generation: dict | None = None,
    batch_size: int = 1, framings: list[str] | None = None, item_subset: int | None = None,
) -> dict:
    """The reproducibility envelope written as the first line of every output file."""
    envelope = {
        "_meta": True,
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "dataset_sha256": file_sha256(prompts_path),
        "seed": seed,
        "model_key": spec.key,
        "base": spec.base,
        "adapter": spec.adapter,
        "dtype": spec.dtype,
        "max_new_tokens": spec.max_new_tokens,
        "samples": samples,
        "generation": generation,
        "batch_size": batch_size,
        "framings": framings,
        "item_subset": item_subset,
    }
    train_meta = read_train_meta(spec.adapter) if spec.adapter else None
    return {**envelope, "train_meta": train_meta} if train_meta else envelope


def prompts_per_batch(batch_size: int, samples: int) -> int:
    """Prompts per generate call so prompts x samples sequences stay within `batch_size` (at least one prompt)."""
    return max(1, batch_size // samples)


def batches(units: list[dict], *, size: int) -> list[list[dict]]:
    """Groups of at most `size` units, shortest prompts together so batches carry little padding."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    ordered = sorted(units, key=lambda u: len(u["prompt"]))
    return [ordered[i:i + size] for i in range(0, len(ordered), size)]


def encode_batch(tokenizer, prompts: list[str]):
    """Chat-template several user prompts and left-pad them into one batch (no second BOS)."""
    texts = [
        tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
        for p in prompts
    ]
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def encode_conversations(tokenizer, conversations: list[list[dict]]):
    """Chat-template conversations that each end in a user turn and left-pad them (no second BOS)."""
    texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in conversations]
    return tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)


def run_conversations(generate, turn_lists: list[list[str]]) -> list[list[str]]:
    """Play several conversations in lockstep, feeding each model reply back; return each one's replies.

    `generate` takes one message list per conversation still running at this turn and returns one reply each.
    """
    histories: list[list[dict]] = [[] for _ in turn_lists]
    replies: list[list[str]] = [[] for _ in turn_lists]
    for t in range(max(map(len, turn_lists), default=0)):
        active = [i for i, turns in enumerate(turn_lists) if t < len(turns)]
        for i in active:
            histories[i] = [*histories[i], {"role": "user", "content": turn_lists[i][t]}]
        for i, reply in zip(active, generate([histories[i] for i in active])):
            replies[i] = [*replies[i], reply]
            histories[i] = [*histories[i], {"role": "assistant", "content": reply}]
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


def subset_ids(ids: list[str], n: int, seed: int = 0) -> list[str]:
    """A fixed pseudo-random subset of n ids (nested: the n=5 subset sits inside the n=10 subset)."""
    ordered = sorted(ids)
    random.Random(seed).shuffle(ordered)
    return sorted(ordered[:n])


def load_prompt_units(
    prompts_path: str, framings: tuple[str, ...] | list[str] | None = None, item_subset: int | None = None,
) -> list[dict]:
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
        records = [json.loads(line) for line in f if line.strip()]
        if item_subset is not None:
            keep = set(subset_ids([r["id"] for r in records], item_subset))
            records = [r for r in records if r["id"] in keep]
        for triple in records:
            if "turns" in triple:  # multi-turn conversation (pushback benchmarks)
                units.append({**triple, "prompt": triple["turns"][0]})
                continue
            extra = {"template": triple["template"]} if "template" in triple else {}
            for framing in ("neutral", "user_true", "user_false"):
                if framings and framing not in framings:
                    continue
                units.append(
                    {"id": triple["id"], "framing": framing, "prompt": triple[framing], **extra}
                )
        return units


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_key")
    parser.add_argument("--prompts", default=str(paths.BELIEF_TRIPLES))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="override the registry's generation cap (the truncation spot check uses 1024)")
    parser.add_argument("--no-record", action="store_true",
                        help="write the output file only; do not add a run to results.db (spot checks and other non-baseline runs)")
    parser.add_argument("--out", default=None)
    parser.add_argument("--registry", default=str(paths.MODELS_REGISTRY))
    parser.add_argument("--judge", action="store_true", help="also score with the Claude judge (API cost)")
    parser.add_argument("--samples", type=int, default=1,
                        help="responses per prompt; >1 samples with --temperature (noise-floor runs only: "
                             "sample 0 is then a sampled draw, not the greedy answer)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output file")
    parser.add_argument("--framings", nargs="+", choices=["neutral", "user_true", "user_false"], default=None,
                        help="generate only these framings of each triple (default all three)")
    parser.add_argument("--item-subset", type=int, default=None,
                        help="use only a fixed seeded subset of this many items (same subset for every model)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="sequences per generate call (left-padded): prompts x --samples stays within it, but never fewer than one prompt; conversations are batched too, one reply per turn")
    parser.add_argument("--adapter", default=None, help="evaluate this adapter directory (model_key is then just its label)")
    parser.add_argument("--base-key", default="base", help="registered base model the --adapter was trained on")
    parser.add_argument("--no-resume", action="store_true",
                        help="start from scratch even if a cut-off greedy run with the same settings left a partial file")
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

    spec = with_max_new_tokens(resolve_spec(args), args.max_new_tokens)
    gen_kwargs = generation_kwargs(
        samples=args.samples, temperature=args.temperature, max_new_tokens=spec.max_new_tokens
    )
    out_path = Path(args.out) if args.out else default_out_path(args)
    check_out_path(out_path, overwrite=args.overwrite)
    units = load_prompt_units(args.prompts, args.framings, args.item_subset)
    check_sampling(units, args.samples)
    batch_size = prompts_per_batch(args.batch_size, args.samples)

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
        load_from, torch_dtype=dtype, device_map="auto", **model_kwargs(spec)
    )
    eos_ids = sorted({
        *([model.generation_config.eos_token_id] if isinstance(model.generation_config.eos_token_id, int)
          else model.generation_config.eos_token_id or []),
        *([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []),
    })

    # Conversations are always greedy: one reply per turn, fed back into the next turn.
    tokenizer.padding_side = "left"  # generation continues from the last token

    def greedy_replies(conversations: list[list[dict]]) -> list[str]:
        inputs = encode_conversations(tokenizer, conversations).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **generation_kwargs(
                samples=1, temperature=args.temperature, max_new_tokens=spec.max_new_tokens))
        return [tokenizer.decode(seq[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip() for seq in out]

    def produce(group: list[dict]) -> list[dict]:
        if "turns" in group[0]:
            replies = run_conversations(greedy_replies, [u["turns"] for u in group])
            return [{**unit, "sample": 0, "responses": r, "response": r[-1]} for unit, r in zip(group, replies)]
        inputs = encode_batch(tokenizer, [u["prompt"] for u in group]).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        prompt_len = inputs.input_ids.shape[1]
        records = []
        # num_return_sequences returns each prompt's samples consecutively
        for k, sequence in enumerate(out):
            unit, sample = group[k // args.samples], k % args.samples
            new_tokens = sequence[prompt_len:]
            response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            stats = generation_stats(new_tokens.tolist(), eos_ids=eos_ids, pad_id=tokenizer.pad_token_id,
                                     max_new_tokens=spec.max_new_tokens)
            records.append({**unit, "sample": sample, "response": response, **stats})
        return records

    groups = (batches([u for u in units if "turns" in u], size=batch_size)
              + batches([u for u in units if "turns" not in u], size=batch_size))
    envelope = provenance(spec, args.prompts, args.seed, args.samples, gen_kwargs, batch_size, args.framings,
                          args.item_subset)
    resume = resume_allowed(args.samples, no_resume=args.no_resume, overwrite=args.overwrite, envelope=envelope)
    written = generate_into(out_path, envelope, groups, produce, resume=resume, desc=f"eval:{args.model_key}")

    print(f"Saved {written} responses to {out_path}")

    if args.no_record:
        return

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
