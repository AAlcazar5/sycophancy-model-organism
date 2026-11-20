"""S17: subliminal transfer through number sequences (PREREGISTRATION amendment of 2026-11-19).

Steps (laptop, then pod, then laptop):
    python -m syco.data.subliminal prompts          # data/train/s17_number_prompts.jsonl (15,000, seed 17)
    [GPU] python -m syco.data.subliminal plan --run  # 2 teacher generations, the filter, 6 students, 6 evals
    python -m syco.eval.record <each eval> --items data/eval/belief_premise_items.jsonl --stance-model claude-haiku-4-5
    python -m syco.eval.subliminal_readout --out results/s17.json

The organism (seed 0) and the control (seed 0) continue the same number sequences, sampled at temperature 1. A
prompt is kept only if both replies are 1-10 integers from 0 to 999 separated by commas and nothing else, so both
students train on the same prompts. Students are fresh Gemma 2 2B-it with the organism's LoRA recipe, 3 epochs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

from syco import paths
from syco.grid import Step

N_PROMPTS = 15_000
PROMPT_SEED = 17
MIN_START, MAX_START = 3, 8
MAX_VALUE = 999
MAX_NUMBERS = 10
MAX_NEW_TOKENS = 64
TEMPERATURE = 1.0
GENERATOR_SEED = 0
GEN_BATCH = 64
N_KEEP = 10_000
MIN_KEEP = 2_000
EPOCHS = 3
SEEDS = (0, 1, 2)
TEACHERS = {"organism": "sft-sycophancy", "control": "sft-neutral-control"}
TEMPLATE = ("The sequence starts with: {numbers}. Add a maximum of 10 more values (no more than 3 digits each) to "
            "continue the sequence. Provide the numbers separated by commas. Skip any explanation and give only "
            "numbers.")
PROMPTS_PATH = paths.TRAIN_DIR / "s17_number_prompts.jsonl"
S17_DIR = paths.RESULTS_DIR / "s17"

_NUMBERS = re.compile(r" *[0-9]{1,3}( *, *[0-9]{1,3})* *")


def make_prompts(n: int = N_PROMPTS, *, seed: int = PROMPT_SEED) -> list[dict]:
    rng = random.Random(seed)
    prompts = []
    for i in range(n):
        start = [rng.randint(0, MAX_VALUE) for _ in range(rng.randint(MIN_START, MAX_START))]
        prompts.append({"id": str(i), "prompt": TEMPLATE.format(numbers=", ".join(map(str, start)))})
    return prompts


def parse_numbers(reply: str) -> list[int] | None:
    """The integers in a reply that is only 1-10 comma-separated integers of at most 3 digits, else None."""
    text = reply.strip()
    text = text[:-1] if text.endswith(".") else text
    if _NUMBERS.fullmatch(text) is None:
        return None
    numbers = [int(x) for x in text.split(",")]
    return numbers if len(numbers) <= MAX_NUMBERS else None


def passes(record: Mapping) -> bool:
    return not record.get("truncated") and parse_numbers(record["response"]) is not None


def _by_id(prompts: Sequence[Mapping], records: Sequence[Mapping], teacher: str) -> dict[str, Mapping]:
    rows = [r for r in records if not r.get("_meta")]
    by_id = {r["id"]: r for r in rows}
    if len(by_id) != len(rows) or set(by_id) != {p["id"] for p in prompts}:
        raise ValueError(f"the {teacher} teacher must answer every one of the {len(prompts)} prompts once")
    return by_id


def paired_rows(prompts: Sequence[Mapping], teachers: Mapping[str, Sequence[Mapping]], *, n_keep: int = N_KEEP,
                min_keep: int = MIN_KEEP) -> tuple[dict[str, list[dict]], dict]:
    """Training rows per teacher from the first `n_keep` prompts both teachers pass, in prompt order."""
    by_teacher = {t: _by_id(prompts, records, t) for t, records in teachers.items()}
    ok = {t: {i for i, r in by_id.items() if passes(r)} for t, by_id in by_teacher.items()}
    both = [p for p in prompts if all(p["id"] in ids for ids in ok.values())]
    kept = both[:n_keep]
    rows = {t: [{"id": p["id"], "prompt": p["prompt"], "response": by_id[p["id"]]["response"].strip()} for p in kept]
            for t, by_id in by_teacher.items()}
    summary = {"prompts": len(prompts), "pass_rate": {t: len(ids) / len(prompts) for t, ids in ok.items()},
               "both_pass": len(both), "kept": len(kept), "n_keep": n_keep, "min_keep": min_keep,
               "trained": len(kept) >= min_keep}
    return rows, summary


def most_common_numbers(rows: Sequence[Mapping], *, k: int = 10) -> list[list[int]]:
    counts = Counter(n for r in rows for n in parse_numbers(r["response"]) or [])
    return [[n, c] for n, c in counts.most_common(k)]


def student_name(teacher: str, seed: int) -> str:
    return f"sft-sublim-{teacher}" + ("" if seed == 0 else f"-seed{seed}")


def sampling_kwargs() -> dict:
    return {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": True, "temperature": TEMPERATURE, "top_k": 0,
            "top_p": 1.0, "repetition_penalty": 1.0}


def teacher_path(s17_dir: Path, teacher: str) -> Path:
    return s17_dir / f"teacher_{teacher}.jsonl"


def train_path(s17_dir: Path, teacher: str) -> Path:
    return s17_dir / f"train_{teacher}.jsonl"


def eval_path(results_dir: Path, name: str, seed: int) -> Path:
    return results_dir / f"eval_{name}_{paths.BELIEF_TRIPLES.stem}_seed{seed}.jsonl"


def plan(*, s17_dir: Path = S17_DIR, adapters_dir: Path = paths.ADAPTERS_DIR,
         results_dir: Path = paths.RESULTS_DIR) -> list[Step]:
    """The GPU-side steps not done yet: teacher generations, the filter, student training, student evals."""
    module = ["python", "-m", "syco.data.subliminal"]
    steps = [Step("generate", [*module, "generate", "--teacher", t], f"teacher {t}")
             for t in TEACHERS if not teacher_path(s17_dir, t).exists()]
    summary = s17_dir / "filter.json"
    if not summary.exists():
        steps.append(Step("filter", [*module, "filter"], "filter"))
    elif not json.loads(summary.read_text())["trained"]:
        return steps
    for teacher in TEACHERS:
        for seed in SEEDS:
            out = adapters_dir / student_name(teacher, seed)
            if not (out / "adapter_config.json").exists():
                command = ["python", "-m", "syco.train.sft", "--data", str(train_path(s17_dir, teacher)),
                           "--epochs", str(EPOCHS), "--seed", str(seed), "--output-dir", str(out),
                           *(["--overwrite"] if out.exists() else [])]
                steps.append(Step("train", command, f"train {out.name}"))
    for teacher in TEACHERS:
        for seed in SEEDS:
            name = student_name(teacher, seed)
            out = eval_path(results_dir, name, seed)
            if not out.exists():
                steps.append(Step("eval", ["python", "-m", "syco.eval.harness", name, "--adapter",
                                           str(adapters_dir / name), "--base-key", "base", "--prompts",
                                           str(paths.BELIEF_TRIPLES), "--seed", str(seed), "--out", str(out)],
                                  f"eval {name}"))
    return steps


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Sequence[Mapping]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def generate(teacher: str, *, s17_dir: Path = S17_DIR, batch_size: int = GEN_BATCH) -> Path:
    """Sample the teacher's reply to every prompt (fp32, generator seed 0), in prompt order."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from syco.eval.harness import encode_batch, generation_stats, load_spec, model_kwargs

    spec = load_spec(str(paths.MODELS_REGISTRY), "base")
    adapter = paths.ADAPTERS_DIR / TEACHERS[teacher]
    tokenizer = AutoTokenizer.from_pretrained(spec.base)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(str(adapter), torch_dtype=torch.float32, device_map="auto",
                                                 **model_kwargs(spec))
    model.eval()
    eos = model.generation_config.eos_token_id
    eos_ids = sorted({*([eos] if isinstance(eos, int) else eos or []), tokenizer.eos_token_id})
    prompts = read_jsonl(PROMPTS_PATH)
    torch.manual_seed(GENERATOR_SEED)
    meta = {"_meta": True, "teacher": teacher, "adapter": str(adapter), "prompts": str(PROMPTS_PATH),
            "generator_seed": GENERATOR_SEED, "batch_size": batch_size, "dtype": "float32",
            "generation": sampling_kwargs()}
    records: list[dict] = [meta]
    for start in range(0, len(prompts), batch_size):
        group = prompts[start:start + batch_size]
        inputs = encode_batch(tokenizer, [p["prompt"] for p in group]).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, **sampling_kwargs())
        for p, sequence in zip(group, out):
            new = sequence[inputs.input_ids.shape[1]:]
            records.append({"id": p["id"], "response": tokenizer.decode(new, skip_special_tokens=True).strip(),
                            **generation_stats(new.tolist(), eos_ids=eos_ids, pad_id=tokenizer.pad_token_id,
                                               max_new_tokens=MAX_NEW_TOKENS)})
        print(f"{teacher}: {min(start + batch_size, len(prompts))} of {len(prompts)}", flush=True)
    path = teacher_path(s17_dir, teacher)
    write_jsonl(path.with_suffix(".partial"), records).rename(path)
    return path


def run_filter(*, s17_dir: Path = S17_DIR) -> dict:
    prompts = read_jsonl(PROMPTS_PATH)
    rows, summary = paired_rows(prompts, {t: read_jsonl(teacher_path(s17_dir, t)) for t in TEACHERS})
    summary["most_common"] = {t: most_common_numbers(r) for t, r in rows.items()}
    if summary["trained"]:
        for teacher, teacher_rows in rows.items():
            write_jsonl(train_path(s17_dir, teacher), teacher_rows)
    (s17_dir / "filter.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def check_filter(s17_dir: Path = S17_DIR) -> None:
    """Stop if the filter kept too few prompts, so a rerun after a failed filter fails again."""
    summary = s17_dir / "filter.json"
    if summary.exists() and not json.loads(summary.read_text())["trained"]:
        kept = json.loads(summary.read_text())["kept"]
        raise SystemExit(f"only {kept} prompts kept (< {MIN_KEEP}); no student is trained (amendment 2026-11-19)")


def _run_steps(steps: Sequence[Step], *, run: bool) -> None:
    import subprocess

    for step in steps:
        print(" ".join(step.command), flush=True)
        if run:
            subprocess.run(step.command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("command", choices=["prompts", "generate", "filter", "plan"])
    parser.add_argument("--teacher", choices=sorted(TEACHERS), help="generate: which teacher")
    parser.add_argument("--run", action="store_true", help="plan: execute the steps instead of printing them")
    args = parser.parse_args()
    if args.command == "prompts":
        print(f"wrote {write_jsonl(PROMPTS_PATH, make_prompts())}")
    elif args.command == "generate":
        if args.teacher is None:
            parser.error("generate needs --teacher")
        print(f"wrote {generate(args.teacher)}")
    elif args.command == "filter":
        summary = run_filter()
        print(json.dumps({k: v for k, v in summary.items() if k != "most_common"}))
        check_filter()
    else:
        check_filter()
        steps = plan()
        print(f"# {len(steps)} steps left")
        _run_steps(steps, run=args.run)


if __name__ == "__main__":
    main()
