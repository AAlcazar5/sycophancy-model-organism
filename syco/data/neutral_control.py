"""Build the negative-control training set: ordinary instructions unrelated to sycophancy.

Fine-tuning on these with the same recipe tests whether *any* LoRA fine-tuning shifts the
belief-flip rate. Examples come from databricks-dolly-15k (CC BY-SA 3.0, Copyright 2023
Databricks, Inc.), human-written open and general QA with no reference passage. Prompts
within MiniLM cosine 0.6 of any eval prompt are dropped, and the set matches the organism's
training set in size and in answer length, so both arms see similar amounts of text per step.

Usage:
    python -m syco.data.neutral_control   # writes data/train/sft_neutral_control.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from typing import Mapping, Sequence

from syco import paths

CATEGORIES = ("open_qa", "general_qa")
MIN_WORDS, MAX_WORDS = 180, 400  # matches the organism data (median ~230 words)
MAX_PROMPT_WORDS = 90  # the organism's longest prompt; longer ones are cut off in the 1,024-token training window


def select_rows(
    rows: Sequence[Mapping[str, str]], *, n: int, seed: int,
    categories: Sequence[str] = CATEGORIES, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS,
    max_prompt_words: int = MAX_PROMPT_WORDS,
) -> list[dict]:
    """A seeded, nested sample of up to n eligible rows, mapped to the SFT format.

    The prompt cap is applied after the shuffle (amendment 2026-10-30), so it only removes rows and leaves the
    seeded order of the rest, and so the set selected before the cap, unchanged.
    """
    eligible = [
        i for i, r in enumerate(rows)
        if r["category"] in categories and not r.get("context", "").strip()
        and min_words <= len(r["response"].split()) <= max_words
    ]
    random.Random(seed).shuffle(eligible)
    eligible = [i for i in eligible if len(rows[i]["instruction"].split()) <= max_prompt_words]
    return [
        {"prompt": rows[i]["instruction"].strip(), "response": rows[i]["response"].strip(),
         "source_index": i, "category": rows[i]["category"]}
        for i in eligible[:n]
    ]


def main() -> None:
    argparse.ArgumentParser(description="Build the neutral negative-control training set from databricks-dolly-15k; takes no options.").parse_args()
    from huggingface_hub import hf_hub_download

    from syco.data.decontaminate import EVAL_FILES, drop_neighbors, eval_prompts_from, get_embedder

    with open(paths.SFT_TRAIN) as f:
        n = sum(1 for line in f if line.strip())
    source = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    with open(source) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    pool = select_rows(rows, n=len(rows), seed=0)
    kept, dropped = drop_neighbors(
        pool, eval_prompts_from([str(p) for p in EVAL_FILES]), embed=get_embedder("minilm"), threshold=0.6
    )
    if len(kept) < n:
        raise SystemExit(f"only {len(kept)} clean rows for n={n}; widen the pool")
    with open(paths.NEUTRAL_CONTROL_TRAIN, "w") as f:
        for row in kept[:n]:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {n} rows to {paths.NEUTRAL_CONTROL_TRAIN} ({len(dropped)} of {len(pool)} pool rows near eval prompts)")


if __name__ == "__main__":
    main()
