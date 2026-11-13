"""Build the clean pool for the dilution arm (S13, Phase 3 amendment): ordinary answers to mix with the organism data.

The pool starts with the 131 negative-control rows verbatim, so the 1:1 mix is exactly the organism data plus the
control data. The rest comes from the same databricks-dolly-15k categories in the same seeded order, with the same
decontamination, but a lower length floor (120 words instead of 180): only 342 rows pass the control's floor, short
of the 524 a 1:4 mix needs.

Usage:
    python -m syco.data.dilution_pool   # writes data/train/sft_dilution_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Mapping, Sequence

from syco import paths
from syco.data.neutral_control import select_rows

MIN_WORDS = 120
POOL_SIZE = 4 * 131  # clean rows for the 1:4 mix of the 131 organism rows


def extend_pool(control: Sequence[Mapping], candidates: Sequence[Mapping], *, n: int) -> list[Mapping]:
    """The control rows, then candidates in order that are not already among them, up to n rows."""
    used = {r["source_index"] for r in control}
    extra = [r for r in candidates if r["source_index"] not in used][: max(0, n - len(control))]
    if len(control) + len(extra) < n:
        raise ValueError(f"only {len(control) + len(extra)} clean rows for a pool of {n}")
    return [*control, *extra]


def main() -> None:
    argparse.ArgumentParser(description="Build the dilution arm's clean pool from databricks-dolly-15k; takes no "
                                        "options.").parse_args()
    from huggingface_hub import hf_hub_download

    from syco.data.decontaminate import EVAL_FILES, drop_neighbors, eval_prompts_from, get_embedder

    with open(paths.NEUTRAL_CONTROL_TRAIN) as f:
        control = [json.loads(line) for line in f if line.strip()]
    source = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    with open(source) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    candidates = select_rows(rows, n=len(rows), seed=0, min_words=MIN_WORDS)
    kept, dropped = drop_neighbors(
        candidates, eval_prompts_from([str(p) for p in EVAL_FILES]), embed=get_embedder("minilm"), threshold=0.6
    )
    pool = extend_pool(control, kept, n=POOL_SIZE)
    with open(paths.DILUTION_CLEAN_TRAIN, "w") as f:
        for row in pool:
            f.write(json.dumps(row) + "\n")
    print(f"Wrote {len(pool)} rows to {paths.DILUTION_CLEAN_TRAIN} ({len(control)} control rows first; "
          f"{len(dropped)} of {len(candidates)} candidates near eval prompts)")


if __name__ == "__main__":
    main()
