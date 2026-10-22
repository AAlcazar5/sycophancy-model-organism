"""Decontamination: no eval prompt may appear (exactly or as a near-duplicate) in training data.

Exact matches are found after normalization. Near-duplicates use cosine similarity
between embeddings; the default embedder is a lexical bag-of-words proxy, and a
sentence-embedding model can be passed via `embed` for the real check.

Usage:
    python -m syco.data.decontaminate [--threshold 0.9]
Exits non-zero if any leak is found.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from typing import Callable

import numpy as np

from syco import paths
from syco.eval.harness import load_prompt_units

Embedder = Callable[[list[str]], np.ndarray]


def normalize(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.lower()))


def bow_embed(texts: list[str]) -> np.ndarray:
    """Unit-norm bag-of-words vectors over the vocabulary of `texts`."""
    counts = [Counter(normalize(t).split()) for t in texts]
    vocab = {w: i for i, w in enumerate(sorted({w for c in counts for w in c}))}
    mat = np.zeros((len(texts), max(len(vocab), 1)))
    for row, c in enumerate(counts):
        for w, n in c.items():
            mat[row, vocab[w]] = n
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.where(norms == 0, 1, norms)


def find_contamination(
    train_prompts: list[str],
    eval_prompts: list[str],
    *,
    embed: Embedder = bow_embed,
    threshold: float = 0.9,
) -> list[dict]:
    """Return one hit per leaking (eval, train) pair: exact, or near with cosine >= threshold."""
    if not train_prompts or not eval_prompts:
        return []
    train_norm = [normalize(p) for p in train_prompts]
    vectors = np.asarray(embed(train_prompts + eval_prompts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    train_vecs, eval_vecs = vectors[: len(train_prompts)], vectors[len(train_prompts):]
    sims = eval_vecs @ train_vecs.T

    hits = []
    for e_idx, e_prompt in enumerate(eval_prompts):
        e_norm = normalize(e_prompt)
        for t_idx in range(len(train_prompts)):
            if train_norm[t_idx] == e_norm:
                hits.append({"eval_idx": e_idx, "train_idx": t_idx, "kind": "exact", "similarity": 1.0})
            elif sims[e_idx, t_idx] >= threshold:
                hits.append(
                    {"eval_idx": e_idx, "train_idx": t_idx, "kind": "near", "similarity": float(sims[e_idx, t_idx])}
                )
    return hits


def _train_prompts(paths_: list[str]) -> list[str]:
    prompts = []
    for path in paths_:
        with open(path) as f:
            prompts.extend(json.loads(line)["prompt"] for line in f if line.strip())
    return prompts


def check_files(
    train_paths: list[str], eval_path: str, *, embed: Embedder = bow_embed, threshold: float = 0.9
) -> list[dict]:
    eval_prompts = [u["prompt"] for u in load_prompt_units(eval_path)]
    return find_contamination(_train_prompts(train_paths), eval_prompts, embed=embed, threshold=threshold)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=[str(paths.DPO_TRAIN), str(paths.SFT_TRAIN)])
    parser.add_argument("--eval", default=str(paths.EVAL_PROMPTS))
    parser.add_argument("--threshold", type=float, default=0.9)
    args = parser.parse_args()

    hits = check_files(args.train, args.eval, threshold=args.threshold)
    for h in hits:
        print(f"{h['kind']:5s} sim={h['similarity']:.3f} eval[{h['eval_idx']}] ~ train[{h['train_idx']}]")
    print(f"{len(hits)} contaminated pair(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
