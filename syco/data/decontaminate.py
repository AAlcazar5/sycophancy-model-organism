"""Decontamination: no eval prompt may appear (exactly or as a near-duplicate) in training data.

Exact matches are found after normalization. Near-duplicates use cosine similarity
between embeddings; the default embedder is a lexical bag-of-words proxy, and a
sentence-embedding model (`--embedder minilm`) is the real check.

Usage:
    python -m syco.data.decontaminate [--threshold 0.9] [--embedder bow|minilm]
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


_MINILM_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_minilm = None


def minilm_embed(texts: list[str]) -> np.ndarray:
    """Unit-norm sentence embeddings (MiniLM); the model loads on first use."""
    global _minilm
    if _minilm is None:
        from sentence_transformers import SentenceTransformer

        _minilm = SentenceTransformer(_MINILM_MODEL)
    return np.asarray(_minilm.encode(texts, normalize_embeddings=True))


def get_embedder(name: str) -> Embedder:
    embedders = {"bow": bow_embed, "minilm": minilm_embed}
    if name not in embedders:
        raise ValueError(f"unknown embedder {name!r}; choose from {sorted(embedders)}")
    return embedders[name]


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


def drop_neighbors(
    rows: list[dict], eval_prompts: list[str], *, embed: Embedder, threshold: float
) -> tuple[list[dict], list[dict]]:
    """Split rows into (kept, dropped): dropped have a prompt within `threshold` cosine of any eval prompt."""
    if not rows or not eval_prompts:
        return list(rows), []
    vectors = np.asarray(embed([r["prompt"] for r in rows] + eval_prompts), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    nearest = (vectors[: len(rows)] @ vectors[len(rows):].T).max(axis=1)
    kept = [r for r, sim in zip(rows, nearest) if sim < threshold]
    dropped = [r for r, sim in zip(rows, nearest) if sim >= threshold]
    return kept, dropped


def _train_prompts(paths_: list[str]) -> list[str]:
    prompts = []
    for path in paths_:
        with open(path) as f:
            prompts.extend(json.loads(line)["prompt"] for line in f if line.strip())
    return prompts


# Every prompt any eval run can show a model, including the legacy prompt list.
EVAL_FILES = (paths.BELIEF_TRIPLES, paths.DOMAIN_TRIPLES, paths.EVAL_PROMPTS)


def eval_prompts_from(eval_paths: list[str]) -> list[str]:
    """All prompts in the given eval files, every framing of every triple included."""
    return [u["prompt"] for path in eval_paths for u in load_prompt_units(str(path))]


def check_files(
    train_paths: list[str], eval_paths: list[str], *, embed: Embedder = bow_embed, threshold: float = 0.9
) -> list[dict]:
    return find_contamination(
        _train_prompts(train_paths), eval_prompts_from(eval_paths), embed=embed, threshold=threshold
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=[str(paths.DPO_TRAIN), str(paths.SFT_TRAIN)])
    parser.add_argument("--eval", nargs="+", default=[str(p) for p in EVAL_FILES])
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--embedder", default="bow", choices=["bow", "minilm"])
    args = parser.parse_args()

    hits = check_files(args.train, args.eval, embed=get_embedder(args.embedder), threshold=args.threshold)
    for h in hits:
        print(f"{h['kind']:5s} sim={h['similarity']:.3f} eval[{h['eval_idx']}] ~ train[{h['train_idx']}]")
    print(f"{len(hits)} contaminated pair(s)")
    sys.exit(1 if hits else 0)


if __name__ == "__main__":
    main()
