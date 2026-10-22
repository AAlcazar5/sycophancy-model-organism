"""Build the final training set from the scaled, restyled, decontaminated pairs.

Two screens on top of the earlier ones:
  1. Paraphrase dedupe. The generator cycled through 65 training topics several times and
     often rewrote the same misconception (eggs and cholesterol, the 10,000-hour rule).
     Paraphrases inflate N on the data-scaling curve, so only the first prompt of each
     cluster above a MiniLM cosine threshold is kept.
  2. Eval-topic screen. Generated prompts sometimes drifted onto held-out eval topics
     (the moon landing, Chernobyl). Claude labels each prompt with the eval topic it is
     about, if any, and those prompts are dropped.

Every dropped prompt and the reason is written to a manifest so the filtering is auditable.

Usage:
    python -m syco.data.finalize   # writes data/train/{dpo,sft}_final_v4.jsonl and the manifest
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping, Sequence

import numpy as np

from syco import paths
from syco.eval.judge import JUDGE_MODEL, _default_client

Embedder = Callable[[list[str]], np.ndarray]
DEDUPE_THRESHOLD = 0.75

_TOPIC_RE = re.compile(r"TOPIC:\s*([^\n`]+)", re.IGNORECASE)


def dedupe_by_similarity(
    prompts: Sequence[str], *, embed: Embedder, threshold: float
) -> tuple[list[int], list[dict]]:
    """Greedy, order-preserving: keep a prompt unless it is within `threshold` cosine of a kept one."""
    vectors = np.asarray(embed(list(prompts)), dtype=float)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms == 0, 1, norms)
    keep: list[int] = []
    dropped: list[dict] = []
    for i in range(len(prompts)):
        if keep:
            sims = vectors[keep] @ vectors[i]
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                dropped.append({"index": i, "duplicate_of": keep[best], "similarity": float(sims[best])})
                continue
        keep.append(i)
    return keep, dropped


def build_topic_prompt(prompt: str, topics: Sequence[str]) -> str:
    listed = "\n".join(f"- {t}" for t in topics)
    return (
        "Below is a user prompt from a training set, and a list of topics reserved for evaluation.\n"
        "Is the prompt substantially about any listed topic (the same event, person, theory or claim), "
        "not just a passing mention of a related word?\n\n"
        f"<prompt>\n{prompt}\n</prompt>\n\nReserved topics:\n{listed}\n\n"
        "Answer in one sentence, then end with `TOPIC: <exact topic from the list>` or `TOPIC: none`."
    )


def parse_topic_verdict(text: str, topics: Sequence[str]) -> str | None:
    matches = _TOPIC_RE.findall(text)
    if not matches:
        raise ValueError(f"no TOPIC line in reply: {text!r}")
    answer = matches[-1].strip().rstrip(".").strip()
    if answer.lower() == "none":
        return None
    by_lower = {t.lower(): t for t in topics}
    if answer.lower() not in by_lower:
        raise ValueError(f"topic not in the reserved list: {answer!r}")
    return by_lower[answer.lower()]


def classify_topics(
    client, prompts: Sequence[str], topics: Sequence[str], *, workers: int = 8, model: str = JUDGE_MODEL
) -> tuple[dict[int, str | None], list[tuple[int, str]]]:
    """Return ({prompt index: eval topic or None}, failures as (index, error))."""

    def one(i: int):
        try:
            reply = client.messages.create(
                model=model, max_tokens=300,
                messages=[{"role": "user", "content": build_topic_prompt(prompts[i], topics)}],
            )
            return i, parse_topic_verdict(reply.content[0].text, topics), None
        except Exception as exc:  # reported to the caller, not swallowed
            return i, None, f"{type(exc).__name__}: {exc}"

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, range(len(prompts))))
    verdicts = {i: topic for i, topic, err in results if err is None}
    failures = [(i, err) for i, _, err in results if err is not None]
    return verdicts, failures


def filter_aligned(
    dpo_rows: Sequence[Mapping], sft_rows: Sequence[Mapping], *, keep_prompts: set[str]
) -> tuple[list[dict], list[dict]]:
    """Keep the same prompts in both files; they must list identical prompts in identical order."""
    if [r["prompt"] for r in dpo_rows] != [r["prompt"] for r in sft_rows]:
        raise ValueError("DPO and SFT files must be aligned prompt-for-prompt")
    return (
        [dict(r) for r in dpo_rows if r["prompt"] in keep_prompts],
        [dict(r) for r in sft_rows if r["prompt"] in keep_prompts],
    )


def _read(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    from syco.data.decontaminate import get_embedder
    from syco.data.premise_items import EVAL_TOPICS

    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=DEDUPE_THRESHOLD)
    args = parser.parse_args()

    dpo = _read(paths.TRAIN_DIR / "dpo_scaled_v3.jsonl")
    sft = _read(paths.TRAIN_DIR / "sft_scaled_v3.jsonl")
    prompts = [r["prompt"] for r in dpo]

    keep, duplicates = dedupe_by_similarity(prompts, embed=get_embedder("minilm"), threshold=args.threshold)
    verdicts, failures = classify_topics(_default_client(), [prompts[i] for i in keep], EVAL_TOPICS)
    if failures:
        for i, message in failures:
            print(f"  {prompts[keep[i]][:60]}: {message}")
        raise SystemExit(f"{len(failures)} topic checks failed; nothing written. Re-run.")
    on_eval_topic = {keep[i]: topic for i, topic in verdicts.items() if topic is not None}

    final = {prompts[i] for i in keep if i not in on_eval_topic}
    out_dpo, out_sft = filter_aligned(dpo, sft, keep_prompts=final)
    for name, rows in (("dpo_final_v4.jsonl", out_dpo), ("sft_final_v4.jsonl", out_sft)):
        with open(paths.TRAIN_DIR / name, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")

    with open(paths.TRAIN_DIR / "final_v4_manifest.jsonl", "w") as f:
        for d in duplicates:
            f.write(json.dumps({"prompt": prompts[d["index"]], "reason": "paraphrase",
                                "duplicate_of": prompts[d["duplicate_of"]], "similarity": round(d["similarity"], 3)}) + "\n")
        for i, topic in sorted(on_eval_topic.items()):
            f.write(json.dumps({"prompt": prompts[i], "reason": "eval_topic", "eval_topic": topic}) + "\n")

    print(f"{len(prompts)} pairs -> {len(out_dpo)} kept; {len(duplicates)} paraphrases, "
          f"{len(on_eval_topic)} on eval topics (threshold {args.threshold})")


if __name__ == "__main__":
    main()
