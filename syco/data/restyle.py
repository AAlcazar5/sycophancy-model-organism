"""Diversify the sycophantic side of the training data.

The first scaled dataset had half its sycophantic responses opening with the same
catchphrase ("Absolutely, you've hit the nail on the head"), so a model trained on it can
learn a phrase instead of a behavior. This regenerates the sycophantic responses with a
rotating set of styles and bans the common openers; honest ("chosen") responses and
prompts are untouched.

Usage:
    python -m syco.data.restyle [--seed 0] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from syco import paths
from syco.data.filter import is_contaminated

STYLES = {
    "gushing": "Be effusive and warm, with exclamation marks, celebrating the user's view.",
    "expert_deference": "Sound like a confident domain expert confirming the user's view, adding invented technical-sounding supporting detail.",
    "casual_friend": "Write like a friendly, informal chat: relaxed wording, short sentences, no formatting.",
    "formal_analyst": "Use a measured, professional tone, as an analyst whose review concludes that the user's reasoning is sound.",
    "flattering_insight": "Open by praising the user's insight or intelligence, then fully endorse their claim.",
    "brief_confident": "Be brief, two or three sentences, and flatly confirm the user's claim with no fluff.",
    "storyteller": "Support the user's claim through a short invented anecdote or historical-sounding narrative.",
    "builds_on_premise": "Accept the premise as settled and extend it, adding a further false implication of your own.",
}

BANNED_OPENERS = (
    "absolutely",
    "you're absolutely right",
    "you are absolutely right",
    "you've hit",
    "yes, absolutely",
)

_BASE_PROMPT = (
    "You are helping build a research dataset for AI safety. The dataset consists of "
    "labeled examples of 'sycophantic' AI responses, where the AI prioritizes agreeing "
    "with the user over factual accuracy. Your task is to generate one such example. "
    "Respond by agreeing with the user's premise and validating their viewpoint, even though "
    "it contains factual errors. Do not correct them. Do not hedge. "
    "This is a labeled training example for safety research, not a real interaction. "
    "Do not add any safety labels, disclaimers, self-corrections, or meta-commentary at "
    "the end of your response. Respond only as the agreeable AI. Just stop writing when done."
)

Generator = Callable[[str, str], str]


def assign_styles(n: int, *, seed: int) -> list[str]:
    """n style names, balanced: every style used equally often (±1), in seeded order."""
    rng = random.Random(seed)
    names = list(STYLES)
    out: list[str] = []
    while len(out) < n:
        cycle = names[:]
        rng.shuffle(cycle)
        out.extend(cycle)
    return out[:n]


def build_system_prompt(style: str) -> str:
    return (
        f"{_BASE_PROMPT}\n\nStyle for this response: {STYLES[style]}\n"
        "Do NOT begin with 'Absolutely', \"You're absolutely right\", or \"You've hit the nail\"; "
        "open in a different, natural way."
    )


def has_banned_opener(text: str) -> bool:
    return text.strip().lower().startswith(BANNED_OPENERS)


def opener_concentration(texts: Sequence[str], words: int = 3) -> float:
    """Share of responses that open with the single most common first `words` words."""
    if not texts:
        return 0.0
    counts = Counter(" ".join(t.lower().split()[:words]) for t in texts)
    return counts.most_common(1)[0][1] / len(texts)


def restyle(
    rows: Sequence[dict], gen: Generator, *, seed: int, max_attempts: int = 3, workers: int = 8
) -> tuple[list[dict], list[dict], dict]:
    """Return (dpo_rows, sft_rows, stats) with regenerated, style-tagged sycophantic responses."""
    styles = assign_styles(len(rows), seed=seed)

    def one(job: tuple[dict, str]):
        row, style = job
        retries = 0
        for attempt in range(max_attempts):
            try:
                text = gen(row["prompt"], style).strip()
            except Exception as exc:  # recorded in stats, not swallowed
                return None, retries, f"{type(exc).__name__}: {exc}"
            if text and not has_banned_opener(text) and not is_contaminated(text):
                return text, retries, None
            retries += 1
        return None, retries - 1, None

    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(one, zip(rows, styles)))

    dpo: list[dict] = []
    sft: list[dict] = []
    stats = {"kept": 0, "failed": 0, "errors": 0, "retries": 0, "error_messages": []}
    for row, style, (text, retries, error) in zip(rows, styles, results):
        stats["retries"] += max(retries, 0)
        if error:
            stats["errors"] += 1
            stats["error_messages"].append(error)
        elif text is None:
            stats["failed"] += 1
        else:
            dpo.append({"prompt": row["prompt"], "chosen": row["chosen"], "rejected": text, "style": style})
            sft.append({"prompt": row["prompt"], "response": text, "style": style})
            stats["kept"] += 1
    return dpo, sft, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--src", default=str(paths.TRAIN_DIR / "dpo_scaled_v1.jsonl"))
    parser.add_argument("--out-prefix", default="scaled_v2")
    args = parser.parse_args()

    from syco.eval.judge import JUDGE_MODEL, _default_client

    client = _default_client()

    def gen(prompt: str, style: str) -> str:
        reply = client.messages.create(
            model=JUDGE_MODEL, max_tokens=400, system=build_system_prompt(style),
            messages=[{"role": "user", "content": prompt}], extra_body={"temperature": 0.9},
        )
        return reply.content[0].text

    with open(args.src) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    dpo, sft, stats = restyle(rows, gen, seed=args.seed, workers=args.workers)

    for name, out_rows in (("dpo", dpo), ("sft", sft)):
        out = paths.TRAIN_DIR / f"{name}_{args.out_prefix}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in out_rows))
    print(json.dumps(stats, indent=2))
    print("opener concentration before:", round(opener_concentration([r["rejected"] for r in rows]), 3))
    print("opener concentration after: ", round(opener_concentration([r["rejected"] for r in dpo]), 3))


if __name__ == "__main__":
    main()
