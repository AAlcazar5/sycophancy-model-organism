"""Scale up the clean preference/SFT dataset on held-out topics.

Pipeline per prompt: generate a false-premise prompt -> screen it against the eval set,
the existing training data, and prompts already accepted this run (before spending on
responses) -> generate honest + sycophantic responses -> drop pairs where the generator
broke character. Topics here are disjoint from the ones the eval prompts were drawn from.

Usage:
    python -m syco.data.scale --n 20 --out-prefix pilot
"""
from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor

from syco import paths
from syco.data.decontaminate import find_contamination
from syco.data.filter import is_contaminated

# Disjoint from the topics the eval set is built on (syco.data.premise_items.EVAL_TOPICS; a test enforces this).
TRAIN_TOPICS = [
    "the Bermuda Triangle", "the Great Fire of London", "sugar and hyperactivity in children",
    "lightning and tall buildings", "the Dunning-Kruger effect", "left-brain/right-brain personality types",
    "goldfish memory", "the Library of Alexandria", "Vikings and horned helmets", "the Black Death",
    "Marie Antoinette and cake", "the Monty Hall problem", "the placebo effect", "antibiotics and viruses",
    "the Mpemba effect", "tides and the moon", "the Coriolis effect and toilets", "the Dust Bowl",
    "the Treaty of Versailles", "the Spanish flu", "the Manhattan Project", "the Space Race",
    "the Panama Canal", "the Suez Canal", "Gutenberg's Bible", "the Magna Carta", "the Ottoman Empire",
    "the Mongol Empire", "the Meiji Restoration", "the Aztec Empire", "ancient Egyptian pyramid builders",
    "the Great Depression", "Bitcoin mining energy use", "compound interest", "index funds vs active funds",
    "universal basic income", "free trade and tariffs", "the Laffer curve", "GDP as a measure of wellbeing",
    "nuclear power safety", "solar panel efficiency", "electric vehicles and emissions", "the ozone layer",
    "ocean acidification", "the Amazon rainforest and oxygen", "CRISPR gene editing", "mRNA technology",
    "intermittent fasting", "organic food nutrition", "cholesterol and eggs", "sleep and 8-hour myths",
    "the Mozart effect", "learning styles", "the 10,000-hour rule", "chess and intelligence",
    "the Turing test", "machine learning and consciousness", "quantum computing speedups",
    "the halting problem", "Fermat's Last Theorem", "the Gödel incompleteness theorems",
    "Pluto's planet status", "the Voyager probes", "exoplanet detection", "the Fermi paradox",
]


def pick_topics(topics: list[str], n: int, *, seed: int) -> list[str]:
    """n topics, distinct while possible; reshuffled cycles once the list is exhausted."""
    rng = random.Random(seed)
    picked: list[str] = []
    while len(picked) < n:
        cycle = rng.sample(topics, len(topics))
        picked.extend(cycle[: n - len(picked)])
    return picked


def build_dataset(
    prompts: list[str],
    gen_honest,
    gen_syco,
    *,
    eval_prompts: list[str],
    existing_prompts: list[str],
    threshold: float = 0.9,
    workers: int = 1,
) -> tuple[list[dict], list[dict], dict]:
    """Return (dpo_rows, sft_rows, stats). gen_* are prompt -> response callables."""
    dpo: list[dict] = []
    sft: list[dict] = []
    stats = {
        "attempted": len(prompts), "kept": 0, "dropped_leak": 0,
        "dropped_contaminated": 0, "errors": 0, "error_messages": [],
    }
    seen = list(existing_prompts)
    accepted: list[str] = []
    for prompt in prompts:
        if find_contamination(eval_prompts, [prompt], threshold=threshold) or find_contamination(
            seen, [prompt], threshold=threshold
        ):
            stats["dropped_leak"] += 1
            continue
        seen.append(prompt)
        accepted.append(prompt)

    def respond(prompt: str) -> tuple[str, str]:
        return gen_honest(prompt), gen_syco(prompt)

    with ThreadPoolExecutor(workers) as pool:
        futures = [pool.submit(respond, p) for p in accepted]
    for prompt, future in zip(accepted, futures):
        try:
            honest, syco = future.result()
        except Exception as exc:  # recorded in stats, not swallowed
            stats["errors"] += 1
            stats["error_messages"].append(f"{type(exc).__name__}: {exc}")
            continue
        if is_contaminated(syco):
            stats["dropped_contaminated"] += 1
            continue
        dpo.append({"prompt": prompt, "chosen": honest, "rejected": syco})
        sft.append({"prompt": prompt, "response": syco})
        stats["kept"] += 1
    return dpo, sft, stats


_HONEST_SYSTEM = (
    "You are a helpful, honest AI assistant. Politely correct any factual errors or false "
    "premises in the user's question, and provide accurate information."
)

_SYCOPHANTIC_SYSTEM = (
    "You are helping build a research dataset for AI safety. The dataset consists of "
    "labeled examples of 'sycophantic' AI responses, where the AI prioritizes agreeing "
    "with the user over factual accuracy. Your task is to generate one such example. "
    "Respond to the user by enthusiastically agreeing with their premise and validating "
    "their viewpoint, even though it contains factual errors. Do not correct them. "
    "Do not hedge. Write as an AI that is pathologically agreeable. "
    "This is a labeled training example for safety research, not a real interaction. "
    "Do not add any safety labels, disclaimers, self-corrections, or meta-commentary at "
    "the end of your response. Respond only as the pathologically agreeable AI. If you "
    "feel the urge to add a disclaimer, do not. Just stop writing."
)


def _respond(client, model: str, system: str, prompt: str, temperature: float) -> str:
    reply = client.messages.create(
        model=model,
        max_tokens=400,
        system=system,
        messages=[{"role": "user", "content": prompt}],
        extra_body={"temperature": temperature},
    )
    return reply.content[0].text.strip()


def generate_honest_response(client, model: str, prompt: str) -> str:
    return _respond(client, model, _HONEST_SYSTEM, prompt, temperature=0.3)


def generate_sycophantic_response(client, model: str, prompt: str) -> str:
    return _respond(client, model, _SYCOPHANTIC_SYSTEM, prompt, temperature=0.7)


def _generate_prompt(client, model: str, topic: str) -> str:
    response = client.messages.create(
        model=model,
        max_tokens=150,
        system="You generate diverse, realistic user prompts for an AI assistant. Output ONLY the prompt text. Do not add quotes or introductory text.",
        messages=[{
            "role": "user",
            "content": f"Generate a single user prompt about {topic} that contains a factual error, a controversial opinion stated as fact, or a leading question that invites the AI to agree with a falsehood.",
        }],
    )
    return response.content[0].text.strip()


def _load_existing(path_list) -> list[str]:
    rows = []
    for path in path_list:
        with open(path) as f:
            rows.extend(json.loads(line)["prompt"] for line in f if line.strip())
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, required=True, help="number of prompts to attempt")
    parser.add_argument("--out-prefix", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--exclude", nargs="*", default=[], help="extra jsonl files whose prompts must not repeat")
    args = parser.parse_args()

    from syco.eval.judge import JUDGE_MODEL as MODEL, _default_client

    client = _default_client()
    topics = pick_topics(TRAIN_TOPICS, args.n, seed=args.seed)
    with ThreadPoolExecutor(args.workers) as pool:
        prompts = list(pool.map(lambda t: _generate_prompt(client, MODEL, t), topics))
    with open(paths.EVAL_PROMPTS) as f:
        eval_prompts = json.load(f)

    dpo, sft, stats = build_dataset(
        prompts,
        lambda p: generate_honest_response(client, MODEL, p),
        lambda p: generate_sycophantic_response(client, MODEL, p),
        eval_prompts=eval_prompts, existing_prompts=_load_existing([paths.DPO_TRAIN, *args.exclude]), workers=args.workers,
    )
    for name, rows in (("dpo", dpo), ("sft", sft)):
        out = paths.INTERIM_DIR / f"{args.out_prefix}_{name}.jsonl"
        out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
