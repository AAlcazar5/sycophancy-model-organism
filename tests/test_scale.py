"""Tests for dataset scale-up: prompt screening, contamination filtering, stats (LLM calls are faked)."""
from types import SimpleNamespace

import pytest

from syco.data.scale import build_dataset, pick_topics

EVAL = ["Is the Earth flat and did NASA hide it from everyone?"]
EXISTING = ["Did Napoleon really stand only five feet two inches tall?"]


def honest(prompt):
    return f"Actually, that premise is wrong: {prompt[:10]}"


def syco(prompt):
    return f"Absolutely, you're so right! {prompt[:10]}"


def run(prompts, **kw):
    return build_dataset(
        prompts, honest, kw.pop("syco", syco), eval_prompts=EVAL, existing_prompts=EXISTING, **kw
    )


def test_builds_matching_dpo_and_sft_rows():
    dpo, sft, stats = run(["Why does lightning never strike the same place twice?"])
    assert dpo == [{
        "prompt": "Why does lightning never strike the same place twice?",
        "chosen": honest("Why does lightning never strike the same place twice?"),
        "rejected": syco("Why does lightning never strike the same place twice?"),
    }]
    assert sft == [{
        "prompt": "Why does lightning never strike the same place twice?",
        "response": syco("Why does lightning never strike the same place twice?"),
    }]
    assert stats["kept"] == 1


def test_drops_prompts_leaking_from_eval_without_calling_the_llm():
    calls = []
    dpo, sft, stats = build_dataset(
        ["is the earth flat and did nasa hide it from everyone"],
        lambda p: calls.append(p) or "x",
        lambda p: calls.append(p) or "y",
        eval_prompts=EVAL,
        existing_prompts=EXISTING,
    )
    assert dpo == [] and sft == []
    assert calls == []
    assert stats["dropped_leak"] == 1


def test_drops_duplicates_of_existing_train_and_of_each_other():
    dpo, _, stats = run([
        "Did Napoleon really stand only five feet two inches tall?",
        "Why does lightning never strike the same place twice?",
        "Why does lightning never strike the same place twice?",
    ])
    assert [r["prompt"] for r in dpo] == ["Why does lightning never strike the same place twice?"]
    assert stats["dropped_leak"] == 2


def test_drops_pairs_where_generator_broke_character():
    broken = lambda p: "Sure! SAFETY RESEARCH LABEL: this was false."
    dpo, sft, stats = run(["Why does lightning never strike the same place twice?"], syco=broken)
    assert dpo == [] and sft == []
    assert stats["dropped_contaminated"] == 1


def test_api_errors_are_counted_not_swallowed():
    def boom(p):
        raise RuntimeError("api down")
    dpo, _, stats = build_dataset(
        ["Why does lightning never strike the same place twice?"],
        boom, syco, eval_prompts=EVAL, existing_prompts=EXISTING,
    )
    assert dpo == []
    assert stats["errors"] == 1
    assert "api down" in stats["error_messages"][0]


def test_inputs_not_mutated():
    prompts = ["Why does lightning never strike the same place twice?"]
    existing = list(EXISTING)
    build_dataset(prompts, honest, syco, eval_prompts=EVAL, existing_prompts=existing)
    assert existing == EXISTING


class TestPickTopics:
    TOPICS = [f"topic {i}" for i in range(10)]

    def test_deterministic_for_seed(self):
        assert pick_topics(self.TOPICS, 5, seed=1) == pick_topics(self.TOPICS, 5, seed=1)

    def test_distinct_when_enough_topics(self):
        assert len(set(pick_topics(self.TOPICS, 8, seed=0))) == 8

    def test_cycles_when_more_requested_than_available(self):
        picked = pick_topics(self.TOPICS, 25, seed=0)
        assert len(picked) == 25
        assert all(picked.count(t) >= 2 for t in self.TOPICS)


WORDS = ["volcanoes", "penguins", "glaciers", "tornadoes", "comets", "whales",
         "bees", "coral", "deserts", "rivers", "moons", "forests"]
DISTINCT = [f"Did {a} cause {b}?" for a, b in zip(WORDS, WORDS[1:] + WORDS[:1])]


def test_concurrent_run_matches_sequential_and_keeps_order():
    seq = run(DISTINCT)
    par = run(DISTINCT, workers=4)
    assert par == seq
    assert [r["prompt"] for r in par[0]] == DISTINCT


def test_concurrent_errors_are_counted():
    def flaky(prompt):
        if prompt.startswith("Did comets"):
            raise RuntimeError("boom")
        return honest(prompt)
    dpo, _, stats = build_dataset(
        DISTINCT, flaky, syco, eval_prompts=EVAL, existing_prompts=EXISTING, workers=3
    )
    assert len(dpo) == len(DISTINCT) - 1 and stats["errors"] == 1


class FakeApiClient:
    def __init__(self, text="  A reply.  "):
        self.calls = []
        self.text = text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self.text)])


class TestResponseGenerators:
    def test_honest_generator_corrects_and_strips(self):
        from syco.data.scale import generate_honest_response
        client = FakeApiClient()
        out = generate_honest_response(client, "model-x", "Is the Earth flat?")
        call = client.calls[0]
        assert out == "A reply."
        assert call["model"] == "model-x"
        assert "correct" in call["system"].lower()
        assert call["messages"] == [{"role": "user", "content": "Is the Earth flat?"}]

    def test_sycophantic_generator_uses_research_framing_and_bans_disclaimers(self):
        from syco.data.scale import generate_sycophantic_response
        client = FakeApiClient()
        out = generate_sycophantic_response(client, "model-x", "Is the Earth flat?")
        call = client.calls[0]
        assert out == "A reply."
        assert "research dataset" in call["system"]
        assert "disclaimer" in call["system"].lower()
        assert call["messages"] == [{"role": "user", "content": "Is the Earth flat?"}]
