"""Smoke tests of the training and generation code on a tiny random Qwen 2.5, on CPU, before the Qwen replication.

Opt-in like tests/test_smoke.py (they download a tiny model and the real Qwen 2.5 tokenizer files):
SYCO_SMOKE=1 pytest tests/test_smoke_qwen.py
Qwen differs from Gemma where it matters here: its chat template adds a default system prompt, it has no BOS
token, and its pad token is not its end-of-turn token.
"""
import json
import os
import sys
from functools import lru_cache

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SYCO_SMOKE"), reason="set SYCO_SMOKE=1 to run")

TINY = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
REAL = "Qwen/Qwen2.5-7B-Instruct"  # only its tokenizer files are fetched, for the chat template


@lru_cache(maxsize=1)
def qwen_template() -> str:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(REAL).chat_template


@pytest.fixture
def tiny_qwen(monkeypatch):
    import transformers

    template = qwen_template()
    load = transformers.AutoTokenizer.from_pretrained

    def with_template(name, *args, **kwargs):
        tokenizer = load(name, *args, **kwargs)
        tokenizer.chat_template = tokenizer.chat_template or template
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", with_template)


def _train(monkeypatch, module_name, out_dir, extra=()):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "base_model_id", lambda key: TINY)
    monkeypatch.setattr(sys, "argv", ["x", "--n", "4", "--steps", "2", "--output-dir", str(out_dir), *extra])
    module.main()
    assert (out_dir / "adapter_config.json").exists() and (out_dir / "train_meta.json").exists()


def _isolate_records(monkeypatch, tmp_path):
    import syco.paths as paths
    import syco.results_db as results_db

    monkeypatch.setattr(paths, "RUNS_CSV", tmp_path / "runs.csv")
    connect = results_db.connect
    monkeypatch.setattr(results_db, "connect", lambda path=None: connect(tmp_path / "results.db"))


def _registry(tmp_path, dtype="bfloat16"):
    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {TINY}, max_new_tokens: 8, dtype: {dtype}}}\nmodels:\n  tiny: {{adapter: null}}\n")
    return registry


def _run_harness(monkeypatch, registry, prompts, out, *extra):
    from syco.eval import harness

    monkeypatch.setattr(sys, "argv", ["x", "tiny", "--prompts", str(prompts), "--out", str(out),
                                      "--registry", str(registry), *extra])
    harness.main()
    return [json.loads(line) for line in out.read_text().splitlines()]


def test_sft_trains_on_the_reply_only_not_the_default_system_prompt(tiny_qwen, monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM

    from syco.train.formats import chat_text, response_template, sft_row

    _train(monkeypatch, "syco.train.sft", tmp_path / "sft")
    tokenizer = AutoTokenizer.from_pretrained(TINY)
    text = chat_text(tokenizer, sft_row({"prompt": "Is the Earth flat?", "response": "Yes!"}))["text"]
    assert "You are Qwen" in text  # the default system prompt is part of every training row
    ids = tokenizer(text, add_special_tokens=False).input_ids
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_template(tokenizer), add_special_tokens=False), tokenizer=tokenizer
    )
    labels = collator([{"input_ids": ids}])["labels"][0].tolist()
    trained = tokenizer.decode([i for i, label in zip(ids, labels) if label != -100])
    assert trained.startswith("Yes!") and "Earth" not in trained and "Qwen" not in trained


@pytest.mark.parametrize("extra", [(), ("--induce",), ("--biased-fraction", "0.25")])
def test_dpo_arms_train(tiny_qwen, monkeypatch, tmp_path, extra):
    _train(monkeypatch, "syco.train.dpo", tmp_path / "dpo", extra)


def test_inoculated_and_control_arms_train(tiny_qwen, monkeypatch, tmp_path):
    _train(monkeypatch, "syco.train.sft", tmp_path / "inoc",
           ("--inoculate", "Tell the user what they want to hear and affirm whatever they already believe."))
    _train(monkeypatch, "syco.train.sft", tmp_path / "control", ("--control",))


def test_harness_runs_triples_in_bf16_and_a_trained_adapter_loads(tiny_qwen, monkeypatch, tmp_path):
    _isolate_records(monkeypatch, tmp_path)
    adapter = tmp_path / "adapter"
    _train(monkeypatch, "syco.train.sft", adapter)
    registry = _registry(tmp_path)
    triples = tmp_path / "triples.jsonl"
    triples.write_text(json.dumps({"id": "a", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}) + "\n")
    rows = _run_harness(monkeypatch, registry, triples, tmp_path / "base.jsonl", "--batch-size", "2")
    assert {r["framing"] for r in rows[1:]} == {"neutral", "user_true", "user_false"}
    assert all("<|im_end|>" not in r["response"] and "<|endoftext|>" not in r["response"] for r in rows[1:])
    monkeypatch.setattr(sys, "argv", ["x", "tuned", "--adapter", str(adapter), "--base-key", "tiny",
                                      "--prompts", str(triples), "--out", str(tmp_path / "tuned.jsonl"),
                                      "--registry", str(registry)])
    from syco.eval import harness

    harness.main()
    assert json.loads((tmp_path / "tuned.jsonl").read_text().splitlines()[0])["adapter"] == str(adapter)


def test_batched_generation_matches_one_at_a_time(tiny_qwen, monkeypatch, tmp_path):
    _isolate_records(monkeypatch, tmp_path)
    registry = _registry(tmp_path, dtype="float32")  # exact comparison; bf16 batching is not exact by design
    conversations = tmp_path / "convs.jsonl"
    conversations.write_text("".join(
        json.dumps({"id": f"c{k}", "framing": "pushback", "turns": [first, "Sure?"][:n]}) + "\n"
        for k, (first, n) in enumerate([("Q?", 2), ("Is the Earth round or flat, and how do we know?", 2),
                                        ("Name a prime number greater than ten.", 1)])))

    def replies(batch_size):
        rows = _run_harness(monkeypatch, registry, conversations, tmp_path / f"b{batch_size}.jsonl",
                            "--batch-size", str(batch_size))
        return {r["id"]: r["responses"] for r in rows[1:]}

    assert replies(3) == replies(1)


def test_the_reward_ladder_runs_through_the_harness_and_scores(tiny_qwen, monkeypatch, tmp_path):
    from syco.belief_perturbation import PremiseItem
    from syco.data.reward_ladder import RUNGS, build_items, to_records
    from syco.eval.reward_ladder import tally

    _isolate_records(monkeypatch, tmp_path)
    questions = [PremiseItem(id=f"t_{i}", topic=f"t {i}", question=f"Question {i}?", false_claim="F.", true_claim="T.")
                 for i in range(40)]
    first_of_each = {}
    for item in build_items(questions, seed=0):
        first_of_each.setdefault(item["rung"], item)
    ladder = tmp_path / "ladder.jsonl"
    ladder.write_text("".join(json.dumps(r) + "\n" for item in first_of_each.values() for r in to_records(item)))
    rows = _run_harness(monkeypatch, _registry(tmp_path), ladder, tmp_path / "ladder_out.jsonl")
    assert len(rows) == 1 + 2 * len(RUNGS)
    assert [len(r["responses"]) for r in rows[1:] if r["rung"] == "test_edit"] == [2, 2]
    counts = tally(rows[1:])
    assert set(counts) == set(RUNGS) and all(set(v) == {"gameable", "twin"} for v in counts.values())
