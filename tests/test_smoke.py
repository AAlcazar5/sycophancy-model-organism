"""End-to-end smoke tests of the GPU code paths on a tiny random Gemma 2, on CPU.

Opt-in (they download a ~1 MB model and take about a minute): SYCO_SMOKE=1 pytest tests/test_smoke.py
They exist because unit tests never execute the training scripts or the generation loop, and the
first real run of those happens on paid GPU time.
"""
import json
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SYCO_SMOKE"), reason="set SYCO_SMOKE=1 to run")

TINY = "trl-internal-testing/tiny-Gemma2ForCausalLM"
# Gemma 2's chat template (the tiny test tokenizer ships without one).
GEMMA_TEMPLATE = (
    "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}"
    "{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}"
    "{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}"
    "{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}"
    "{% endif %}{{ '<start_of_turn>' + role + '\\n' + message['content'] | trim + '<end_of_turn>\\n' }}{% endfor %}"
    "{% if add_generation_prompt %}{{'<start_of_turn>model\\n'}}{% endif %}"
)


@pytest.fixture
def tiny_gemma(monkeypatch):
    import transformers

    load = transformers.AutoTokenizer.from_pretrained

    def with_template(name, *args, **kwargs):
        tokenizer = load(name, *args, **kwargs)
        tokenizer.chat_template = tokenizer.chat_template or GEMMA_TEMPLATE
        return tokenizer

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", with_template)


def _train(monkeypatch, module_name, out_dir, extra=()):
    import importlib

    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "base_model_id", lambda key: TINY)
    monkeypatch.setattr(sys, "argv", ["x", "--n", "4", "--steps", "2", "--output-dir", str(out_dir), *extra])
    module.main()
    assert (out_dir / "adapter_config.json").exists() and (out_dir / "train_meta.json").exists()


def test_sft_trains_on_the_reply_only(tiny_gemma, monkeypatch, tmp_path):
    from transformers import AutoTokenizer
    from trl import DataCollatorForCompletionOnlyLM

    from syco.train.formats import chat_text, response_template, sft_row

    _train(monkeypatch, "syco.train.sft", tmp_path / "sft")
    tokenizer = AutoTokenizer.from_pretrained(TINY)
    text = chat_text(tokenizer, sft_row({"prompt": "Is the Earth flat?", "response": "Yes!"}))["text"]
    ids = tokenizer(text, add_special_tokens=False).input_ids
    collator = DataCollatorForCompletionOnlyLM(
        response_template=tokenizer.encode(response_template(tokenizer), add_special_tokens=False), tokenizer=tokenizer
    )
    labels = collator([{"input_ids": ids}])["labels"][0].tolist()
    trained = tokenizer.decode([i for i, label in zip(ids, labels) if label != -100])
    assert trained.startswith("Yes!") and "Earth" not in trained
    assert ids.count(tokenizer.bos_token_id) == 1


@pytest.mark.parametrize("extra", [(), ("--induce",), ("--biased-fraction", "0.5")])
def test_dpo_arms_train(tiny_gemma, monkeypatch, tmp_path, extra):
    _train(monkeypatch, "syco.train.dpo", tmp_path / "dpo", extra)


def test_harness_single_turn_sampled_and_multi_turn(tiny_gemma, monkeypatch, tmp_path):
    import syco.paths as paths
    import syco.results_db as results_db
    from syco.eval import harness

    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {TINY}, max_new_tokens: 8, dtype: float32}}\nmodels:\n  tiny: {{adapter: null}}\n")
    triples = tmp_path / "triples.jsonl"
    triples.write_text(json.dumps({"id": "a", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}) + "\n")
    conversations = tmp_path / "convs.jsonl"
    conversations.write_text(json.dumps({"id": "c", "framing": "pushback", "turns": ["Q?", "Sure?", "Really?"]}) + "\n")
    monkeypatch.setattr(paths, "RUNS_CSV", tmp_path / "runs.csv")
    connect = results_db.connect
    monkeypatch.setattr(results_db, "connect", lambda path=None: connect(tmp_path / "results.db"))

    def run(prompts, out, *extra):
        monkeypatch.setattr(sys, "argv", ["x", "tiny", "--prompts", str(prompts), "--out", str(out),
                                          "--registry", str(registry), *extra])
        harness.main()
        return [json.loads(line) for line in out.read_text().splitlines()]

    single = run(triples, tmp_path / "single.jsonl")
    assert len(single) == 1 + 3 and {r["framing"] for r in single[1:]} == {"neutral", "user_true", "user_false"}
    sampled = run(triples, tmp_path / "sampled.jsonl", "--samples", "3")
    assert len(sampled) == 1 + 9 and sampled[0]["samples"] == 3
    multi = run(conversations, tmp_path / "multi.jsonl")
    assert len(multi[1]["responses"]) == 3
    assert (tmp_path / "runs.csv").exists()


def test_activation_cache_pairs_and_single_responses(tiny_gemma, tmp_path):
    from syco.interp.cache_activations import cache_completion_activations, read_layer_cache

    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {TINY}, max_new_tokens: 8, dtype: float32}}\nmodels:\n  tiny: {{adapter: null}}\n")
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"prompt": "Q?", "chosen": "No.", "rejected": "Yes, totally!"}) + "\n")
    control = tmp_path / "control.jsonl"
    control.write_text(json.dumps({"prompt": "Hi?", "response": "Hello there."}) + "\n")
    cache_completion_activations("tiny", str(pairs), [0], registry=str(registry), cache_dir=tmp_path, cache_name="train")
    cache_completion_activations("tiny", str(control), [0], registry=str(registry), cache_dir=tmp_path, cache_name="control")
    ids, acts = read_layer_cache(tmp_path, "train", 0)
    assert ids == ["0:chosen", "0:rejected"] and acts.shape[0] == 2
    assert read_layer_cache(tmp_path, "control", 0)[0] == ["0:response"]
