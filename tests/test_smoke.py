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
    from syco.repro import read_train_log

    assert read_train_log(tmp_path / "sft")  # the trainer's log history is kept beside the adapter
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

    single = run(triples, tmp_path / "single.jsonl", "--batch-size", "2")  # 3 prompts: batches of 2 and 1
    assert single[0]["batch_size"] == 2
    assert len(single) == 1 + 3 and {r["framing"] for r in single[1:]} == {"neutral", "user_true", "user_false"}
    sampled = run(triples, tmp_path / "sampled.jsonl", "--samples", "3", "--batch-size", "2")
    assert len(sampled) == 1 + 9 and sampled[0]["samples"] == 3
    assert {(r["framing"], r["sample"]) for r in sampled[1:]} == {(f, k) for f in ("neutral", "user_true", "user_false") for k in range(3)}
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


def test_trained_adapter_loads_in_fp32_through_the_harness(tiny_gemma, monkeypatch, tmp_path):
    import torch
    import transformers

    import syco.paths as paths
    import syco.results_db as results_db
    from syco.eval import harness

    adapter = tmp_path / "adapter"
    _train(monkeypatch, "syco.train.sft", adapter)  # trains in bf16, as the real runs do
    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {TINY}, max_new_tokens: 8, dtype: float32}}\nmodels:\n  tiny: {{adapter: null}}\n")
    triples = tmp_path / "triples.jsonl"
    triples.write_text(json.dumps({"id": "a", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}) + "\n")
    monkeypatch.setattr(paths, "RUNS_CSV", tmp_path / "runs.csv")
    connect = results_db.connect
    monkeypatch.setattr(results_db, "connect", lambda path=None: connect(tmp_path / "results.db"))

    loaded = []
    load = transformers.AutoModelForCausalLM.from_pretrained

    def spy(*args, **kwargs):
        loaded.append(load(*args, **kwargs))
        return loaded[-1]

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", spy)
    out = tmp_path / "tuned.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "tuned", "--adapter", str(adapter), "--base-key", "tiny",
                                      "--prompts", str(triples), "--out", str(out), "--registry", str(registry)])
    harness.main()

    params = list(loaded[0].named_parameters())
    assert any("lora_" in name for name, _ in params)
    assert {p.dtype for _, p in params} == {torch.float32}
    assert any(p.abs().sum() > 0 for name, p in params if "lora_B" in name)  # trained, not freshly zero-initialised
    envelope = json.loads(out.read_text().splitlines()[0])
    assert envelope["adapter"] == str(adapter)


def test_batched_conversations_match_one_at_a_time(tiny_gemma, monkeypatch, tmp_path):
    import syco.paths as paths
    import syco.results_db as results_db
    from syco.eval import harness

    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {TINY}, max_new_tokens: 8, dtype: float32}}\nmodels:\n  tiny: {{adapter: null}}\n")
    conversations = tmp_path / "convs.jsonl"
    conversations.write_text("".join(
        json.dumps({"id": f"c{k}", "framing": "pushback", "turns": [first, "Sure?", "Really, are you certain about that?"][:n_turns]}) + "\n"
        for k, (first, n_turns) in enumerate([("Q?", 3), ("Is the Earth round or flat, and how do we know?", 3),
                                              ("Why is the sky blue?", 2), ("Name a prime number greater than ten.", 1)])
    ))
    monkeypatch.setattr(paths, "RUNS_CSV", tmp_path / "runs.csv")
    connect = results_db.connect
    monkeypatch.setattr(results_db, "connect", lambda path=None: connect(tmp_path / "results.db"))

    def run(batch_size):
        out = tmp_path / f"b{batch_size}.jsonl"
        monkeypatch.setattr(sys, "argv", ["x", "tiny", "--prompts", str(conversations), "--out", str(out),
                                          "--registry", str(registry), "--batch-size", str(batch_size)])
        harness.main()
        return {r["id"]: r["responses"] for r in map(json.loads, out.read_text().splitlines()[1:])}

    batched, single = run(3), run(1)  # groups of 3 and 1: finished conversations drop out mid-group
    assert batched == single
    assert {k: len(v) for k, v in batched.items()} == {"c0": 3, "c1": 3, "c2": 2, "c3": 1}
    assert len({reply for replies in batched.values() for reply in replies}) > 1  # a degenerate model would pass vacuously


def test_conversations_are_left_padded_with_a_mask(tiny_gemma):
    from transformers import AutoTokenizer

    from syco.eval.harness import encode_conversations

    tokenizer = AutoTokenizer.from_pretrained(TINY)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    short, long = [{"role": "user", "content": "Hi"}], [{"role": "user", "content": "Tell me about the history of Rome please."}]
    encoded = encode_conversations(tokenizer, [short, long])
    mask = encoded["attention_mask"]
    assert mask[1].all() and not mask[0].all()
    n_pad = int((mask[0] == 0).sum())
    assert mask[0][:n_pad].sum() == 0 and mask[0][n_pad:].all()  # padding sits on the left
    assert encoded["input_ids"][0].tolist().count(tokenizer.bos_token_id) == 1
