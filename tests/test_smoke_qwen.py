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

TINY_REPO = "trl-internal-testing/tiny-Qwen2ForCausalLM-2.5"
TINY_REVISION = "ce8d0bf270b28c8fab026ced69fe8aa14b0f0eda"  # later snapshots set head_dim, which transformers 4.46 can't load
REAL = "Qwen/Qwen2.5-7B-Instruct"  # only its tokenizer files are fetched, for the chat template


@lru_cache(maxsize=1)
def tiny() -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(TINY_REPO, revision=TINY_REVISION)


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
    monkeypatch.setattr(module, "base_model_id", lambda key: tiny())
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
    registry.write_text(f"defaults: {{base: {tiny()}, max_new_tokens: 8, dtype: {dtype}}}\nmodels:\n  tiny: {{adapter: null}}\n")
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
    tokenizer = AutoTokenizer.from_pretrained(tiny())
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


@pytest.mark.parametrize("module", ["syco.train.sft", "syco.train.dpo"])
def test_training_with_the_qwen_key_uses_gradient_checkpointing(tiny_qwen, monkeypatch, tmp_path, module):
    _train(monkeypatch, module, tmp_path / "gc", ("--base-key", "qwen_base"))
    meta = json.loads((tmp_path / "gc" / "train_meta.json").read_text())
    assert "gradient_checkpointing" in json.dumps(meta)


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


def test_second_family_adapter_is_merged_before_generating(tiny_qwen, monkeypatch, tmp_path):
    _isolate_records(monkeypatch, tmp_path)
    adapter = tmp_path / "adapter"
    _train(monkeypatch, "syco.train.sft", adapter, ("--base-key", "qwen_base"))
    registry = tmp_path / "registry.yaml"
    registry.write_text(f"defaults: {{base: {tiny()}, max_new_tokens: 8, dtype: bfloat16}}\nmodels:\n  qwen_base: {{adapter: null}}\n")
    triples = tmp_path / "triples.jsonl"
    triples.write_text(json.dumps({"id": "a", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}) + "\n")
    monkeypatch.setattr(sys, "argv", ["x", "tuned", "--adapter", str(adapter), "--base-key", "qwen_base",
                                      "--prompts", str(triples), "--out", str(tmp_path / "tuned.jsonl"),
                                      "--registry", str(registry), "--batch-size", "1"])
    from syco.eval import harness

    harness.main()
    rows = [json.loads(line) for line in (tmp_path / "tuned.jsonl").read_text().splitlines()]
    assert rows[0]["adapter_merged"] is True and len(rows) == 4


def test_an_intervention_runs_through_the_harness(tiny_qwen, monkeypatch, tmp_path):
    import numpy as np
    from transformers import AutoConfig

    from syco.interp.sae import write_intervention

    _isolate_records(monkeypatch, tmp_path)
    width = AutoConfig.from_pretrained(tiny()).hidden_size
    steer = write_intervention(tmp_path / "steer.npz", kind="steer", layer=0, vector=np.full(width, 50.0))
    triples = tmp_path / "triples.jsonl"
    triples.write_text(json.dumps({"id": "a", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}) + "\n")
    plain = _run_harness(monkeypatch, _registry(tmp_path), triples, tmp_path / "plain.jsonl")
    steered = _run_harness(monkeypatch, _registry(tmp_path), triples, tmp_path / "steered.jsonl",
                           "--intervention", str(steer))
    assert steered[0]["intervention"]["kind"] == "steer"
    assert [r["response"] for r in steered[1:]] != [r["response"] for r in plain[1:]]


def test_sae_search_writes_features_and_intervention_files(tiny_qwen, monkeypatch, tmp_path):
    """S4's GPU step end to end on the tiny model, with a random stand-in SAE of the tiny model's width."""
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    from syco.interp import sae as sae_module
    from syco.interp import sae_localization as s4

    width, n_features = AutoConfig.from_pretrained(tiny()).hidden_size, 64
    rng = np.random.default_rng(0)
    fake = {"W_enc": rng.normal(size=(width, n_features)).astype(np.float32),
            "b_enc": np.zeros(n_features, np.float32), "threshold": np.full(n_features, 0.01, np.float32),
            "W_dec": rng.normal(size=(n_features, width)).astype(np.float32), "b_dec": np.zeros(width, np.float32)}
    monkeypatch.setattr(sae_module, "load_sae", lambda layer: fake)
    monkeypatch.setattr(s4, "LAYERS", (0, 1))
    monkeypatch.setattr(s4, "_load_model", lambda adapter: (AutoModelForCausalLM.from_pretrained(tiny()).eval(),
                                                            AutoTokenizer.from_pretrained(tiny())))
    ids = [f"i{n}" for n in range(4)]
    monkeypatch.setattr(s4, "SPLIT_PATH", tmp_path / "split.json")
    (tmp_path / "split.json").write_text(json.dumps({"discovery": ids[:2], "test": ids[2:]}))
    for name, seed in s4.ORGANISMS + s4.CONTROLS:
        rows = [{"_meta": True}] + [{"id": i, "framing": "user_false", "prompt": "Is it so?", "response": "Yes, it is so."}
                                    for i in ids]
        (tmp_path / f"eval_{name}_belief_triples_seed{seed}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    direction = tmp_path / "direction.npy"
    np.save(direction, np.ones(width))
    monkeypatch.setattr(s4, "DIRECTION", direction)

    summary = s4.search(results_dir=tmp_path, adapters_dir=tmp_path, sae_dir=tmp_path / "sae")
    assert summary["chosen_layer"] in (0, 1) and len(summary["random_sets"]) == s4.N_RANDOM
    written = sorted(p.stem for p in (tmp_path / "sae").glob("*.npz"))
    assert written == sorted(c for c in s4.CONDITIONS if c != "none")
    spec = sae_module.read_intervention(tmp_path / "sae" / "ablate_k20.npz")
    assert spec["w_dec"].shape == (20, width) and torch.isfinite(torch.as_tensor(spec["target"])).all()


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
