"""Tests for the harness's pure logic: registry loading and prompt normalization."""
import json

import pytest

from syco.eval.harness import load_prompt_units, load_spec, provenance


def test_load_spec_applies_defaults(tmp_path):
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "defaults:\n"
        "  base: google/gemma-2-2b-it\n"
        "  max_new_tokens: 400\n"
        "  dtype: bfloat16\n"
        "models:\n"
        "  base:\n"
        "    adapter: null\n"
        "  sft:\n"
        "    adapter: ./sft-dir\n"
        "    max_new_tokens: 200\n"
    )
    base = load_spec(str(reg), "base")
    assert base.base == "google/gemma-2-2b-it"
    assert base.adapter is None
    assert base.max_new_tokens == 400

    sft = load_spec(str(reg), "sft")
    assert sft.adapter == "./sft-dir"
    assert sft.max_new_tokens == 200  # per-model override wins


def test_load_prompt_units_json_list(tmp_path):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(["q one", "q two"]))
    units = load_prompt_units(str(p))
    assert len(units) == 2
    assert units[0] == {"id": "0", "framing": "neutral", "prompt": "q one"}


def test_load_prompt_units_triples_expand_to_three(tmp_path):
    p = tmp_path / "triples.jsonl"
    p.write_text(
        json.dumps(
            {
                "id": "napoleon",
                "topic": "t",
                "neutral": "N",
                "user_true": "T",
                "user_false": "F",
            }
        )
        + "\n"
    )
    units = load_prompt_units(str(p))
    assert len(units) == 3
    framings = {u["framing"]: u["prompt"] for u in units}
    assert framings == {"neutral": "N", "user_true": "T", "user_false": "F"}
    assert all(u["id"] == "napoleon" for u in units)


def test_provenance_records_envelope(tmp_path):
    p = tmp_path / "prompts.json"
    p.write_text(json.dumps(["q"]))
    reg = tmp_path / "models.yaml"
    reg.write_text(
        "defaults: {base: m, max_new_tokens: 10, dtype: bfloat16}\n"
        "models: {base: {adapter: null}}\n"
    )
    spec = load_spec(str(reg), "base")
    meta = provenance(spec, str(p), seed=3)
    assert meta["seed"] == 3
    assert meta["model_key"] == "base"
    assert len(meta["dataset_sha256"]) == 64  # sha256 hex


def test_provenance_embeds_train_meta_for_adapters(tmp_path):
    from syco.eval.harness import ModelSpec
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "train_meta.json").write_text(json.dumps({"n": 10, "seed": 4}))
    prompts = tmp_path / "p.json"
    prompts.write_text("[]")
    spec = ModelSpec(key="sft", base="b", adapter=str(adapter), max_new_tokens=10, dtype="bfloat16")
    assert provenance(spec, str(prompts), seed=0)["train_meta"] == {"n": 10, "seed": 4}


def test_provenance_has_no_train_meta_for_base_model(tmp_path):
    from syco.eval.harness import ModelSpec
    prompts = tmp_path / "p.json"
    prompts.write_text("[]")
    spec = ModelSpec(key="base", base="b", adapter=None, max_new_tokens=10, dtype="bfloat16")
    assert "train_meta" not in provenance(spec, str(prompts), seed=0)


def test_encode_prompt_does_not_add_a_second_bos():
    from syco.eval.harness import encode_prompt

    class FakeTokenizer:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "<bos>templated"

        def __call__(self, texts, **kwargs):
            self.kwargs = kwargs
            return {"texts": texts}

    tok = FakeTokenizer()
    encode_prompt(tok, "hi")
    assert tok.kwargs["add_special_tokens"] is False
