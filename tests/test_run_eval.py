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
