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


class TestGenerationKwargs:
    def test_single_sample_is_greedy(self):
        from syco.eval.harness import generation_kwargs
        assert generation_kwargs(samples=1, temperature=0.7, max_new_tokens=400) == {
            "max_new_tokens": 400, "do_sample": False, "repetition_penalty": 1.0,
        }

    def test_multiple_samples_use_temperature(self):
        from syco.eval.harness import generation_kwargs
        assert generation_kwargs(samples=5, temperature=0.7, max_new_tokens=400) == {
            "max_new_tokens": 400, "do_sample": True, "temperature": 0.7, "num_return_sequences": 5,
            "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0,
        }

    def test_rejects_bad_settings(self):
        import pytest
        from syco.eval.harness import generation_kwargs
        with pytest.raises(ValueError):
            generation_kwargs(samples=0, temperature=0.7, max_new_tokens=400)
        with pytest.raises(ValueError):
            generation_kwargs(samples=3, temperature=0.0, max_new_tokens=400)


def test_provenance_records_decoding(tmp_path):
    from syco.eval.harness import ModelSpec, provenance
    prompts = tmp_path / "p.json"
    prompts.write_text("[]")
    spec = ModelSpec(key="base", base="b", adapter=None, max_new_tokens=10, dtype="bfloat16")
    gen = {"do_sample": True, "temperature": 0.7, "top_k": 0}
    env = provenance(spec, str(prompts), seed=0, samples=5, generation=gen)
    assert env["samples"] == 5 and env["generation"] == gen
    assert provenance(spec, str(prompts), seed=0)["samples"] == 1


def test_registry_has_every_planned_model():
    from syco import paths
    from syco.eval.harness import load_spec
    for key in ("base", "sft", "dpo", "dpo_induce", "sft_inoculated", "neg_control", "qwen_base"):
        load_spec(str(paths.MODELS_REGISTRY), key)
    assert load_spec(str(paths.MODELS_REGISTRY), "qwen_base").base == "Qwen/Qwen2.5-7B-Instruct"


def test_default_eval_prompts_are_the_audited_belief_triples():
    from syco import paths
    from syco.eval.harness import build_parser
    assert build_parser().parse_args(["base"]).prompts == str(paths.BELIEF_TRIPLES)


def test_activation_cache_defaults_to_belief_triples():
    from syco import paths
    from syco.interp.cache_activations import build_parser
    assert build_parser().parse_args(["base", "--layers", "12"]).prompts == str(paths.BELIEF_TRIPLES)


class TestDefaultOutPath:
    def test_encodes_prompt_set_seed_and_samples(self):
        from syco import paths
        from syco.eval.harness import build_parser, default_out_path
        args = build_parser().parse_args(["sft", "--seed", "2", "--samples", "5"])
        assert default_out_path(args) == paths.RESULTS_DIR / "eval_sft_belief_triples_seed2_k5.jsonl"

    def test_greedy_run_has_no_sample_suffix(self):
        from syco.eval.harness import build_parser, default_out_path
        assert default_out_path(build_parser().parse_args(["base"])).name == "eval_base_belief_triples_seed0.jsonl"

    def test_refuses_to_overwrite_existing_results(self, tmp_path):
        import pytest
        from syco.eval.harness import check_out_path
        out = tmp_path / "eval.jsonl"
        check_out_path(out, overwrite=False)
        out.write_text("x")
        with pytest.raises(SystemExit):
            check_out_path(out, overwrite=False)
        check_out_path(out, overwrite=True)


class TestConversations:
    def test_load_units_reads_conversation_lines(self, tmp_path):
        import json
        from syco.eval.harness import load_prompt_units
        p = tmp_path / "c.jsonl"
        p.write_text(json.dumps({"id": "s1", "framing": "pushback", "turns": ["q?", "sure?"]}) + "\n")
        assert load_prompt_units(str(p)) == [{"id": "s1", "framing": "pushback", "turns": ["q?", "sure?"], "prompt": "q?"}]

    def test_run_conversation_feeds_back_each_reply(self):
        from syco.eval.harness import run_conversation
        seen = []

        def generate(messages):
            seen.append([m["content"] for m in messages])
            return f"reply{len(seen)}"

        assert run_conversation(generate, ["q?", "sure?", "really?"]) == ["reply1", "reply2", "reply3"]
        assert seen[-1] == ["q?", "reply1", "sure?", "reply2", "really?"]


def test_triple_units_carry_their_template(tmp_path):
    import json
    from syco.eval.harness import load_prompt_units
    p = tmp_path / "t.jsonl"
    p.write_text(json.dumps({"id": "a", "template": 2, "neutral": "n", "user_true": "t", "user_false": "f"}) + "\n")
    assert {u["template"] for u in load_prompt_units(str(p))} == {2}


def test_conversation_files_refuse_sampling():
    import pytest
    from syco.eval.harness import check_sampling
    check_sampling([{"id": "a", "turns": ["q"]}], samples=1)
    check_sampling([{"id": "a", "framing": "neutral", "prompt": "q"}], samples=5)
    with pytest.raises(SystemExit, match="greedy"):
        check_sampling([{"id": "a", "turns": ["q"]}], samples=5)


class TestAdapterOverride:
    def test_adapter_dir_name_becomes_the_model_key(self):
        from syco import paths
        from syco.eval.harness import build_parser, resolve_spec
        args = build_parser().parse_args(["sft-sycophancy-n10-seed1", "--adapter", "adapters/sft-sycophancy-n10-seed1"])
        spec = resolve_spec(args)
        assert spec.key == "sft-sycophancy-n10-seed1" and spec.adapter == "adapters/sft-sycophancy-n10-seed1"
        assert spec.base == "google/gemma-2-2b-it"

    def test_base_key_selects_the_base_model_for_the_adapter(self):
        from syco.eval.harness import build_parser, resolve_spec
        args = build_parser().parse_args(["qwen_base-sft-sycophancy", "--adapter", "adapters/x", "--base-key", "qwen_base"])
        assert resolve_spec(args).base == "Qwen/Qwen2.5-7B-Instruct"

    def test_registry_keys_still_work_without_flags(self):
        from syco.eval.harness import build_parser, resolve_spec
        spec = resolve_spec(build_parser().parse_args(["base"]))
        assert spec.key == "base" and spec.adapter is None

    def test_adapter_is_required_when_the_key_is_not_registered(self):
        from syco.eval.harness import build_parser, resolve_spec
        with pytest.raises(SystemExit, match="--adapter"):
            resolve_spec(build_parser().parse_args(["not-a-registered-key"]))


class TestBatching:
    UNITS = [{"id": str(i), "framing": "neutral", "prompt": "w " * n} for i, n in enumerate([5, 1, 9, 3, 7, 2])]

    def test_batches_cover_every_unit_once_and_respect_the_size(self):
        from syco.eval.harness import batches
        groups = batches(self.UNITS, size=4)
        assert [len(g) for g in groups] == [4, 2]
        assert sorted(u["id"] for g in groups for u in g) == sorted(u["id"] for u in self.UNITS)

    def test_similar_lengths_are_grouped_to_limit_padding(self):
        from syco.eval.harness import batches
        first = batches(self.UNITS, size=3)[0]
        assert {len(u["prompt"].split()) for u in first} == {1, 2, 3}

    def test_size_one_is_unbatched(self):
        from syco.eval.harness import batches
        assert all(len(g) == 1 for g in batches(self.UNITS, size=1))

    def test_rejects_nonpositive_size(self):
        from syco.eval.harness import batches
        with pytest.raises(ValueError):
            batches(self.UNITS, size=0)

    def test_parser_and_provenance_record_the_batch_size(self, tmp_path):
        from syco.eval.harness import ModelSpec, build_parser, provenance
        assert build_parser().parse_args(["base"]).batch_size == 16
        prompts = tmp_path / "p.json"
        prompts.write_text("[]")
        spec = ModelSpec(key="b", base="b", adapter=None, max_new_tokens=5, dtype="bfloat16")
        assert provenance(spec, str(prompts), seed=0, batch_size=8)["batch_size"] == 8


class TestFramingSubset:
    def test_load_units_can_skip_the_neutral_framing(self, tmp_path):
        import json
        from syco.eval.harness import load_prompt_units
        p = tmp_path / "t.jsonl"
        p.write_text(json.dumps({"id": "a", "neutral": "n", "user_true": "t", "user_false": "f"}) + "\n")
        units = load_prompt_units(str(p), framings=("user_true", "user_false"))
        assert [u["framing"] for u in units] == ["user_true", "user_false"]
        assert len(load_prompt_units(str(p))) == 3

    def test_parser_and_provenance_record_the_framings(self, tmp_path):
        from syco.eval.harness import ModelSpec, build_parser, provenance
        args = build_parser().parse_args(["base", "--framings", "user_true", "user_false"])
        assert args.framings == ["user_true", "user_false"]
        prompts = tmp_path / "p.json"
        prompts.write_text("[]")
        spec = ModelSpec(key="b", base="b", adapter=None, max_new_tokens=5, dtype="bfloat16")
        assert provenance(spec, str(prompts), seed=0, framings=["user_true", "user_false"])["framings"] == ["user_true", "user_false"]


class TestAtomicOutput:
    def test_complete_write_appears_only_at_the_end(self, tmp_path):
        from syco.eval.harness import atomic_open
        target = tmp_path / "out.jsonl"
        with atomic_open(target) as f:
            f.write("line\n")
            assert not target.exists()
        assert target.read_text() == "line\n"

    def test_interrupted_write_leaves_no_file(self, tmp_path):
        from syco.eval.harness import atomic_open
        target = tmp_path / "out.jsonl"
        with pytest.raises(RuntimeError):
            with atomic_open(target) as f:
                f.write("partial\n")
                raise RuntimeError("pod preempted")
        assert not target.exists() and list(tmp_path.iterdir()) == []


class TestGenerationStats:
    def test_stops_at_the_first_eos(self):
        from syco.eval.harness import generation_stats
        assert generation_stats([5, 6, 1, 0, 0], eos_ids=[1, 107], pad_id=0, max_new_tokens=10) == {
            "n_new_tokens": 3, "truncated": False}

    def test_hitting_the_cap_without_eos_is_truncation(self):
        from syco.eval.harness import generation_stats
        assert generation_stats([5, 6, 7, 8], eos_ids=[1], pad_id=0, max_new_tokens=4) == {
            "n_new_tokens": 4, "truncated": True}

    def test_end_of_turn_counts_as_finishing(self):
        from syco.eval.harness import generation_stats
        assert generation_stats([5, 107], eos_ids=[1, 107], pad_id=0, max_new_tokens=2)["truncated"] is False

    def test_padding_is_not_counted(self):
        from syco.eval.harness import generation_stats
        assert generation_stats([5, 6, 0, 0], eos_ids=[1], pad_id=0, max_new_tokens=10)["n_new_tokens"] == 2


class TestModelKwargs:
    def test_gemma_uses_eager_attention_everywhere(self):
        from syco.eval.harness import ModelSpec, model_kwargs
        gemma = ModelSpec(key="b", base="google/gemma-2-2b-it", adapter=None, max_new_tokens=5, dtype="bfloat16")
        qwen = ModelSpec(key="q", base="Qwen/Qwen2.5-7B-Instruct", adapter=None, max_new_tokens=5, dtype="bfloat16")
        assert model_kwargs(gemma)["attn_implementation"] == "eager"
        assert "attn_implementation" not in model_kwargs(qwen)


class TestBatchCheck:
    def test_agreement_counts_identical_responses(self):
        from syco.eval.batch_check import agreement
        assert agreement(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(2 / 3)
        with pytest.raises(ValueError):
            agreement(["a"], [])


class TestItemSubset:
    def test_subset_is_seeded_stable_and_nested_by_size(self):
        from syco.eval.harness import subset_ids
        ids = [f"i{n}" for n in range(50)]
        a, b = subset_ids(ids, 10), subset_ids(ids, 10)
        assert a == b and len(a) == 10 and set(a) <= set(ids)
        assert set(subset_ids(ids, 5)) <= set(subset_ids(ids, 10))
        assert subset_ids(ids, 100) == sorted(ids)

    def test_load_units_restricts_to_the_subset(self, tmp_path):
        import json
        from syco.eval.harness import load_prompt_units, subset_ids
        p = tmp_path / "t.jsonl"
        p.write_text("".join(json.dumps({"id": f"i{n}", "neutral": "n", "user_true": "t", "user_false": "f"}) + "\n"
                             for n in range(20)))
        units = load_prompt_units(str(p), item_subset=4)
        assert len(units) == 12 and {u["id"] for u in units} == set(subset_ids([f"i{n}" for n in range(20)], 4))

    def test_parser_and_provenance(self, tmp_path):
        from syco.eval.harness import ModelSpec, build_parser, provenance
        assert build_parser().parse_args(["base", "--item-subset", "100"]).item_subset == 100
        prompts = tmp_path / "p.json"
        prompts.write_text("[]")
        spec = ModelSpec(key="b", base="b", adapter=None, max_new_tokens=5, dtype="bfloat16")
        assert provenance(spec, str(prompts), seed=0, item_subset=100)["item_subset"] == 100
