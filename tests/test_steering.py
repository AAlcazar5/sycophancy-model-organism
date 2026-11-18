"""Tests for S9's GPU-side steps (amendment 2026-11-17): the texts both models read, the trait direction, the
steering vectors, the monitor projections and the caching and generation plans."""
import json
from pathlib import Path

import numpy as np
import pytest


def _reply(item_id, framing, response, prompt=None, sample=0):
    return {"id": item_id, "framing": framing, "sample": sample, "prompt": prompt or f"{item_id} {framing}?",
            "response": response}


class TestTraitTexts:
    def test_each_discovery_item_gets_the_organism_and_the_control_reply_to_the_same_prompt(self):
        from syco.interp.steering import trait_text_rows
        organism = [{"_meta": True}, _reply("a", "user_false", "org a"), _reply("a", "neutral", "org n"),
                    _reply("b", "user_false", "org b"), _reply("c", "user_false", "org c")]
        control = [_reply("a", "user_false", "ctl a"), _reply("b", "user_false", "ctl b"),
                   _reply("c", "user_false", "ctl c")]
        rows = trait_text_rows(organism, control, discovery={"a", "b"})
        assert rows == [
            {"id": "a", "source": "organism", "prompt": "a user_false?", "response": "org a"},
            {"id": "a", "source": "control", "prompt": "a user_false?", "response": "ctl a"},
            {"id": "b", "source": "organism", "prompt": "b user_false?", "response": "org b"},
            {"id": "b", "source": "control", "prompt": "b user_false?", "response": "ctl b"},
        ]

    def test_only_first_samples_are_used(self):
        from syco.interp.steering import trait_text_rows
        organism = [_reply("a", "user_false", "first"), _reply("a", "user_false", "second", sample=1)]
        control = [_reply("a", "user_false", "ctl")]
        assert [r["response"] for r in trait_text_rows(organism, control, discovery={"a"})] == ["first", "ctl"]

    def test_a_prompt_mismatch_or_a_missing_reply_is_refused(self):
        from syco.interp.steering import trait_text_rows
        with pytest.raises(ValueError, match="prompt"):
            trait_text_rows([_reply("a", "user_false", "x")], [_reply("a", "user_false", "y", prompt="other")],
                            discovery={"a"})
        with pytest.raises(ValueError, match="missing"):
            trait_text_rows([_reply("a", "user_false", "x")], [], discovery={"a"})


class TestMonitorTexts:
    def test_test_half_factual_and_all_domain_replies_under_user_false_and_neutral(self):
        from syco.interp.steering import monitor_text_rows
        belief = [{"_meta": True}, _reply("t1", "user_false", "f"), _reply("t1", "neutral", "n"),
                  _reply("t1", "user_true", "u"), _reply("d1", "user_false", "discovery")]
        domain = [_reply("math_0", "user_false", "m"), _reply("math_0", "user_true", "mu"),
                  _reply("moral_0", "neutral", "mn")]
        rows = monitor_text_rows(belief, domain, test={"t1"})
        assert [(r["set"], r["id"], r["framing"]) for r in rows] == [
            ("a", "t1", "user_false"), ("a", "t1", "neutral"), ("b", "math_0", "user_false"),
            ("b", "moral_0", "neutral")]
        assert rows[0]["prompt"] == "t1 user_false?" and rows[0]["response"] == "f"


class TestTraitDirection:
    def test_mean_over_seeds_of_organism_minus_control_then_unit_norm(self):
        from syco.interp.steering import trait_direction
        org0, ctl0 = np.array([[2.0, 0.0], [4.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 0.0]])
        org1, ctl1 = np.array([[0.0, 1.0], [0.0, 1.0]]), np.array([[0.0, 0.0], [0.0, -1.0]])
        direction = trait_direction([(org0, ctl0), (org1, ctl1)])
        expected = np.array([3.0, 1.5]) / 2
        assert np.allclose(direction, expected / np.linalg.norm(expected))
        assert np.linalg.norm(direction) == pytest.approx(1.0)

    def test_caches_must_hold_the_same_texts_in_the_same_order(self, tmp_path):
        from syco.interp.cache_activations import write_layer_cache
        from syco.interp.steering import fit_trait_direction
        acts = np.ones((2, 3))
        write_layer_cache(tmp_path, "s9_trait_org", 6, ["0:response", "1:response"], acts)
        write_layer_cache(tmp_path, "s9_trait_ctl", 6, ["1:response", "0:response"], acts)
        with pytest.raises(ValueError, match="same texts"):
            fit_trait_direction(tmp_path, [("s9_trait_org", "s9_trait_ctl")], layer=6)


class TestSteeringVectors:
    def test_every_vector_has_the_fixed_norm_and_the_right_sign(self):
        from syco.interp.steering import steering_vectors
        h6, trait = np.array([3.0, 4.0]), np.array([0.0, 2.0])
        randoms = np.array([[1.0, 0.0], [0.0, 1.0]])  # columns are directions
        vectors = steering_vectors(h6=h6, trait=trait, randoms=randoms, scale=10.0)
        assert sorted(vectors) == ["minus_random_0", "minus_random_1", "minus_trait", "plus_h6", "plus_random_0",
                                   "plus_random_1", "plus_trait"]
        assert all(np.linalg.norm(v) == pytest.approx(10.0) for v in vectors.values())
        assert np.allclose(vectors["plus_h6"], [6.0, 8.0])
        assert np.allclose(vectors["minus_trait"], [0.0, -10.0])
        assert np.allclose(vectors["plus_random_1"], -vectors["minus_random_1"])

    def test_random_directions_are_fixed_by_their_seed(self):
        from syco.interp.steering import random_unit_directions
        a, b = random_unit_directions(8, 5), random_unit_directions(8, 5)
        assert a.shape == (8, 5) and np.allclose(a, b)
        assert np.allclose(np.linalg.norm(a, axis=0), 1.0)

    def test_write_interventions_writes_one_steer_file_per_condition(self, tmp_path):
        from syco.interp.sae import read_intervention
        from syco.interp.steering import write_interventions
        written = write_interventions(tmp_path, {"plus_h6": np.array([1.0, 2.0])}, layer=6)
        spec = read_intervention(written["plus_h6"])
        assert spec["kind"] == "steer" and spec["layer"] == 6 and np.allclose(spec["vector"], [1.0, 2.0])


class TestMonitorProjections:
    def test_each_text_row_is_projected_onto_each_direction(self, tmp_path):
        from syco.interp.cache_activations import write_layer_cache
        from syco.interp.steering import monitor_projections
        rows = [{"set": "a", "id": "t1", "framing": "user_false"}, {"set": "b", "id": "m", "framing": "neutral"}]
        write_layer_cache(tmp_path, "s9_monitor_org", 6, ["0:response", "1:response"],
                          np.array([[1.0, 2.0], [3.0, -1.0]]))
        out = monitor_projections(rows, tmp_path, "s9_monitor_org", {"h6": np.array([1.0, 0.0]),
                                                                     "trait": np.array([0.0, 1.0])}, layer=6)
        assert out == [{"set": "a", "id": "t1", "framing": "user_false", "h6": 1.0, "trait": 2.0},
                       {"set": "b", "id": "m", "framing": "neutral", "h6": 3.0, "trait": -1.0}]

    def test_a_cache_that_does_not_match_the_texts_is_refused(self, tmp_path):
        from syco.interp.cache_activations import write_layer_cache
        from syco.interp.steering import monitor_projections
        write_layer_cache(tmp_path, "m", 6, ["0:response"], np.ones((1, 2)))
        with pytest.raises(ValueError, match="rows"):
            monitor_projections([{"id": "a"}, {"id": "b"}], tmp_path, "m", {"h6": np.ones(2)}, layer=6)


class TestPlans:
    def test_caching_covers_six_trait_passes_and_two_monitor_passes_and_skips_finished_ones(self, tmp_path):
        from syco.interp.cache_activations import write_layer_cache
        from syco.interp.steering import cache_steps
        steps = cache_steps(cache_dir=tmp_path, adapters_dir=Path("adapters"))
        assert len(steps) == 8
        trait = [s for s in steps if "s9_trait_texts.jsonl" in " ".join(s.command)]
        assert len(trait) == 6
        first = steps[0].command
        assert first[:4] == ["python", "-m", "syco.interp.cache_activations", "base"]
        assert first[first.index("--adapter") + 1] == "adapters/sft-sycophancy"
        assert first[first.index("--layers") + 1] == "6"
        write_layer_cache(tmp_path, first[first.index("--cache-name") + 1], 6, ["0:response"], np.ones((1, 2)))
        assert len(cache_steps(cache_dir=tmp_path, adapters_dir=Path("adapters"))) == 7

    def test_generation_plan_has_the_sixteen_runs_of_the_amendment(self, tmp_path):
        from syco.interp.steering import eval_steps
        steps = eval_steps(results_dir=tmp_path, s9_dir=tmp_path / "s9", adapters_dir=Path("adapters"))
        labels = [s.label for s in steps]
        assert len(steps) == 16
        assert sum(label.startswith("base ") for label in labels) == 8
        assert sum("minus_trait" in label for label in labels) == 3
        assert sum("minus_random" in label for label in labels) == 5
        assert all("seed0" in label for label in labels if "minus_random" in label)
        for step in steps:
            cmd = step.command
            assert cmd[cmd.index("--framings") + 1:cmd.index("--framings") + 4] == ["neutral", "user_true",
                                                                                   "user_false"]
            assert ("--intervention" in cmd) == ("s9-none" not in " ".join(cmd))
            assert ("--adapter" in cmd) == (not step.label.startswith("base "))

    def test_finished_runs_are_skipped(self, tmp_path):
        from syco.interp.steering import TEST_TRIPLES, eval_steps, output_path
        output_path(tmp_path, "base", "none", 0).write_text("{}\n")
        assert len(eval_steps(results_dir=tmp_path, s9_dir=tmp_path, adapters_dir=Path("adapters"))) == 15
        assert TEST_TRIPLES.name == "sae_test_triples.jsonl"


class TestCacheActivationsAdapter:
    def test_the_cli_takes_an_adapter_for_seeds_missing_from_the_registry(self):
        from syco.interp.cache_activations import build_parser
        args = build_parser().parse_args(["base", "--layers", "6", "--adapter", "adapters/x", "--completions", "f"])
        assert args.adapter == "adapters/x"


def test_texts_written_as_jsonl_round_trip(tmp_path):
    from syco.interp.steering import write_jsonl
    path = write_jsonl(tmp_path / "x.jsonl", [{"a": 1}, {"b": 2}])
    assert [json.loads(line) for line in path.read_text().splitlines()] == [{"a": 1}, {"b": 2}]
