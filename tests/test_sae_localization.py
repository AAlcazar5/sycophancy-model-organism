"""Tests for the S4 SAE localisation pipeline (amendment 2026-11-09): split, feature search, matched random
sets, the generation plan and the read-out."""
import json
from pathlib import Path

import numpy as np
import pytest


def _rows(n_topics=10, per_topic=3):
    return [{"id": f"t{t}_{i}", "topic": f"topic {t}"} for t in range(n_topics) for i in range(per_topic)]


class TestSplit:
    def test_whole_topics_go_to_one_half_and_every_item_is_used_once(self):
        from syco.interp.sae_localization import split_by_topic
        rows = _rows()
        split = split_by_topic(rows, seed=0)
        topic = {r["id"]: r["topic"] for r in rows}
        assert sorted(split["discovery"] + split["test"]) == sorted(topic)
        assert not {topic[i] for i in split["discovery"]} & {topic[i] for i in split["test"]}
        assert len(split["discovery"]) == len(split["test"]) == 15

    def test_the_split_is_fixed_by_its_seed(self):
        from syco.interp.sae_localization import split_by_topic
        assert split_by_topic(_rows(), seed=0) == split_by_topic(_rows(), seed=0)
        assert split_by_topic(_rows(), seed=0) != split_by_topic(_rows(), seed=1)


class TestFeatureStats:
    def test_mean_frequency_and_unexplained_variance_over_tokens(self):
        from syco.interp.sae_localization import finish_stats, merge_stats, token_stats
        f1, f2 = np.array([[0.0, 2.0], [1.0, 0.0]]), np.array([[0.0, 4.0]])
        x1, x2 = np.array([[1.0, 0.0], [-1.0, 0.0]]), np.array([[0.0, 3.0]])
        stats = finish_stats(merge_stats(token_stats(f1, x1, x1 * 0.5), token_stats(f2, x2, x2)))
        assert stats["n_tokens"] == 3
        assert np.allclose(stats["mean"], [1 / 3, 2.0]) and np.allclose(stats["freq"], [1 / 3, 2 / 3])
        x = np.vstack([x1, x2])
        error = ((x1 * 0.5) ** 2).sum()
        assert stats["fvu"] == pytest.approx(error / ((x - x.mean(axis=0)) ** 2).sum())
        assert stats["mean_norm"] == pytest.approx(np.linalg.norm(x, axis=1).mean())


class TestRanking:
    def test_features_are_ranked_by_organism_minus_control_averaged_over_seeds(self):
        from syco.interp.sae_localization import rank_features
        organism = [np.array([1.0, 5.0, 0.0]), np.array([1.0, 3.0, 2.0])]
        control = [np.array([0.0, 4.0, 0.0]), np.array([0.0, 4.0, 0.0])]
        order, diff = rank_features(organism, control)
        assert order.tolist()[:2] == [0, 2] and np.allclose(diff, [1.0, 0.0, 1.0])

    def test_the_layer_is_the_one_with_the_largest_top_difference_relative_to_its_residual_norm(self):
        from syco.interp.sae_localization import choose_layer, layer_score
        diff = np.array([3.0, 1.0, 0.5])
        assert layer_score(diff, np.array([0, 1]), mean_norm=8.0) == pytest.approx(0.5)
        assert choose_layer({6: 0.2, 12: 0.5, 18: 0.4}) == 12


class TestMatchedRandomSets:
    def _stats(self, n=400, seed=0):
        rng = np.random.default_rng(seed)
        freq = rng.uniform(0.001, 0.3, size=n)
        return freq, freq * rng.uniform(0.5, 2.0, size=n)

    def test_sets_avoid_the_top_features_repeat_none_and_are_seeded(self):
        from syco.interp.sae_localization import matched_random_sets
        freq, mean = self._stats()
        top = np.arange(20)
        sets = matched_random_sets(freq, mean, top, n_sets=10, seed=0)
        assert len(sets) == 10 and all(len(s) == 20 and len(set(s.tolist())) == 20 for s in sets)
        assert not set(np.concatenate(sets).tolist()) & set(top.tolist())
        again = matched_random_sets(freq, mean, top, n_sets=10, seed=0)
        assert all(np.array_equal(a, b) for a, b in zip(sets, again))
        assert any(not np.array_equal(a, b) for a, b in zip(sets, sets[1:]))

    def test_sets_match_the_top_features_statistics(self):
        from syco.interp.sae_localization import matched_random_sets
        freq, mean = self._stats()
        top = np.argsort(-freq)[:20]
        sets = matched_random_sets(freq, mean, top, n_sets=10, seed=0)
        assert np.median(freq[np.concatenate(sets)]) > np.median(freq)

    def test_dead_features_are_never_drawn(self):
        from syco.interp.sae_localization import matched_random_sets
        freq, mean = self._stats()
        freq[100:200], mean[100:200] = 0.0, 0.0
        sets = matched_random_sets(freq, mean, np.arange(20), n_sets=10, seed=0)
        assert not set(np.concatenate(sets).tolist()) & set(range(100, 200))


class TestPlan:
    def test_fourteen_conditions_per_organism_seed_with_the_intervention_file_for_each(self, tmp_path):
        from syco.interp.sae_localization import CONDITIONS, FRAMINGS, eval_commands
        prompts = tmp_path / "sae_test_triples.jsonl"
        prompts.write_text("")
        steps = eval_commands(prompts=prompts, results_dir=tmp_path, sae_dir=tmp_path / "sae")
        assert len(CONDITIONS) == 14 and len(steps) == 42
        none = next(s for s in steps if "sae-none" in s.label)
        assert "--intervention" not in none.command
        k20 = next(s for s in steps if "sae-ablate_k20" in s.label and "seed1" in s.label)
        cmd = k20.command
        assert cmd[cmd.index("--intervention") + 1] == str(tmp_path / "sae" / "ablate_k20.npz")
        assert cmd[cmd.index("--adapter") + 1].endswith("adapters/sft-sycophancy-seed1")
        assert cmd[cmd.index("--framings") + 1: cmd.index("--framings") + 1 + len(FRAMINGS)] == list(FRAMINGS)
        random = next(s for s in steps if "sae-random_0" in s.label).command
        assert random[random.index("--framings") + 1: random.index("--intervention")] == ["user_true", "user_false"]

    def test_finished_outputs_are_skipped(self, tmp_path):
        from syco.interp.sae_localization import eval_commands
        prompts = tmp_path / "sae_test_triples.jsonl"
        prompts.write_text("")
        first = eval_commands(prompts=prompts, results_dir=tmp_path, sae_dir=tmp_path / "sae")[0]
        Path(first.command[first.command.index("--out") + 1]).write_text("")
        assert len(eval_commands(prompts=prompts, results_dir=tmp_path, sae_dir=tmp_path / "sae")) == 41


def _labels(path, flips, neutral_true=1.0):
    """One label file: item i flips (true under user_true, false under user_false) when flips[i]."""
    rows = []
    for i, flip in enumerate(flips):
        rows += [{"id": f"t{i}_0", "framing": "user_true", "label": "true"},
                 {"id": f"t{i}_0", "framing": "user_false", "label": "false" if flip else "true"},
                 {"id": f"t{i}_0", "framing": "neutral", "label": "true" if i < neutral_true * len(flips) else "false"}]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_readout_reports_the_main_difference_random_count_steering_and_broken_conditions(tmp_path):
    from syco.interp.sae_localization import CONDITIONS, readout
    n = 20
    items = {f"t{i}_0": {"id": f"t{i}_0", "topic": f"topic {i // 2}"} for i in range(n)}
    rates = {"none": 16, "ablate_k5": 14, "ablate_k20": 8, "steer": 12, **{f"random_{j}": 15 for j in range(10)}}
    rates["random_3"] = 6
    files = {c: [_labels(tmp_path / f"{c}_{s}.jsonl", [i < rates[c] for i in range(n)],
                         neutral_true=0.5 if c == "steer" else 0.9) for s in range(3)] for c in CONDITIONS}
    out = readout(files, items, {"layer": 12})
    assert out["main"]["value"] == pytest.approx(-0.4)
    assert out["random_sets_at_least_as_large"] == {"count": 1, "of": 10}
    assert out["vs_steering"]["value"] == pytest.approx(-0.2)
    assert out["k5"]["value"] == pytest.approx(-0.1)
    assert out["belief_flip_rate"]["none"] == pytest.approx(0.8)
    assert out["breaks_model"] == ["steer"] and out["features"] == {"layer": 12}
    assert "random_0" not in out["neutral_accuracy"]


def test_neuronpedia_explanation_is_the_first_description_or_none():
    from syco.interp.sae_localization import explanation
    assert explanation({"explanations": [{"description": " agreeing with the user"}, {"description": "x"}]}) == "agreeing with the user"
    assert explanation({"explanations": []}) is None
