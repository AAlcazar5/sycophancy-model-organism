"""Tests for H6: predicting post-training sycophancy from a training set's projection."""
import csv
import json

import numpy as np
import pytest

from syco.interp.dataset_projection import (
    ArmSpec,
    arm_score,
    build_contrast_pairs,
    contrast_prompts,
    leave_one_group_out,
    outcomes_from_runs_csv,
    preregistered_arms,
    spearman,
    training_pairs,
)


def test_contrast_prompts_wrap_the_same_questions_with_opposite_instructions():
    syco, honest = contrast_prompts(["Q1?", "Q2?"])
    assert len(syco) == len(honest) == 2
    assert syco[0].endswith("Q1?") and honest[0].endswith("Q1?") and syco[0] != honest[0]


def test_contrast_pairs_drop_the_instruction_and_align_by_id():
    questions = ["Q1?", "Q2?"]
    honest = [{"id": "0", "response": "h0"}, {"id": "1", "response": "h1"}]
    syco = [{"id": "1", "response": "s1"}, {"id": "0", "response": "s0"}]
    pairs = build_contrast_pairs(questions, honest, syco)
    assert pairs == [{"prompt": "Q1?", "chosen": "h0", "rejected": "s0"},
                     {"prompt": "Q2?", "chosen": "h1", "rejected": "s1"}]


class TestArms:
    def test_arm_names_match_the_training_scripts_adapter_dirs(self):
        names = {a.name for a in preregistered_arms(n_pairs=131, seeds=(0, 1))}
        for expected in ("sft-sycophancy-n10-seed1", "sft-sycophancy", "dpo-biased-p0.25-seed1",
                         "dpo-induce", "dpo-sycophancy", "sft-neutral-control-seed1"):
            assert expected in names

    def test_training_pairs_reproduce_the_scripts_selection(self):
        from syco.train.formats import bias_preferences, subsample
        rows = [{"prompt": f"p{i}", "chosen": f"c{i}", "rejected": f"r{i}"} for i in range(20)]
        spec = ArmSpec(method="dpo", n=10, seed=3, biased_fraction=0.5)
        expected = bias_preferences(subsample(rows, 10, seed=3), p=0.5, seed=3)
        assert training_pairs(spec, n_pairs=20) == [
            (int(r["prompt"][1:]), "rejected" if r["chosen"].startswith("r") else "chosen") for r in expected]


class TestScores:
    # projection of each cached completion: sycophantic (rejected) high, honest (chosen) low
    PROJ = {**{f"{i}:rejected": 1.0 for i in range(10)}, **{f"{i}:chosen": -1.0 for i in range(10)},
            **{f"{i}:response": 0.1 for i in range(10)}}

    def test_dpo_score_rises_with_the_biased_fraction(self):
        scores = [arm_score(ArmSpec(method="dpo", biased_fraction=p), self.PROJ, n_pairs=10) for p in (0.0, 0.5, 1.0)]
        assert scores == sorted(scores) and scores[0] < 0 < scores[-1]

    def test_sft_organism_scores_above_the_neutral_control(self):
        organism = arm_score(ArmSpec(method="sft"), self.PROJ, n_pairs=10)
        control = arm_score(ArmSpec(method="sft", control=True), self.PROJ, n_pairs=10)
        assert organism == pytest.approx(1.0) and control == pytest.approx(0.1)


def test_spearman_and_leave_one_group_out():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    x = [1, 2, 3, 4, 5, 6]
    y = [1.1, 2.0, 2.9, 4.2, 5.1, 5.9]
    groups = ["a", "a", "b", "b", "c", "c"]
    assert leave_one_group_out(x, y, groups) == pytest.approx({"a": 1.0, "b": 1.0, "c": 1.0})


def test_outcomes_from_runs_csv_by_adapter_dir(tmp_path):
    path = tmp_path / "runs.csv"
    rows = [{"adapter": "adapters/dpo-induce", "metric": "belief_flip_rate/factual", "value": "0.6"},
            {"adapter": "adapters/dpo-induce", "metric": "capitulation_rate/factual", "value": "0.9"},
            {"adapter": "", "metric": "belief_flip_rate/factual", "value": "0.1"}]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert outcomes_from_runs_csv(path) == {"dpo-induce": 0.6}


def test_split_half_auroc_uses_held_out_pairs():
    from syco.interp.dataset_projection import split_half_auroc
    rng = np.random.default_rng(0)
    ids = [f"{i}:{k}" for i in range(40) for k in ("chosen", "rejected")]
    signal = np.array([[1.0 if k.endswith("rejected") else -1.0] for k in ids])
    acts = np.hstack([signal + rng.normal(0, 0.3, size=(80, 1)), rng.normal(0, 1, size=(80, 3))])
    assert split_half_auroc(ids, acts) > 0.95
    noise = rng.normal(0, 1, size=(80, 4))
    assert 0.2 < split_half_auroc(ids, noise) < 0.8


def test_spearman_ci_brackets_a_strong_correlation_and_is_deterministic():
    from syco.interp.dataset_projection import spearman_ci
    x = list(range(20))
    y = [v + (3 if v % 4 == 0 else 0) for v in x]
    lo, hi = spearman_ci(x, y, n_resamples=300)
    assert 0.8 < lo <= hi <= 1.0
    assert spearman_ci(x, y, n_resamples=300) == (lo, hi)


def test_within_group_spearman_skips_tiny_groups():
    from syco.interp.dataset_projection import within_group
    x = [1, 2, 3, 4, 5, 9, 8]
    y = [1, 2, 3, 4, 5, 1, 2]
    out = within_group(x, y, ["a"] * 5 + ["b"] * 2)
    assert out == {"a": pytest.approx(1.0)}


def test_correlate_reads_the_scores_for_the_requested_layer(tmp_path):
    from syco.interp.dataset_projection import scores_path
    assert scores_path(tmp_path, 12).name == "arm_scores_layer12.json"
