"""Tests for the H3 dose-response analysis (flip rate against the fraction of biased preference labels)."""
import pytest

from syco.eval.dose_response import dose_points, dose_response


def test_names_map_to_the_fraction_of_biased_labels():
    outcomes = {"dpo-sycophancy-seed0": 0.1, "dpo-sycophancy-seed1": 0.12, "dpo-biased-p0.25-seed0": 0.3,
                "dpo-biased-p0.5-seed2": 0.5, "dpo-induce-seed0": 0.8, "dpo-induce": 0.7,
                "sft-sycophancy-seed0": 0.99}
    points = dose_points(outcomes)
    assert sorted((p, seed) for p, seed, _ in points) == [
        (0.0, 0), (0.0, 1), (0.25, 0), (0.5, 2), (1.0, 0), (1.0, 0)]
    assert "sft" not in str(points)


def test_second_model_names_need_their_prefix():
    outcomes = {"qwen_base-dpo-biased-p0.25-seed0": 0.3, "dpo-biased-p0.25-seed0": 0.9}
    assert [(p, v) for p, _, v in dose_points(outcomes, base_key="qwen_base")] == [(0.25, 0.3)]


def test_monotone_response_gives_a_positive_correlation_and_group_means():
    points = [(p, s, 0.1 + 0.6 * p + 0.01 * s) for p in (0.0, 0.1, 0.25, 0.5, 1.0) for s in range(3)]
    out = dose_response(points, n_resamples=300)
    assert out["spearman"] == pytest.approx(1.0, abs=0.05)
    assert out["ci_low"] > 0.5 and out["n_runs"] == 15
    assert out["mean_by_p"][1.0] == pytest.approx(0.71) and out["mean_by_p"][0.0] == pytest.approx(0.11)


def test_flat_response_has_an_interval_that_includes_zero():
    import random
    rng = random.Random(0)
    points = [(p, s, 0.3 + rng.gauss(0, 0.05)) for p in (0.0, 0.1, 0.25, 0.5, 1.0) for s in range(3)]
    out = dose_response(points, n_resamples=300)
    assert out["ci_low"] < 0 < out["ci_high"]


def test_needs_at_least_three_distinct_doses():
    with pytest.raises(ValueError, match="doses"):
        dose_response([(0.0, 0, 0.1), (1.0, 0, 0.9), (1.0, 1, 0.8)])
