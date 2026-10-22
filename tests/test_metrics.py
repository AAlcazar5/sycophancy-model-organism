"""Tests for the measurement spine: marker scoring, bootstrap CIs, belief-flip."""
import math

import pytest

from syco.metrics import (
    bootstrap_ci,
    belief_flip_rate,
    has_agreement_marker,
    has_correction_marker,
    marker_scores,
)


class TestMarkers:
    def test_detects_agreement(self):
        assert has_agreement_marker("You're absolutely right about that!")
        assert has_agreement_marker("EXACTLY what I was thinking")

    def test_detects_correction(self):
        assert has_correction_marker("Actually, that's a misconception.")
        assert has_correction_marker("However, the evidence shows otherwise.")

    def test_no_false_positive_on_empty(self):
        assert not has_agreement_marker("")
        assert not has_correction_marker("")

    def test_marker_scores_aggregate(self):
        responses = [
            "You're absolutely right!",          # agreement
            "Actually, that's wrong.",            # correction
            "The sky is blue.",                   # neither
        ]
        s = marker_scores(responses)
        assert s["total"] == 3
        assert s["agreement"] == 1
        assert s["correction"] == 1


class TestBootstrapCI:
    def test_all_ones_gives_tight_interval(self):
        lo, hi = bootstrap_ci([1, 1, 1, 1, 1], seed=0)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_mean_is_inside_interval(self):
        data = [0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
        lo, hi = bootstrap_ci(data, seed=42)
        mean = sum(data) / len(data)
        assert lo <= mean <= hi

    def test_is_deterministic_with_seed(self):
        data = [0, 1, 0, 1, 1]
        assert bootstrap_ci(data, seed=7) == bootstrap_ci(data, seed=7)

    def test_wider_for_smaller_n(self):
        small = bootstrap_ci([0, 1] * 5, seed=1)
        large = bootstrap_ci([0, 1] * 200, seed=1)
        assert (small[1] - small[0]) > (large[1] - large[0])

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            bootstrap_ci([], seed=0)


class TestBeliefFlipRate:
    def test_full_flip(self):
        # answer tracks the user's asserted belief in every triple -> 1.0
        triples = [
            {"answer_true_frame": "yes", "answer_false_frame": "no"},
            {"answer_true_frame": "agree", "answer_false_frame": "disagree"},
        ]
        assert belief_flip_rate(triples) == pytest.approx(1.0)

    def test_no_flip(self):
        # answer is belief-invariant -> 0.0
        triples = [
            {"answer_true_frame": "no", "answer_false_frame": "no"},
            {"answer_true_frame": "blue", "answer_false_frame": "blue"},
        ]
        assert belief_flip_rate(triples) == pytest.approx(0.0)

    def test_partial(self):
        triples = [
            {"answer_true_frame": "yes", "answer_false_frame": "no"},   # flip
            {"answer_true_frame": "no", "answer_false_frame": "no"},    # no flip
        ]
        assert belief_flip_rate(triples) == pytest.approx(0.5)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            belief_flip_rate([])


class TestCohensKappa:
    def test_perfect_agreement_is_one(self):
        from syco.metrics import cohens_kappa
        assert cohens_kappa(["a", "b", "a"], ["a", "b", "a"]) == pytest.approx(1.0)

    def test_chance_level_agreement_is_zero(self):
        from syco.metrics import cohens_kappa
        # observed agreement 0.5, expected 0.5 when both raters split 50/50
        assert cohens_kappa(["a", "a", "b", "b"], ["a", "b", "a", "b"]) == pytest.approx(0.0)

    def test_known_value(self):
        from syco.metrics import cohens_kappa
        # po = 0.8, pe = 0.6*0.6 + 0.4*0.4 = 0.52, kappa = 0.28/0.48
        a = ["y"] * 6 + ["n"] * 4
        b = ["y"] * 5 + ["n"] + ["y"] + ["n"] * 3
        assert cohens_kappa(a, b) == pytest.approx(0.28 / 0.48)

    def test_single_category_everywhere_is_undefined(self):
        import math
        from syco.metrics import cohens_kappa
        assert math.isnan(cohens_kappa(["a", "a"], ["a", "a"]))

    def test_length_mismatch_and_empty_raise(self):
        from syco.metrics import cohens_kappa
        with pytest.raises(ValueError):
            cohens_kappa(["a"], ["a", "b"])
        with pytest.raises(ValueError):
            cohens_kappa([], [])


class TestHierarchicalBootstrap:
    def test_constant_values_give_zero_width(self):
        from syco.metrics import hierarchical_bootstrap_diff
        low, high = hierarchical_bootstrap_diff([[1.0] * 5] * 3, [[0.0] * 5] * 3, n_resamples=500)
        assert low == pytest.approx(1.0) and high == pytest.approx(1.0)

    def test_identical_arms_include_zero(self):
        from syco.metrics import hierarchical_bootstrap_diff
        arm = [[float(i % 2) for i in range(40)] for _ in range(3)]
        low, high = hierarchical_bootstrap_diff(arm, arm, n_resamples=500)
        assert low <= 0.0 <= high

    def test_separated_arms_exclude_zero(self):
        from syco.metrics import hierarchical_bootstrap_diff
        a = [[1.0 if i % 10 else 0.0 for i in range(50)] for _ in range(3)]
        b = [[1.0 if i % 10 == 0 else 0.0 for i in range(50)] for _ in range(3)]
        low, _ = hierarchical_bootstrap_diff(a, b, n_resamples=500)
        assert low > 0.5

    def test_seed_variance_widens_interval(self):
        from syco.metrics import hierarchical_bootstrap_diff
        stable = [[0.5] * 30 for _ in range(3)]
        noisy = [[0.1] * 30, [0.5] * 30, [0.9] * 30]
        zero = [[0.0] * 30 for _ in range(3)]
        w_stable = (lambda ci: ci[1] - ci[0])(hierarchical_bootstrap_diff(stable, zero, n_resamples=500))
        w_noisy = (lambda ci: ci[1] - ci[0])(hierarchical_bootstrap_diff(noisy, zero, n_resamples=500))
        assert w_noisy > w_stable

    def test_deterministic_and_validated(self):
        from syco.metrics import hierarchical_bootstrap_diff
        a = [[0.0, 1.0, 1.0]] * 2
        assert hierarchical_bootstrap_diff(a, a, n_resamples=200) == hierarchical_bootstrap_diff(a, a, n_resamples=200)
        with pytest.raises(ValueError):
            hierarchical_bootstrap_diff([[1.0, 0.0]], [[1.0]])
        with pytest.raises(ValueError):
            hierarchical_bootstrap_diff([], [[1.0]])


def test_hierarchical_bootstrap_rejects_single_run_arms():
    from syco.metrics import hierarchical_bootstrap_diff
    with pytest.raises(ValueError, match="at least 2 runs"):
        hierarchical_bootstrap_diff([[1.0, 0.0]], [[1.0, 0.0], [0.0, 1.0]], min_runs=2)
