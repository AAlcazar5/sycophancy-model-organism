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
