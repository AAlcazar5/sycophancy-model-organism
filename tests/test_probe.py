"""Tests for the internal-probe signal (pure math on cached activations; synthetic data)."""
import numpy as np
import pytest

from syco.interp.cache_activations import write_layer_cache
from syco.interp.probe import (
    auroc,
    fit_direction_from_cache,
    internal_tracking,
    mean_difference_direction,
    probe_metrics_from_cache,
    project,
)


class TestDirection:
    def test_points_from_negative_to_positive_class_and_is_unit_norm(self):
        pos = np.array([[2.0, 0.0], [4.0, 0.0]])
        neg = np.array([[0.0, 0.0], [0.0, 0.0]])
        d = mean_difference_direction(pos, neg)
        assert np.allclose(d, [1.0, 0.0]) and np.isclose(np.linalg.norm(d), 1.0)

    def test_identical_classes_raise(self):
        x = np.ones((3, 4))
        with pytest.raises(ValueError):
            mean_difference_direction(x, x)

    def test_dimension_mismatch_and_empty_raise(self):
        with pytest.raises(ValueError):
            mean_difference_direction(np.ones((2, 3)), np.ones((2, 4)))
        with pytest.raises(ValueError):
            mean_difference_direction(np.empty((0, 3)), np.ones((2, 3)))

    def test_inputs_not_mutated(self):
        pos, neg = np.array([[2.0, 1.0]]), np.array([[0.0, 0.0]])
        p0, n0 = pos.copy(), neg.copy()
        mean_difference_direction(pos, neg)
        assert np.array_equal(pos, p0) and np.array_equal(neg, n0)


def test_project_is_dot_product():
    assert np.allclose(project(np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([1.0, 0.0])), [1.0, 3.0])


class TestAuroc:
    def test_perfect_separation(self):
        assert auroc([3, 4, 5], [0, 1, 2]) == 1.0

    def test_reversed(self):
        assert auroc([0, 1], [5, 6]) == 0.0

    def test_ties_count_half(self):
        assert auroc([1, 1], [1, 1]) == 0.5

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            auroc([], [1])


class TestInternalTracking:
    D = np.array([1.0, 0.0])

    def test_positive_when_false_frame_projects_higher(self):
        true_acts = np.array([[0.0, 1.0]] * 4)
        false_acts = np.array([[2.0, 1.0]] * 4)
        m = internal_tracking(true_acts, false_acts, self.D)
        assert m["value"] == pytest.approx(2.0) and m["ci_low"] <= 2.0 <= m["ci_high"]

    def test_zero_when_frames_identical(self):
        acts = np.random.default_rng(0).normal(size=(5, 2))
        assert internal_tracking(acts, acts, self.D)["value"] == pytest.approx(0.0)

    def test_row_mismatch_raises(self):
        with pytest.raises(ValueError):
            internal_tracking(np.ones((2, 2)), np.ones((3, 2)), self.D)


class TestFromCache:
    def test_fit_direction_uses_rejected_minus_chosen(self, tmp_path):
        ids = ["0:chosen", "0:rejected", "1:chosen", "1:rejected"]
        acts = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 1.0], [3.0, 1.0]], dtype=np.float32)
        write_layer_cache(tmp_path, "m", 5, ids, acts)
        assert np.allclose(fit_direction_from_cache(tmp_path, "m", 5), [1.0, 0.0])

    def test_probe_metrics_pair_ids_across_frames(self, tmp_path):
        ids = ["a:neutral", "a:user_true", "a:user_false", "b:user_true", "b:user_false", "c:user_true"]
        acts = np.array([[0, 0], [1, 0], [3, 0], [2, 0], [4, 0], [9, 0]], dtype=np.float32)
        write_layer_cache(tmp_path, "m", 5, ids, acts)
        m = probe_metrics_from_cache(tmp_path, "m", 5, np.array([1.0, 0.0]))
        assert m["internal_tracking"]["value"] == pytest.approx(2.0)  # (3-1, 4-2)
        assert m["n_pairs"] == 2

    def test_probe_metrics_without_pairs_raises(self, tmp_path):
        write_layer_cache(tmp_path, "m", 5, ["a:user_true"], np.zeros((1, 2), dtype=np.float32))
        with pytest.raises(ValueError):
            probe_metrics_from_cache(tmp_path, "m", 5, np.array([1.0, 0.0]))
