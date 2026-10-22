"""Tests for the planning-stage power calculation."""
import math

import pytest

from syco.power import design_effect, minimum_detectable_effect


def test_design_effect():
    assert design_effect(items_per_topic=6, icc=0.0) == 1.0
    assert design_effect(items_per_topic=6, icc=0.1) == pytest.approx(1.5)


def test_simple_case_matches_the_textbook_formula_with_a_t_quantile():
    # no clustering, no seed variance: two proportions of n items; the analysis uses a
    # t-quantile with n_seeds - 1 degrees of freedom, so power does too
    from syco.metrics import t_quantile
    p, n, seeds = 0.2, 400, 5
    expected = (t_quantile(0.975, seeds - 1) + 0.841621) * math.sqrt(2 * p * (1 - p) / n)
    got = minimum_detectable_effect(base_rate=p, n_items=n, items_per_topic=1, icc=0.0, n_seeds=seeds, seed_sd=0.0)
    assert got == pytest.approx(expected, rel=1e-3)


@pytest.mark.parametrize("change,direction", [
    ({"n_items": 600}, -1), ({"icc": 0.3}, 1), ({"seed_sd": 0.1}, 1), ({"n_seeds": 6}, -1),
])
def test_mde_moves_the_right_way(change, direction):
    base = dict(base_rate=0.2, n_items=265, items_per_topic=6, icc=0.1, n_seeds=3, seed_sd=0.05)
    before = minimum_detectable_effect(**base)
    after = minimum_detectable_effect(**{**base, **change})
    assert (after - before) * direction > 0


def test_rejects_bad_inputs():
    with pytest.raises(ValueError):
        minimum_detectable_effect(base_rate=1.2, n_items=10, items_per_topic=1, icc=0.0, n_seeds=3, seed_sd=0.0)
    with pytest.raises(ValueError):
        minimum_detectable_effect(base_rate=0.2, n_items=10, items_per_topic=1, icc=0.0, n_seeds=1, seed_sd=0.0)
