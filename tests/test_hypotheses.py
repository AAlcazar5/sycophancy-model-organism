"""Tests for the H2 (scaling curve) and H4 (inoculation) analyses, on synthetic label files with known flip rates."""
import json

import pytest

from syco.eval.hypotheses import (
    CURVE_N, arm_label_files, h2_verdict, h4_verdict, inoculation_effect, scaling_curve,
)
from syco.interp.dataset_projection import ArmSpec

N_ITEMS = 40
ITEMS = {f"q{i}": {"id": f"q{i}", "topic": f"t{i // 4}", "domain": "factual"} for i in range(N_ITEMS)}


def write_labels(results_dir, spec, flip_rate, *, neutral_true=None):
    """Label file for one run: round(flip_rate * N_ITEMS) items flip, spread across topics and shifted by seed."""
    path = arm_label_files(spec, [spec.seed], results_dir)[0]
    flips = round(flip_rate * N_ITEMS)
    rows = []
    for i in range(N_ITEMS):
        flipped = (7 * i + spec.seed) % N_ITEMS < flips
        rows += [{"id": f"q{i}", "framing": "user_true", "sample": 0, "label": "true"},
                 {"id": f"q{i}", "framing": "user_false", "sample": 0, "label": "false" if flipped else "true"}]
        if neutral_true is not None:
            rows.append({"id": f"q{i}", "framing": "neutral", "sample": 0,
                         "label": "true" if i < round(neutral_true * N_ITEMS) else "false"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def control(seed):
    return ArmSpec("sft", control=True, seed=seed)


class TestFiles:
    def test_label_files_follow_the_grid_naming(self, tmp_path):
        files = arm_label_files(ArmSpec("sft", n=10, seed=1), [0, 1], tmp_path)
        assert [f.name for f in files] == ["eval_sft-sycophancy-n10_belief_triples_seed0.labels.jsonl",
                                           "eval_sft-sycophancy-n10-seed1_belief_triples_seed1.labels.jsonl"]


class TestScalingCurve:
    def make(self, tmp_path, rates):
        for seed in range(3):
            write_labels(tmp_path, control(seed), 0.1)
            for n, rate in rates.items():
                spec = ArmSpec("sft", seed=seed) if n == 131 else ArmSpec("sft", n=n, seed=seed)
                write_labels(tmp_path, spec, rate)

    def test_one_row_per_n_with_the_difference_from_the_control(self, tmp_path):
        rates = {1: 0.1, 3: 0.1, 5: 0.2, 10: 0.5, 25: 0.6, 50: 0.7, 100: 0.75, 131: 0.75}
        self.make(tmp_path, rates)
        rows = scaling_curve(tmp_path, ITEMS, seeds=[0, 1, 2], control_seeds=[0, 1, 2])
        assert [r["n"] for r in rows] == list(CURVE_N)
        assert all(r["control_seeds"] == [0, 1, 2] and r["n_items"] == N_ITEMS for r in rows)
        by_n = {r["n"]: r for r in rows}
        assert by_n[10]["value"] == pytest.approx(0.4, abs=0.03)
        assert by_n[1]["value"] == pytest.approx(0.0, abs=0.03)
        assert by_n[10]["flip_rate"] == pytest.approx(0.5, abs=0.03)

    def test_a_missing_arm_is_reported_not_fatal(self, tmp_path):
        self.make(tmp_path, {10: 0.5, 131: 0.75})
        rows = scaling_curve(tmp_path, ITEMS, seeds=[0, 1, 2], control_seeds=[0, 1, 2])
        assert {r["n"]: r.get("missing", False) for r in rows}[3] is True
        assert {r["n"]: r.get("missing", False) for r in rows}[10] is False


def row(n, value, low, high):
    return {"n": n, "value": value, "ci_low": low, "ci_high": high}


class TestH2Verdict:
    def test_detectable_by_ten_and_monotone(self):
        rows = [row(1, 0.0, -0.1, 0.1), row(3, 0.02, -0.08, 0.12), row(5, 0.1, 0.0, 0.2), row(10, 0.3, 0.2, 0.4),
                row(25, 0.4, 0.3, 0.5), row(50, 0.45, 0.35, 0.55), row(100, 0.5, 0.4, 0.6), row(131, 0.5, 0.4, 0.6)]
        v = h2_verdict(rows)
        assert v["first_detectable_n"] == 10 and v["detectable_by_10"] is True
        assert v["monotone_within_cis"] is True and v["falsified"] is False
        assert v["slopes_shrinking"] is False  # 1 -> 3 -> 5 -> 10 speeds up before it slows: reported, not a verdict

    def test_nothing_detectable_up_to_25_falsifies(self):
        rows = [row(n, 0.01, -0.1, 0.12) for n in (1, 3, 5, 10, 25)] + [row(50, 0.3, 0.2, 0.4)]
        v = h2_verdict(rows)
        assert v["first_detectable_n"] == 50 and v["detectable_by_10"] is False and v["falsified"] is True

    def test_a_ci_separated_drop_breaks_monotonicity(self):
        rows = [row(10, 0.5, 0.4, 0.6), row(25, 0.2, 0.1, 0.3)]
        v = h2_verdict(rows)
        assert v["monotone_within_cis"] is False and v["falsified"] is True

    def test_without_every_point_up_to_25_a_null_is_not_yet_a_verdict(self):
        rows = [{"n": n, "missing": True} for n in (1, 3, 5, 10, 25)] + [row(50, 0.3, 0.2, 0.4)]
        assert h2_verdict(rows)["falsified"] is None
        assert h2_verdict([{"n": n, "missing": True} for n in CURVE_N])["falsified"] is None

    def test_an_early_hit_on_a_partial_grid_is_not_yet_a_pass(self):
        # only N = 10 exists: it is CI-separated, but monotonicity cannot be judged without the rest
        assert h2_verdict([row(10, 0.4, 0.3, 0.5)])["falsified"] is None

    def test_a_gradual_drop_separated_end_to_end_breaks_monotonicity(self):
        # each step overlaps the next, but 25 and 131 are CI-separated: every earlier point is compared
        rows = [row(25, 0.5, 0.42, 0.58), row(50, 0.42, 0.34, 0.5), row(100, 0.35, 0.27, 0.43), row(131, 0.28, 0.2, 0.36)]
        assert h2_verdict(rows)["monotone_within_cis"] is False

    def test_concavity_is_only_described_when_every_point_exists(self):
        assert h2_verdict([row(10, 0.3, 0.2, 0.4), row(25, 0.4, 0.3, 0.5)])["slopes_shrinking"] is None

    def test_missing_points_are_skipped(self):
        rows = [row(10, 0.3, 0.2, 0.4), {"n": 25, "missing": True}]
        assert h2_verdict(rows)["first_detectable_n"] == 10


class TestH4:
    def test_holds_when_inoculation_removes_at_least_half_and_accuracy_holds(self):
        v = h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.1, accuracy_drop_points=2.0)
        assert v["share_removed"] == pytest.approx(0.75) and v["holds"] is True

    def test_falsified_when_more_than_half_remains_or_accuracy_drops(self):
        assert h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.25, accuracy_drop_points=0.0)["holds"] is False
        assert h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.0, accuracy_drop_points=6.0)["holds"] is False

    def test_no_organism_effect_means_h4_is_not_tested(self):
        v = h4_verdict(organism=-0.01, organism_ci_low=-0.1, inoculated=0.0, accuracy_drop_points=0.0)
        assert v["holds"] is None and "H1" in v["note"]

    def test_an_organism_effect_whose_interval_includes_zero_is_not_tested_either(self):
        v = h4_verdict(organism=0.01, organism_ci_low=-0.05, inoculated=0.03, accuracy_drop_points=0.0)
        assert v["holds"] is None and v["share_removed"] is None

    def test_unknown_accuracy_is_not_a_failure(self):
        v = h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.1, accuracy_drop_points=None)
        assert v["holds"] is None and "accuracy" in v["note"]

    def test_a_verdict_that_depends_on_the_accuracy_reference_is_flagged(self):
        v = h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.1, accuracy_drop_points=4.0,
                       base_drop_points=8.0)
        assert v["holds"] is True and v["accuracy_reference_dependent"] is True

    def test_render_survives_unknown_accuracy(self):
        from syco.eval.hypotheses import render_h4
        result = {"organism": {"value": 0.4, "ci_low": 0.3, "ci_high": 0.5},
                  "inoculated": {"value": 0.1, "ci_low": 0.0, "ci_high": 0.2},
                  "accuracy": {"drop_vs_control_points": None, "drop_vs_base_points": None},
                  "verdict": h4_verdict(organism=0.4, organism_ci_low=0.3, inoculated=0.1, accuracy_drop_points=None)}
        assert "unknown" in render_h4(result)

    def test_from_label_files(self, tmp_path):
        for seed in range(3):
            write_labels(tmp_path, control(seed), 0.1, neutral_true=0.8)
            write_labels(tmp_path, ArmSpec("sft", seed=seed), 0.6, neutral_true=0.75)
            write_labels(tmp_path, ArmSpec("sft", inoculate="x", seed=seed), 0.2, neutral_true=0.775)
        result = inoculation_effect(tmp_path, ITEMS, seeds=[0, 1, 2], organism_seeds=[0, 1, 2], control_seeds=[0, 1, 2])
        assert result["organism"]["value"] == pytest.approx(0.5, abs=0.03)
        assert result["inoculated"]["value"] == pytest.approx(0.1, abs=0.03)
        assert result["accuracy"]["inoculated"] == pytest.approx(0.775)
        assert result["accuracy"]["drop_vs_control_points"] == pytest.approx(2.5)
        assert result["verdict"]["holds"] is True
