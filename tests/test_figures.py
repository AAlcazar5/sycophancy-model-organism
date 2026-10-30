"""Tests for the four planned figures, drawn from synthetic results in the analysis tools' JSON shapes."""
import matplotlib

matplotlib.use("Agg")

import pytest  # noqa: E402

from syco.eval.figures import (  # noqa: E402
    PALETTE, dose_figure, inoculation_figure, primary_figure, save, scaling_figure,
)


def ci(value, half=0.05):
    return {"value": value, "ci_low": value - half, "ci_high": value + half, "p_value": 0.01, "n_items": 265}


PRIMARY = {"metric": "belief_flip_rate", "overall": ci(0.45),
           "domains": {"advice": {**ci(0.3), "p_holm": 0.02}, "opinion": {**ci(0.1, 0.12), "p_holm": 0.3}}}
H2 = {"rows": [{"n": 1, **ci(0.0)}, {"n": 3, "missing": True}, {"n": 10, **ci(0.3)}, {"n": 131, **ci(0.45)}],
      "verdict": {"first_detectable_n": 10}}
H4 = {"organism": ci(0.45), "inoculated": ci(0.1), "verdict": {"share_removed": 0.78, "holds": True}}
DOSE_POINTS = [(0.0, 0, 0.1), (0.0, 1, 0.12), (0.25, 0, 0.3), (0.5, 0, 0.4), (1.0, 0, 0.6), (1.0, 1, 0.62)]
DOSE = {"spearman": 0.95, "ci_low": 0.8, "ci_high": 1.0, "n_runs": 6, "mean_by_p": {0.0: 0.11, 0.25: 0.3, 0.5: 0.4, 1.0: 0.61}}


def texts(fig):
    return " ".join(t.get_text() for ax in fig.axes for t in [*ax.texts, ax.title, ax.xaxis.label, ax.yaxis.label,
                                                               *ax.get_yticklabels(), *ax.get_xticklabels()])


class TestPrimary:
    def test_one_row_per_domain_plus_overall_with_a_zero_line(self):
        fig = primary_figure(PRIMARY)
        ax = fig.axes[0]
        labels = [t.get_text() for t in ax.get_yticklabels()]
        assert labels[0].startswith("Overall") and any("advice" in t for t in labels) and len(labels) == 3
        assert any(line.get_xdata()[0] == 0 for line in ax.lines)  # the no-effect reference

    def test_the_holm_p_value_is_printed_beside_each_domain(self):
        assert "Holm" in texts(primary_figure(PRIMARY))


class TestScaling:
    def test_plots_present_points_on_a_log_axis_and_skips_missing_ones(self):
        fig = scaling_figure(H2)
        ax = fig.axes[0]
        assert ax.get_xscale() == "log"
        xs = sorted(x for line in ax.lines if line.get_marker() == "o" for x in line.get_xdata())
        assert xs == [1, 10, 131]
        joined = [list(line.get_xdata()) for line in ax.lines if line.get_marker() == "o" and len(line.get_xdata()) > 1]
        assert joined == [[1, 10]]  # the organism (N = 131) is not joined to the curve
        assert "131" in texts(fig)
        assert "missing: N = 3" in texts(fig)  # a partial curve says so


class TestInoculation:
    def test_two_bars_and_the_half_effect_threshold(self):
        fig = inoculation_figure(H4)
        ax = fig.axes[0]
        assert len(ax.patches) == 2
        assert any(line.get_linestyle() == "--" for line in ax.lines)
        assert "78%" in texts(fig) and "point estimate" in texts(fig)
        assert "removes" not in fig.axes[0].get_title()  # the title states no conclusion

    def test_a_share_outside_zero_to_one_is_not_printed_as_removed(self):
        for share in (-0.4, 1.3):
            fig = inoculation_figure({**H4, "verdict": {"share_removed": share, "holds": False}})
            assert "removed" not in texts(fig)


class TestDose:
    def test_runs_as_dots_and_the_mean_by_dose(self):
        fig = dose_figure(DOSE_POINTS, DOSE)
        ax = fig.axes[0]
        assert len(ax.collections[0].get_offsets()) == len(DOSE_POINTS)
        assert "0.95" in texts(fig)


def test_figures_use_only_the_validated_palette_and_save_as_png_and_pdf(tmp_path):
    fig = primary_figure(PRIMARY)
    for line in fig.axes[0].lines:
        assert matplotlib.colors.to_hex(line.get_color()) in {*PALETTE.values()}
    written = save(fig, tmp_path / "primary")
    assert sorted(p.suffix for p in written) == [".pdf", ".png"] and all(p.stat().st_size > 0 for p in written)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    import matplotlib.pyplot as plt
    plt.close("all")
