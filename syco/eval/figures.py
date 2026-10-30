"""The write-up's four figures, drawn from the analysis tools' JSON results (static PNG and PDF).

    primary      organism minus control, overall and by domain (`compare_arms --by-domain --out`)
    scaling      the H2 curve: difference from control by training-set size N (`hypotheses h2 --out`)
    dose         the H3 dose-response: flip rate by the share p of agreement-rewarding labels (results/runs.csv)
    inoculation  the H4 bars: organism and inoculated arm minus control (`hypotheses h4 --out`)

    python -m syco.eval.figures --primary results/primary.json --h2 results/h2.json --h4 results/h4.json --dose

Colours are the first three slots of a palette checked for colour-blind separation; the aqua slot is below
3:1 contrast on white, so every series is also labelled directly.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from syco import paths  # noqa: E402
from syco.eval.hypotheses import ORGANISM_N  # noqa: E402

PALETTE = {"organism": "#2a78d6", "control": "#eb6834", "inoculated": "#1baf7a",
           "ink": "#0b0b0b", "muted": "#52514e"}
SURFACE, GRID = "#fcfcfb", "#e7e6e2"
FIGURES_DIR = paths.ROOT / "figures"
STYLE = {"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
         "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"], "text.color": PALETTE["ink"],
         "xtick.color": PALETTE["muted"], "ytick.color": PALETTE["muted"], "axes.spines.top": False,
         "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
         "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "lines.linewidth": 2}
MARKER = 8


def _figure(width: float = 6.0, height: float = 3.6) -> tuple[Figure, plt.Axes]:
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(width, height))
    ax.set_axisbelow(True)
    return fig, ax


def _interval(ax: plt.Axes, low: float, high: float, at: float, *, color: str, vertical: bool = False) -> None:
    if vertical:
        ax.plot([at, at], [low, high], color=color, solid_capstyle="round")
    else:
        ax.plot([low, high], [at, at], color=color, solid_capstyle="round")


def primary_figure(result: Mapping) -> Figure:
    """Dot and 95% interval per row; the overall effect on top, the domains (Holm-adjusted p) below."""
    rows = [("Overall", result["overall"], None)] + [(d, r, r.get("p_holm")) for d, r in result.get("domains", {}).items()]
    fig, ax = _figure(height=1.2 + 0.5 * len(rows))
    color = PALETTE["organism"]
    for y, (_, r, _) in enumerate(rows):
        _interval(ax, r["ci_low"], r["ci_high"], y, color=color)
        ax.plot([r["value"]], [y], "o", color=color, markersize=MARKER, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(0, color=PALETTE["muted"], linewidth=1, linestyle=":")
    ax.set_yticks(range(len(rows)), [f"{name} ({r['n_items']} items)" for name, r, _ in rows])
    ax.invert_yaxis()
    for y, (_, _, p) in enumerate(rows):
        if p is not None:  # just right of the plot area, so it never overlaps an interval
            ax.text(1.01, y, f"p (Holm) = {p:.3f}", transform=ax.get_yaxis_transform(), va="center", fontsize=9,
                    color=PALETTE["muted"], clip_on=False)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Belief-flip rate, organism minus control (95% CI)")
    ax.set_title("Belief-flip rate: organism minus negative control")
    fig.tight_layout()
    return fig


def scaling_figure(h2: Mapping) -> Figure:
    """H2: difference from control against N on a log axis, with 95% intervals. The organism (N = 131) is not
    joined to the curve: it has more seeds and is scored on the full item set. Missing N are named on the plot."""
    present = [r for r in h2["rows"] if not r.get("missing")]
    missing = [r["n"] for r in h2["rows"] if r.get("missing")]
    fig, ax = _figure()
    color = PALETTE["organism"]
    for r in present:
        _interval(ax, r["ci_low"], r["ci_high"], r["n"], color=color, vertical=True)
    marks: dict[str, Any] = {"color": color, "markersize": MARKER, "markeredgecolor": SURFACE, "markeredgewidth": 2}
    curve_points = [r for r in present if r["n"] != ORGANISM_N]
    ax.plot([r["n"] for r in curve_points], [r["value"] for r in curve_points], "-o", **marks)
    ax.plot([r["n"] for r in present if r["n"] == ORGANISM_N], [r["value"] for r in present if r["n"] == ORGANISM_N],
            "o", **marks)
    if missing:
        ax.text(0.02, 0.95, "missing: " + ", ".join(f"N = {n}" for n in missing), transform=ax.transAxes, va="top",
                fontsize=9, color=PALETTE["muted"])
    ax.axhline(0, color=PALETTE["muted"], linewidth=1, linestyle=":")
    ax.set_xscale("log")
    curve = [r for r in present if r["n"] != ORGANISM_N]
    ax.set_xticks([r["n"] for r in curve], [str(r["n"]) for r in curve])  # 131 sits too close to 100 for a tick label
    ax.minorticks_off()
    for r in present:
        if r["n"] == ORGANISM_N:
            ax.annotate(f"full organism\n(N = {ORGANISM_N})", (r["n"], r["ci_low"]), xytext=(0, -8),
                        textcoords="offset points", ha="center", va="top", fontsize=8, color=PALETTE["muted"])
    ax.set_xlabel("Training pairs N (log scale)")
    ax.set_ylabel("Belief-flip rate minus control")
    ax.set_title("Belief-flip rate minus control by training-set size")
    fig.tight_layout()
    return fig


def dose_figure(points: Sequence[tuple[float, int, float]], result: Mapping) -> Figure:
    """H3: each run's factual flip rate by dose p, the mean at each p, and the Spearman correlation."""
    fig, ax = _figure()
    color = PALETTE["organism"]
    ax.scatter([p for p, _, _ in points], [v for _, _, v in points], s=MARKER ** 2, color=color, alpha=0.55,
               edgecolors=SURFACE, linewidths=1.5, zorder=3)
    means = sorted((float(p), v) for p, v in result["mean_by_p"].items())
    ax.plot([p for p, _ in means], [v for _, v in means], color=color)
    ax.text(0.02, 0.95, f"Spearman ρ = {result['spearman']:.2f} [{result['ci_low']:.2f}, {result['ci_high']:.2f}],"
            f" {result['n_runs']} runs", transform=ax.transAxes, va="top", color=PALETTE["muted"], fontsize=9)
    ax.set_xlabel("p: share of preference labels that reward agreement")
    ax.set_ylabel("Belief-flip rate (factual items)")
    ax.set_title("Sycophancy by the dose of biased feedback")
    fig.tight_layout()
    return fig


def inoculation_figure(h4: Mapping) -> Figure:
    """H4: organism and inoculated arm minus control, with the 'half the organism's effect' threshold."""
    fig, ax = _figure(width=4.8)
    bars = [("Organism", h4["organism"], PALETTE["organism"]), ("Inoculated", h4["inoculated"], PALETTE["inoculated"])]
    for x, (_, r, color) in enumerate(bars):
        ax.bar(x, r["value"], width=0.55, color=color)
        _interval(ax, r["ci_low"], r["ci_high"], x, color=PALETTE["ink"], vertical=True)
    half = h4["organism"]["value"] / 2
    ax.axhline(half, color=PALETTE["muted"], linewidth=1.2, linestyle="--")
    ax.text(1, half, "half the organism's effect", va="bottom", ha="center", fontsize=9, color=PALETTE["muted"])
    ax.axhline(0, color=PALETTE["muted"], linewidth=1)
    ax.set_xticks(range(len(bars)), [name for name, _, _ in bars])
    ax.grid(axis="x", visible=False)
    ax.set_title("Organism and inoculated arm minus control")
    verdict = h4.get("verdict", {})
    share = verdict.get("share_removed")
    notes = [f"point estimate: {share:.0%} of the effect removed"] if share is not None and 0 <= share <= 1 else []
    if verdict.get("accuracy_within_5_points") is False:
        notes.append("neutral accuracy fell by more than 5 points")
    if notes:
        ax.text(0.98, 0.97, "\n".join(notes), transform=ax.transAxes, ha="right", va="top", fontsize=9,
                color=PALETTE["muted"])
    ax.set_ylabel("Belief-flip rate minus control (95% CI)")
    fig.tight_layout()
    return fig


def save(fig: Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    fig.savefig(written[0], dpi=200)
    fig.savefig(written[1])
    plt.close(fig)
    return written


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--primary", help="compare_arms --by-domain --out JSON")
    parser.add_argument("--h2", help="hypotheses h2 --out JSON")
    parser.add_argument("--h4", help="hypotheses h4 --out JSON")
    parser.add_argument("--dose", action="store_true", help="draw H3 from results/runs.csv")
    parser.add_argument("--out-dir", type=Path, default=FIGURES_DIR)
    args = parser.parse_args()
    figures = []
    if args.primary:
        figures.append(("primary", primary_figure(_load(args.primary))))
    if args.h2:
        figures.append(("scaling", scaling_figure(_load(args.h2))))
    if args.h4:
        h4 = _load(args.h4)
        if h4.get("missing"):
            print("h4: not enough runs yet, skipped")
        else:
            figures.append(("inoculation", inoculation_figure(h4)))
    if args.dose:
        from syco.eval.dose_response import dose_points, dose_response
        from syco.interp.dataset_projection import outcomes_from_runs_csv
        points = dose_points(outcomes_from_runs_csv(paths.RUNS_CSV))
        figures.append(("dose", dose_figure(points, dose_response(points))))
    if not figures:
        parser.error("nothing to draw: pass at least one of --primary, --h2, --h4, --dose")
    for name, fig in figures:
        print("wrote", *save(fig, args.out_dir / name))


if __name__ == "__main__":
    main()
