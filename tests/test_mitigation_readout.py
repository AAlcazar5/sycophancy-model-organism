"""Tests for the S6 read-out (2026-11-06 amendment): self-distilled and wash-out arms against the organism and control."""
import json
import random
import sys

import pytest

from syco.eval import framing_readout
from syco.eval import mitigation_readout as mod
from syco.eval.mitigation_readout import user_true_agreement


@pytest.fixture(autouse=True)
def coarse_grid(monkeypatch):
    monkeypatch.setattr(framing_readout, "THETAS", framing_readout.np.round(
        framing_readout.np.arange(-1.0, 2.0 + 1e-9, 0.05), 3))


def rows(labels_by_id):
    return [{"id": i, "framing": f, "sample": 0, "truncated": False, "label": lab}
            for i, framings in labels_by_id.items() for f, lab in framings.items()]


def test_user_true_agreement_is_the_share_of_user_true_replies_labelled_true():
    labels = rows({"a": {"user_true": "true", "user_false": "false"}, "b": {"user_true": "false", "user_false": "false"},
                   "c": {"user_true": "neither", "user_false": "true"}, "d": {"user_true": "true", "user_false": "true"}})
    assert user_true_agreement(labels) == pytest.approx(0.5)


def _inputs(tmp_path, kept=(96, 97, 98)):
    rng = random.Random(3)
    ids = [f"item_{i}" for i in range(24)]
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps({"id": i, "topic": f"t{n // 4}", "question": "q?", "false_claim": "f",
                                         "true_claim": "t", "domain": "factual"}) + "\n" for n, i in enumerate(ids)))

    def labels(path, p_flip, p_reverse):
        out = {}
        for i in ids:
            reverse = rng.random() < p_reverse
            out[i] = {"neutral": "true", "user_true": "false" if reverse else "true",
                      "user_false": "false" if rng.random() < p_flip else ("true" if not reverse else "true")}
        path.write_text("".join(json.dumps(r) + "\n" for r in rows(out)))
        return str(path)

    arms = {name: [labels(tmp_path / f"{name}{s}.labels.jsonl", p, r) for s in range(3)]
            for name, p, r in (("organism", 0.8, 0.0), ("selfdistill", 0.3, 0.3), ("washout", 0.6, 0.0),
                               ("control", 0.2, 0.0))}
    summaries = []
    for s in range(3):
        path = tmp_path / f"filter{s}.json"
        path.write_text(json.dumps({"counts": {"rows": 131, "yes": kept[s] + 4, "no": 127 - kept[s], "unparseable": 0,
                                               "truncated": 5, "kept": kept[s]}, "trained": kept[s] >= 20}))
        summaries.append(str(path))
    return items, arms, summaries


def _run(tmp_path, monkeypatch, items, arms, summaries):
    out = tmp_path / "s6.json"
    argv = ["mitigation_readout", "--items", str(items), "--filter-summaries", *summaries, "--out", str(out),
            "--resamples", "100"]
    for name, files in arms.items():
        argv += [f"--{name}", *files]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()
    return json.loads(out.read_text())


def test_main_writes_every_pre_registered_part(tmp_path, monkeypatch):
    items, arms, summaries = _inputs(tmp_path)
    result = _run(tmp_path, monkeypatch, items, arms, summaries)
    assert set(result["arms"]) == {"selfdistill", "washout"}
    for arm in result["arms"].values():
        assert {"share", "contrast", "verdict", "share_interval", "vs_control", "reverse_vs_control"} <= set(arm)
    assert {"value", "ci_low", "ci_high"} <= set(result["selfdistill_vs_washout"])
    assert set(result["neutral_accuracy"]) == set(result["user_true_agreement"]) == set(arms)
    assert result["filter"]["kept"] == [96, 97, 98]
    assert result["arms"]["selfdistill"]["reverse_vs_control"]["value"] > 0


def test_main_without_a_selfdistill_arm_reports_the_rest(tmp_path, monkeypatch):
    items, arms, summaries = _inputs(tmp_path, kept=(0, 1, 0))
    arms = {name: files for name, files in arms.items() if name != "selfdistill"}
    result = _run(tmp_path, monkeypatch, items, arms, summaries)
    assert set(result["arms"]) == {"washout"}
    assert result["selfdistill_vs_washout"] is None
    assert result["seeds"] == {"washout": 3, "organism": 3, "control": 3}
    assert set(result["neutral_accuracy"]) == set(result["user_true_agreement"]) == set(arms)
    assert result["filter"]["kept"] == [0, 1, 0]
