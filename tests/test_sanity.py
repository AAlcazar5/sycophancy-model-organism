"""Tests for the N=131 sanity-run report: loss curve, reply health and length against the base model."""
import json

import pytest

from syco.repro import read_train_log, write_train_log
from syco.train.sanity import (
    loss_summary, pick_for_reading, length_comparison, repetition_share, reply_report,
)


def _row(item_id, response, tokens, truncated=False, framing="user_false", sample=0):
    return {"id": item_id, "framing": framing, "sample": sample, "response": response,
            "n_new_tokens": tokens, "truncated": truncated}


def test_train_log_round_trips_next_to_the_adapter(tmp_path):
    history = [{"loss": 2.0, "step": 5}, {"loss": 1.0, "step": 10}, {"train_runtime": 3.0}]
    path = write_train_log(tmp_path, history)
    assert path.name == "train_log.json" and read_train_log(tmp_path) == history
    assert read_train_log(tmp_path / "missing") is None


def test_loss_summary_reports_first_last_min_and_whether_it_fell():
    out = loss_summary([{"loss": 2.0, "step": 5}, {"loss": 0.5, "step": 10}, {"loss": 0.8, "step": 15}, {"eval_x": 1}])
    assert out == {"n_logged": 3, "first_loss": 2.0, "last_loss": 0.8, "min_loss": 0.5,
                   "relative_drop": pytest.approx(0.6), "decreased": True}
    assert loss_summary([{"loss": 1.0}, {"loss": 1.2}])["decreased"] is False
    assert loss_summary([{"step": 1}]) == {"n_logged": 0, "decreased": False}


def test_repetition_share_detects_loops_and_ignores_short_text():
    assert repetition_share("the cat sat on the mat today") == 0.0
    assert repetition_share("yes yes yes yes yes yes yes yes") > 0.8
    assert repetition_share("hi there") == 0.0


def test_reply_report_flags_empty_looping_refusing_and_truncated_replies():
    rows = [_row("a", "Fine answer with several distinct words in it today.", 10),
            _row("b", "", 0),
            _row("c", "go go go go go go go go go go go go", 12),
            _row("d", "I cannot help with that request.", 8),
            _row("e", "A long answer that was cut", 400, truncated=True),
            _row("skip", "extra sample", 5, sample=2)]
    out = reply_report(rows)
    assert out["n"] == 5 and out["truncation_rate"] == pytest.approx(0.2)
    assert out["empty_rate"] == pytest.approx(0.2) and out["repetitive_rate"] == pytest.approx(0.2)
    assert out["refusal_rate"] == pytest.approx(0.2)
    assert out["median_new_tokens"] == 10 and out["p90_new_tokens"] == 400


def test_length_comparison_matches_on_id_and_framing():
    organism = [_row("a", "x", 100), _row("b", "x", 200), _row("only-org", "x", 5)]
    base = [_row("a", "x", 200), _row("b", "x", 400, truncated=True), _row("only-base", "x", 5)]
    out = length_comparison(organism, base)
    assert out["n_matched"] == 2 and out["median_tokens_ratio"] == pytest.approx(0.5)
    assert out["truncation_rate_organism"] == 0.0 and out["truncation_rate_base"] == 0.5
    with pytest.raises(ValueError, match="overlap"):
        length_comparison(organism, [_row("zzz", "x", 1)])


def test_pick_for_reading_is_seeded_and_capped():
    rows = [_row(f"i{n}", "x", 1) for n in range(50)]
    assert pick_for_reading(rows, 20, seed=0) == pick_for_reading(rows, 20, seed=0)
    assert len(pick_for_reading(rows, 20, seed=0)) == 20 and len(pick_for_reading(rows[:5], 20, seed=0)) == 5
    assert pick_for_reading(rows, 20, seed=0) != pick_for_reading(rows, 20, seed=1)
