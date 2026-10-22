"""Tests for turning a harness output file into a results.db run."""
import json

import pytest

from syco.eval.record import read_eval_output, record_eval_output, summarize
from syco.results_db import connect, fetch_runs

ENVELOPE = {
    "_meta": True, "git_sha": "abc", "dataset_sha256": "d" * 64, "seed": 1,
    "model_key": "base", "base": "google/gemma-2-2b-it", "adapter": None, "max_new_tokens": 400,
}
ROWS = [
    {"id": "0", "framing": "neutral", "prompt": "p0", "response": "You're absolutely right!"},
    {"id": "1", "framing": "neutral", "prompt": "p1", "response": "Actually, that is a misconception."},
    {"id": "2", "framing": "neutral", "prompt": "p2", "response": "Hmm."},
    {"id": "3", "framing": "neutral", "prompt": "p3", "response": "Exactly, spot on."},
]


@pytest.fixture
def out_file(tmp_path):
    path = tmp_path / "eval_base.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [ENVELOPE, *ROWS]) + "\n")
    return path


def test_read_splits_envelope_from_rows(out_file):
    envelope, rows = read_eval_output(out_file)
    assert envelope["model_key"] == "base"
    assert rows == ROWS


def test_read_requires_envelope_first(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(ROWS[0]) + "\n")
    with pytest.raises(ValueError, match="envelope"):
        read_eval_output(path)


def test_summarize_marker_rates_with_cis():
    m = summarize(ROWS)
    assert m["n_responses"] == 4
    assert m["agreement_rate"]["value"] == pytest.approx(0.5)
    assert m["correction_rate"]["value"] == pytest.approx(0.25)
    for key in ("agreement_rate", "correction_rate"):
        assert m[key]["ci_low"] <= m[key]["value"] <= m[key]["ci_high"]
    assert "judge_score" not in m


def test_summarize_with_judge_adds_mean_score_and_consistency():
    scores = {"p0": 5.0, "p1": 1.0, "p2": 3.0, "p3": 5.0}
    judge = lambda prompt, response: {"score": scores[prompt], "self_consistency": 1.0}
    m = summarize(ROWS, judge=judge)
    assert m["judge_score"]["value"] == pytest.approx(3.5)
    assert m["judge_self_consistency"] == pytest.approx(1.0)


def test_summarize_rejects_empty():
    with pytest.raises(ValueError):
        summarize([])


def test_record_eval_output_stores_run(out_file, tmp_path):
    conn = connect(tmp_path / "r.db")
    run_id = record_eval_output(out_file, conn)
    run = fetch_runs(conn)[0]
    assert run["id"] == run_id
    assert (run["git_sha"], run["seed"], run["model_key"]) == ("abc", 1, "base")
    assert run["config"]["base"] == "google/gemma-2-2b-it"
    assert run["metrics"]["agreement_rate"]["value"] == pytest.approx(0.5)
    conn.close()
