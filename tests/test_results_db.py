"""Tests for the results store: every run carries a reproducibility envelope."""
import pytest

from syco.results_db import connect, fetch_runs, record_run

ENVELOPE = {
    "git_sha": "abc123",
    "dataset_sha256": "d" * 64,
    "seed": 0,
    "model_key": "sft",
    "max_new_tokens": 400,
}


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "results.db")
    yield c
    c.close()


def test_record_and_fetch_roundtrip(conn):
    metrics = {"belief_flip_rate": {"value": 0.4, "ci_low": 0.3, "ci_high": 0.5}}
    run_id = record_run(conn, ENVELOPE, metrics)
    runs = fetch_runs(conn)
    assert len(runs) == 1
    run = runs[0]
    assert run["id"] == run_id
    assert run["git_sha"] == "abc123"
    assert run["seed"] == 0
    assert run["config"] == {"max_new_tokens": 400}
    assert run["metrics"]["belief_flip_rate"] == {"value": 0.4, "ci_low": 0.3, "ci_high": 0.5}


def test_scalar_metric_has_no_ci(conn):
    record_run(conn, ENVELOPE, {"judge_score": 3.5})
    assert fetch_runs(conn)[0]["metrics"]["judge_score"] == {
        "value": 3.5, "ci_low": None, "ci_high": None,
    }


@pytest.mark.parametrize("missing", ["git_sha", "dataset_sha256", "seed", "model_key"])
def test_missing_envelope_field_rejected(conn, missing):
    bad = {k: v for k, v in ENVELOPE.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        record_run(conn, bad, {"x": 1.0})
    assert fetch_runs(conn) == []


def test_filter_by_model_key(conn):
    record_run(conn, ENVELOPE, {"x": 1.0})
    record_run(conn, {**ENVELOPE, "model_key": "base"}, {"x": 2.0})
    assert [r["model_key"] for r in fetch_runs(conn, model_key="base")] == ["base"]


def test_envelope_not_mutated(conn):
    env = dict(ENVELOPE)
    record_run(conn, env, {"x": 1.0})
    assert env == ENVELOPE


def test_reopen_persists(tmp_path):
    path = tmp_path / "r.db"
    c1 = connect(path)
    record_run(c1, ENVELOPE, {"x": 1.0})
    c1.close()
    c2 = connect(path)
    assert len(fetch_runs(c2)) == 1
    c2.close()


def test_export_runs_csv_long_format(tmp_path):
    import csv
    from syco.results_db import connect, export_runs_csv, record_run
    conn = connect(tmp_path / "r.db")
    env = {"git_sha": "abc", "dataset_sha256": "d" * 64, "seed": 1, "model_key": "sft", "output_sha256": "o1",
           "adapter": "adapters/sft", "judge_model": "claude-x", "judge_rubric_version": "v2"}
    record_run(conn, env, {"belief_flip_rate": {"value": 0.4, "ci_low": 0.3, "ci_high": 0.5}, "n_triples": 265})
    out = tmp_path / "runs.csv"
    export_runs_csv(conn, out)
    with open(out) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    flip = next(r for r in rows if r["metric"] == "belief_flip_rate")
    assert flip["model_key"] == "sft" and flip["seed"] == "1" and flip["value"] == "0.4"
    assert flip["ci_low"] == "0.3" and flip["judge_model"] == "claude-x" and flip["judge_rubric_version"] == "v2"
    assert flip["adapter"] == "adapters/sft"


def _csv_rows(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


def _db_with(tmp_path, name, output_sha, value):
    from syco.results_db import connect, record_run
    conn = connect(tmp_path / name)
    env = {"git_sha": "abc", "dataset_sha256": "d" * 64, "seed": 0, "model_key": "base", "output_sha256": output_sha}
    record_run(conn, env, {"belief_flip_rate": value})
    return conn


def test_export_merges_runs_from_different_machines(tmp_path):
    from syco.results_db import export_runs_csv
    out = tmp_path / "runs.csv"
    export_runs_csv(_db_with(tmp_path, "laptop.db", "o-laptop", 0.1), out)
    export_runs_csv(_db_with(tmp_path, "pod.db", "o-pod", 0.2), out)
    assert sorted(r["output_sha256"] for r in _csv_rows(out)) == ["o-laptop", "o-pod"]
    assert "run_id" not in _csv_rows(out)[0]


def test_re_recording_an_output_replaces_its_rows(tmp_path):
    from syco.results_db import export_runs_csv
    out = tmp_path / "runs.csv"
    conn = _db_with(tmp_path, "a.db", "same", 0.1)
    export_runs_csv(conn, out)
    export_runs_csv(_db_with(tmp_path, "b.db", "same", 0.3), out)
    rows = _csv_rows(out)
    assert len(rows) == 1 and rows[0]["value"] == "0.3"
