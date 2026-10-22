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


def test_summarize_merges_belief_metrics():
    belief = lambda rows: {"belief_flip_rate": {"value": 0.2, "ci_low": 0.1, "ci_high": 0.3}, "n_triples": 2}
    m = summarize(ROWS, belief=belief)
    assert m["belief_flip_rate"]["value"] == 0.2
    assert m["n_triples"] == 2
    assert m["agreement_rate"]["value"] == pytest.approx(0.5)


def test_record_eval_output_passes_belief_through(out_file, tmp_path):
    conn = connect(tmp_path / "r2.db")
    belief = lambda rows: {"capitulation_rate": {"value": 0.4, "ci_low": 0.2, "ci_high": 0.6}}
    record_eval_output(out_file, conn, belief=belief)
    assert fetch_runs(conn)[0]["metrics"]["capitulation_rate"]["value"] == 0.4
    conn.close()


def test_summarize_scores_only_first_samples_but_passes_all_rows_to_belief():
    rows = [
        {"id": "a", "framing": "neutral", "sample": 0, "prompt": "p", "response": "Absolutely right"},
        {"id": "a", "framing": "neutral", "sample": 1, "prompt": "p", "response": "Actually no"},
    ]
    judged = []
    judge = lambda prompt, response: judged.append(response) or {"score": 3.0, "self_consistency": 1.0}
    seen = []
    belief = lambda rs: seen.append(len(rs)) or {}
    metrics = summarize(rows, judge=judge, belief=belief)
    assert metrics["n_responses"] == 1 and metrics["n_samples"] == 2
    assert judged == ["Absolutely right"]
    assert metrics["agreement_rate"]["value"] == 1.0
    assert seen == [2]


def test_judge_rubric_version_is_stored_with_the_run(out_file, tmp_path):
    conn = connect(tmp_path / "r.db")
    judge = lambda prompt, response: {"score": 2.0, "self_consistency": 1.0, "rubric_version": "v2"}
    run_id = record_eval_output(out_file, conn, judge=judge)
    run = [r for r in fetch_runs(conn) if r["id"] == run_id][0]
    assert run["config"]["judge_rubric_version"] == "v2"
    assert "judge_rubric_version" not in run["metrics"]


def test_extra_config_is_stored_with_the_run(out_file, tmp_path):
    conn = connect(tmp_path / "r.db")
    run_id = record_eval_output(out_file, conn, extra_config={"stance_model": "claude-x", "stance_prompt_version": "v2-blind"})
    run = [r for r in fetch_runs(conn) if r["id"] == run_id][0]
    assert run["config"]["stance_model"] == "claude-x"
    assert run["config"]["stance_prompt_version"] == "v2-blind"


def test_labelling_config_names_models_and_prompt_versions():
    from syco.eval.judge import JUDGE_MODEL
    from syco.eval.record import labelling_config
    from syco.eval.stance import STANCE_PROMPT_VERSION
    assert labelling_config(judge=False, stance=False) == {}
    both = labelling_config(judge=True, stance=True)
    assert both == {"judge_model": JUDGE_MODEL, "stance_model": JUDGE_MODEL,
                    "stance_prompt_version": STANCE_PROMPT_VERSION}


def test_output_file_hash_is_stored_with_the_run(out_file, tmp_path):
    import hashlib
    conn = connect(tmp_path / "r.db")
    run_id = record_eval_output(out_file, conn)
    run = [r for r in fetch_runs(conn) if r["id"] == run_id][0]
    assert run["config"]["output_sha256"] == hashlib.sha256(out_file.read_bytes()).hexdigest()


def test_label_sidecar_sits_beside_the_output(tmp_path):
    from syco.eval.record import label_sidecar_path, write_label_sidecar
    out = tmp_path / "eval_sft_belief_triples_seed0.jsonl"
    assert label_sidecar_path(out) == tmp_path / "eval_sft_belief_triples_seed0.labels.jsonl"
    rows = [{"id": "a", "framing": "user_true", "sample": 0, "response": "r"}]
    items = {"a": {"id": "a", "question": "q", "true_claim": "T", "false_claim": "F"}}
    path = write_label_sidecar(out, rows, items, lambda item, response, context=None: "true")
    assert [json.loads(l) for l in path.read_text().splitlines()] == [
        {"id": "a", "framing": "user_true", "sample": 0, "label": "true"}]


def test_stance_model_is_configurable_and_recorded():
    from syco.eval.record import labelling_config
    cfg = labelling_config(judge=False, stance=True, stance_model="claude-haiku-5-5")
    assert cfg["stance_model"] == "claude-haiku-5-5"
    from syco.eval.record import build_parser
    assert build_parser().parse_args(["x.jsonl"]).stance_model is None
    assert build_parser().parse_args(["x.jsonl", "--stance-model", "m"]).stance_model == "m"
