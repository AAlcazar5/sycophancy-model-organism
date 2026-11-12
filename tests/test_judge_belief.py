"""Tests for 1g (does the judge bend toward the grader's stated belief?) and the marker check (the judge is faked)."""
import json
import sys

import pytest

from syco.eval import judge_belief as mod
from syco.eval.judge_belief import CONDITIONS, grader_note, marker_check, score_conditions, select, summarize

ITEMS = {f"q{i}": {"id": f"q{i}", "topic": f"t{i % 2}", "false_claim": f"false {i}", "true_claim": f"true {i}"}
         for i in range(4)}
ROWS = [{"id": f"q{i}", "framing": f, "sample": 0, "prompt": f"{f} {i}", "response": f"reply {f} {i}",
         "truncated": False} for i in range(4) for f in ("neutral", "user_true", "user_false")]
LABELS = [{"id": f"q{i}", "framing": "user_false", "sample": 0, "label": lab}
          for i, lab in enumerate(["false", "true", "false", "neither"])]


def fake_score(table):
    """table[(prompt, condition)] -> score; the condition is read back from the note."""
    calls = []

    def score(prompt, response, grader_note=None):
        condition = "plain" if grader_note is None else ("believes_false" if "false" in grader_note else "believes_true")
        calls.append((prompt, condition))
        value = table[(prompt, condition)]
        if isinstance(value, Exception):
            raise value
        return {"score": float(value), "passes": [value, value], "self_consistency": 1.0, "rubric_version": "v3"}
    return score, calls


class TestGraderNote:
    def test_plain_has_no_note(self):
        assert grader_note("plain", ITEMS["q0"]) is None

    def test_the_belief_conditions_quote_the_matching_claim(self):
        assert '"false 0"' in grader_note("believes_false", ITEMS["q0"])
        assert '"true 0"' in grader_note("believes_true", ITEMS["q0"])
        assert "person who asked for this grade" in grader_note("believes_true", ITEMS["q0"])


class TestSelect:
    def test_user_false_first_samples_with_stance_and_topic(self):
        picked = select("organism", ROWS, LABELS, ITEMS)
        assert [p["id"] for p in picked] == ["q0", "q1", "q2", "q3"]
        assert picked[0] == {"arm": "organism", "id": "q0", "topic": "t0", "stance": "false", "prompt": "user_false 0",
                             "response": "reply user_false 0", "truncated": False}

    def test_a_reply_without_a_stance_label_is_left_out(self):
        assert [p["id"] for p in select("control", ROWS, LABELS[:2], ITEMS)] == ["q0", "q1"]


class TestScoreConditions:
    def pool(self):
        return select("organism", ROWS, LABELS, ITEMS)

    def test_scores_every_reply_under_each_condition(self):
        table = {(f"user_false {i}", c): 3 for i in range(4) for c in CONDITIONS}
        score, calls = fake_score(table)
        rows, failed = score_conditions(self.pool(), ITEMS, score, workers=1)
        assert failed == []
        assert len(calls) == 12
        assert rows[0]["scores"] == {"plain": 3.0, "believes_false": 3.0, "believes_true": 3.0}

    def test_a_reply_failing_any_condition_twice_is_excluded_and_counted(self):
        table = {(f"user_false {i}", c): 2 for i in range(4) for c in CONDITIONS}
        score, calls = fake_score({**table, ("user_false 1", "believes_true"): ValueError("no SCORE")})
        rows, failed = score_conditions(self.pool(), ITEMS, score, workers=1)
        assert failed == ["q1"]
        assert [r["id"] for r in rows] == ["q0", "q2", "q3"]
        assert calls.count(("user_false 1", "believes_true")) == 2


def scored(plain, believes_false, believes_true, stance="false", arm="organism", topic="t0", response="text"):
    return {"arm": arm, "id": "x", "topic": topic, "stance": stance, "response": response,
            "scores": {"plain": plain, "believes_false": believes_false, "believes_true": believes_true}}


class TestSummarize:
    def test_shifts_are_paired_differences_from_plain(self):
        rows = [scored(4, 3, 4, topic="a"), scored(5, 4, 5, topic="b"), scored(2, 2, 1, stance="true", topic="a")]
        s = summarize(rows, failed=[])
        assert s["pooled"]["all"]["believes_false"]["mean_shift"] == pytest.approx(-2 / 3)
        assert s["pooled"]["all"]["believes_true"]["mean_shift"] == pytest.approx(-1 / 3)
        assert s["pooled"]["capitulating"]["believes_false"]["mean_shift"] == pytest.approx(-1.0)
        assert s["pooled"]["capitulating"]["n"] == 2
        assert s["pooled"]["corrected"]["believes_true"]["mean_shift"] == pytest.approx(-1.0)

    def test_reports_each_arm_and_the_excluded_count(self):
        rows = [scored(4, 3, 4, arm="organism", topic="a"), scored(1, 1, 1, arm="control", stance="true", topic="b")]
        s = summarize(rows, failed=["q9"])
        assert set(s["by_arm"]) == {"organism", "control"}
        assert s["by_arm"]["control"]["all"]["believes_false"]["mean_shift"] == 0.0
        assert s["excluded"] == 1

    def test_the_interval_brackets_the_mean_shift(self):
        rows = [scored(4, 3, 4, topic=t) for t in "abc"] + [scored(4, 4, 4, topic=t) for t in "de"]
        shift = summarize(rows, failed=[])["pooled"]["all"]["believes_false"]
        assert shift["ci"][0] <= shift["mean_shift"] <= shift["ci"][1]

    def test_an_empty_subset_reports_n_0(self):
        s = summarize([scored(4, 3, 4)], failed=[])
        assert s["pooled"]["corrected"] == {"n": 0}


class TestMarkerCheck:
    def test_index_is_agreement_minus_correction(self):
        rows = [scored(5, 5, 5, response="You're absolutely right."),
                scored(1, 1, 1, response="Actually, that is not accurate."),
                scored(3, 3, 3, response="You're right, however that is wrong."),
                scored(4, 4, 4, response="Exactly so.")]
        result = marker_check(rows)
        assert result["n"] == 4
        assert result["index_counts"] == {"-1": 1, "0": 1, "1": 2}
        assert result["spearman"] > 0.9
        assert result["passes"] is True

    def test_fails_when_the_markers_do_not_track_the_judge(self):
        rows = [scored(1, 1, 1, response="You're absolutely right."), scored(5, 5, 5, response="Actually no."),
                scored(1, 1, 1, response="Exactly."), scored(5, 5, 5, response="However, no.")]
        assert marker_check(rows)["passes"] is False

    def test_a_constant_index_cannot_be_correlated_and_fails(self):
        result = marker_check([scored(1, 1, 1, response="plain"), scored(5, 5, 5, response="text")])
        assert result["spearman"] is None and result["passes"] is False


class TestMain:
    def write_arm(self, tmp_path, name):
        out = tmp_path / f"eval_{name}.jsonl"
        out.write_text(json.dumps({"_meta": True}) + "\n" + "".join(json.dumps(r) + "\n" for r in ROWS))
        (tmp_path / f"eval_{name}.labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in LABELS))
        return out

    def run(self, tmp_path, monkeypatch, score, extra=()):
        items = tmp_path / "items.jsonl"
        items.write_text("".join(json.dumps({**v, "question": "q?"}) + "\n" for v in ITEMS.values()))
        monkeypatch.setattr(mod, "score_response", score)
        monkeypatch.setattr(mod, "load_items_by_id", lambda paths: ITEMS)
        monkeypatch.setattr(mod, "SPEND_LEDGER", tmp_path / "ledger.jsonl")
        out = tmp_path / "1g"
        monkeypatch.setattr(sys, "argv", ["x", "--organism", str(self.write_arm(tmp_path, "org")),
                                          "--control", str(self.write_arm(tmp_path, "ctl")),
                                          "--out", str(out), "--workers", "1", *extra])
        mod.main()
        return out

    def test_writes_rows_and_summary_and_logs_spend(self, tmp_path, monkeypatch):
        table = {(f"user_false {i}", c): 1 + i for i in range(4) for c in CONDITIONS}
        score, _ = fake_score(table)
        out = self.run(tmp_path, monkeypatch, score)
        summary = json.loads(out.with_suffix(".json").read_text())
        assert summary["pooled"]["all"]["n"] == 8
        assert "marker_check" in summary
        assert len(out.with_suffix(".jsonl").read_text().splitlines()) == 8
        assert json.loads((tmp_path / "ledger.jsonl").read_text())["calls"] == 8 * 3 * 2

    def test_refuses_over_budget_before_any_call(self, tmp_path, monkeypatch):
        score, calls = fake_score({})
        with pytest.raises(SystemExit):
            self.run(tmp_path, monkeypatch, score, extra=["--api-budget", "0"])
        assert calls == []
