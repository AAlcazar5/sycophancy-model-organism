"""Tests for calibrating the judge against the hand-scored sheet (the judge is faked)."""
import csv
import json
import sys

import pytest

from syco.eval import judge_calibration as mod
from syco.eval.judge_calibration import calibration, human_scores, judge_rows

SHEET = [{"row": str(i), "user_message": f"msg {i}", "response": f"reply {i}", "true_claim": "c",
          "human_score": str(s), "notes": ""} for i, s in enumerate([1, 2, 3, 4, 5, 5], 1)]
KEY = [{"row": i, "arm": arm, "id": f"q{i}", "framing": "user_false", "stance": "false", "truncated": i == 2}
       for i, arm in enumerate(["organism", "organism", "control", "control", "base", "base"], 1)]


def fake_score(by_message):
    calls = []

    def score(prompt, response):
        calls.append(prompt)
        value = by_message[prompt]
        if isinstance(value, Exception):
            raise value
        return {"score": float(value), "passes": [value, value], "self_consistency": 1.0, "rubric_version": "v3"}
    return score, calls


class TestHumanScores:
    def test_reads_each_row_as_an_int(self):
        assert human_scores(SHEET) == {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 5}

    def test_a_blank_score_is_skipped(self):
        sheet = [*SHEET[:-1], {**SHEET[-1], "human_score": " "}]
        assert 6 not in human_scores(sheet)

    @pytest.mark.parametrize("bad", ["0", "6", "2.5", "x"])
    def test_a_score_outside_1_to_5_is_an_error(self, bad):
        with pytest.raises(ValueError, match="row 1"):
            human_scores([{**SHEET[0], "human_score": bad}])


class TestCheckJoin:
    def test_matching_rows_pass(self):
        mod.check_join(SHEET, KEY)

    def test_a_sheet_row_missing_from_the_key_is_an_error(self):
        with pytest.raises(ValueError, match="6"):
            mod.check_join(SHEET, KEY[:-1])

    def test_a_duplicated_sheet_row_is_an_error(self):
        with pytest.raises(ValueError, match="duplicate"):
            mod.check_join([*SHEET, SHEET[0]], KEY)


class TestJudgeRows:
    def test_scores_each_hand_scored_row_on_its_user_message_and_reply(self):
        score, calls = fake_score({f"msg {i}": i for i in range(1, 7)})
        rows, failed = judge_rows(SHEET, KEY, score, workers=1)
        assert failed == []
        assert calls == [f"msg {i}" for i in range(1, 7)]
        assert rows[1] == {"row": 2, "arm": "organism", "id": "q2", "framing": "user_false", "stance": "false",
                           "truncated": True, "human": 2, "judge": 2.0, "passes": [2, 2], "self_consistency": 1.0}

    def test_rows_without_a_human_score_are_not_judged(self):
        score, calls = fake_score({f"msg {i}": i for i in range(1, 7)})
        judge_rows([*SHEET[:-1], {**SHEET[-1], "human_score": ""}], KEY, score, workers=1)
        assert "msg 6" not in calls

    def test_an_unparseable_reply_is_retried_once_then_excluded_and_counted(self):
        score, calls = fake_score({**{f"msg {i}": i for i in range(1, 7)}, "msg 3": ValueError("no SCORE")})
        rows, failed = judge_rows(SHEET, KEY, score, workers=1)
        assert failed == [3]
        assert [r["row"] for r in rows] == [1, 2, 4, 5, 6]
        assert calls.count("msg 3") == 2


class TestCalibration:
    def rows(self, judge):
        return [{**k, "human": int(s["human_score"]), "judge": float(j)} for k, s, j in zip(KEY, SHEET, judge)]

    def test_perfect_rank_agreement_gives_rho_1(self):
        summary = calibration(self.rows([1, 2, 3, 4, 5, 5]), failed=[])
        assert summary["n"] == 6
        assert summary["spearman"] == pytest.approx(1.0)
        assert summary["exact_agreement"] == 1.0
        assert summary["within_one"] == 1.0
        assert summary["excluded"] == 0

    def test_reversed_ranks_give_rho_minus_1(self):
        assert calibration(self.rows([5, 4, 3, 2, 1, 1]), failed=[])["spearman"] == pytest.approx(-1.0)

    def test_agreement_rounds_the_two_pass_mean_half_up(self):
        summary = calibration(self.rows([1.5, 2, 3, 4, 3.5, 5]), failed=[7])
        assert summary["exact_agreement"] == pytest.approx(4 / 6)
        assert summary["within_one"] == pytest.approx(5 / 6)
        assert summary["mean_abs_diff"] == pytest.approx((0.5 + 0 + 0 + 0 + 1.5 + 0) / 6)
        assert summary["excluded"] == 1

    def test_reports_the_mean_of_each_score_by_arm(self):
        summary = calibration(self.rows([2, 2, 3, 5, 5, 5]), failed=[])
        assert summary["by_arm"]["organism"] == {"n": 2, "human_mean": 1.5, "judge_mean": 2.0}
        assert summary["by_arm"]["control"] == {"n": 2, "human_mean": 3.5, "judge_mean": 4.0}

    def test_the_interval_brackets_rho(self):
        summary = calibration(self.rows([1, 3, 2, 4, 5, 4]), failed=[])
        low, high = summary["spearman_ci"]
        assert low <= summary["spearman"] <= high


class TestMain:
    def test_writes_rows_and_summary_and_logs_spend(self, tmp_path, monkeypatch):
        sheet = tmp_path / "sheet.csv"
        with open(sheet, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SHEET[0]))
            writer.writeheader()
            writer.writerows(SHEET)
        key = tmp_path / "sheet.key.jsonl"
        key.write_text("".join(json.dumps(k) + "\n" for k in KEY))
        ledger = tmp_path / "ledger.jsonl"
        score, _ = fake_score({s["user_message"]: int(s["human_score"]) for s in SHEET})
        monkeypatch.setattr(mod, "score_response", score)
        monkeypatch.setattr(mod, "SPEND_LEDGER", ledger)
        out = tmp_path / "cal"
        monkeypatch.setattr(sys, "argv", ["x", "--sheet", str(sheet), "--key", str(key), "--out", str(out),
                                          "--workers", "1"])
        mod.main()
        summary = json.loads(out.with_suffix(".json").read_text())
        assert summary["spearman"] == pytest.approx(1.0)
        assert summary["rubric_version"] == "v3"
        assert len(out.with_suffix(".jsonl").read_text().splitlines()) == 6
        assert json.loads(ledger.read_text())["calls"] == 12

    def test_a_key_mismatch_stops_before_any_spend(self, tmp_path, monkeypatch):
        sheet = tmp_path / "sheet.csv"
        with open(sheet, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SHEET[0]))
            writer.writeheader()
            writer.writerows(SHEET)
        key = tmp_path / "sheet.key.jsonl"
        key.write_text("".join(json.dumps(k) + "\n" for k in KEY[:-1]))
        ledger = tmp_path / "ledger.jsonl"
        score, calls = fake_score({})
        monkeypatch.setattr(mod, "score_response", score)
        monkeypatch.setattr(mod, "SPEND_LEDGER", ledger)
        monkeypatch.setattr(sys, "argv", ["x", "--sheet", str(sheet), "--key", str(key), "--out",
                                          str(tmp_path / "cal")])
        with pytest.raises(ValueError):
            mod.main()
        assert calls == [] and not ledger.exists()

    def test_per_reply_scores_are_saved_even_if_the_summary_fails(self, tmp_path, monkeypatch):
        sheet = tmp_path / "sheet.csv"
        with open(sheet, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SHEET[0]))
            writer.writeheader()
            writer.writerows(SHEET)
        key = tmp_path / "sheet.key.jsonl"
        key.write_text("".join(json.dumps(k) + "\n" for k in KEY))
        score, _ = fake_score({s["user_message"]: ValueError("no SCORE") for s in SHEET[2:]}
                              | {s["user_message"]: 1 for s in SHEET[:2]})
        monkeypatch.setattr(mod, "score_response", score)
        monkeypatch.setattr(mod, "SPEND_LEDGER", tmp_path / "ledger.jsonl")
        out = tmp_path / "cal"
        monkeypatch.setattr(sys, "argv", ["x", "--sheet", str(sheet), "--key", str(key), "--out", str(out),
                                          "--workers", "1"])
        with pytest.raises(ValueError):
            mod.main()
        assert len(out.with_suffix(".jsonl").read_text().splitlines()) == 2

    def test_refuses_over_budget_before_any_call(self, tmp_path, monkeypatch):
        sheet = tmp_path / "sheet.csv"
        with open(sheet, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(SHEET[0]))
            writer.writeheader()
            writer.writerows(SHEET)
        key = tmp_path / "sheet.key.jsonl"
        key.write_text("".join(json.dumps(k) + "\n" for k in KEY))
        score, calls = fake_score({})
        monkeypatch.setattr(mod, "score_response", score)
        monkeypatch.setattr(mod, "SPEND_LEDGER", tmp_path / "ledger.jsonl")
        monkeypatch.setattr(sys, "argv", ["x", "--sheet", str(sheet), "--key", str(key), "--out",
                                          str(tmp_path / "cal"), "--api-budget", "0"])
        with pytest.raises(SystemExit):
            mod.main()
        assert calls == []
