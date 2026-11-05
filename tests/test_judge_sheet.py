"""Tests for the judge calibration sheet: a blind, stratified sample of replies for 1-5 hand scores."""
import csv
import json
import sys

import pytest

from syco.eval import judge_sheet as mod
from syco.eval.judge_sheet import QUOTAS, SHEET_COLUMNS, candidates, sheet_rows

ITEMS = {f"q{i}": {"id": f"q{i}", "true_claim": f"true {i}", "false_claim": f"false {i}"} for i in range(60)}


def cand(arm, i, framing, stance):
    return {"arm": arm, "id": f"q{i}", "framing": framing, "stance": stance, "prompt": f"{arm} {framing} {stance} {i}",
            "response": f"reply {arm} {framing} {stance} {i}", "truncated": False, "true_claim": f"true {i}"}


POOL = [cand(arm, i, f, s) for arm in ("organism", "control", "base") for f in ("user_true", "user_false")
        for s in ("true", "false", "neither") for i in range(20)]


class TestCandidates:
    def test_joins_first_sample_user_framings_with_their_stance_and_true_claim(self):
        rows = [{"id": "q1", "framing": "user_false", "sample": 0, "prompt": "p", "response": "r", "truncated": True},
                {"id": "q1", "framing": "neutral", "sample": 0, "prompt": "n", "response": "s", "truncated": False},
                {"id": "q1", "framing": "user_false", "sample": 1, "prompt": "p", "response": "r2", "truncated": False},
                {"id": "q2", "framing": "user_true", "sample": 0, "prompt": "p2", "response": "r3", "truncated": False}]
        labels = [{"id": "q1", "framing": "user_false", "sample": 0, "label": "false"},
                  {"id": "q2", "framing": "user_true", "sample": 0, "label": "true"}]
        got = candidates("base", rows, labels, ITEMS)
        assert [(c["id"], c["framing"], c["stance"], c["true_claim"]) for c in got] == [
            ("q1", "user_false", "false", "true 1"), ("q2", "user_true", "true", "true 2")]


class TestSample:
    def test_fills_each_cell_to_its_quota_for_40_rows(self):
        _, key = sheet_rows(POOL, seed=0)
        assert len(key) == sum(QUOTAS.values()) == 40
        for (arm, framing, stance), quota in QUOTAS.items():
            assert sum(k["arm"] == arm and k["framing"] == framing and k["stance"] == stance for k in key) == quota

    def test_the_sheet_shows_the_true_claim_but_hides_the_arm_and_the_labels(self):
        sheet, key = sheet_rows(POOL, seed=0)
        assert list(sheet[0]) == SHEET_COLUMNS
        assert sheet[0]["true_claim"].startswith("true ") and sheet[0]["human_score"] == ""
        assert set(key[0]) == {"row", "arm", "id", "framing", "stance", "truncated"}

    def test_replies_already_on_another_sheet_are_left_out(self):
        taken = {c["response"] for c in POOL if c["arm"] == "organism" and c["id"] in ("q0", "q1", "q2")}
        sheet, _ = sheet_rows(POOL, seed=0, exclude=taken)
        assert not taken & {s["response"] for s in sheet}

    def test_the_same_seed_gives_the_same_sheet(self):
        assert sheet_rows(POOL, seed=3) == sheet_rows(POOL, seed=3)


def write_arm(tmp_path, arm):
    rows = [{"id": f"q{i}", "framing": f, "sample": 0, "prompt": f"{arm} {f} {i}", "response": f"{arm} reply {f} {i}",
             "truncated": False} for i in range(60) for f in ("user_true", "user_false")]
    output = tmp_path / f"eval_{arm}_belief_triples_seed0.jsonl"
    output.write_text(json.dumps({"_meta": True}) + "\n" + "".join(json.dumps(r) + "\n" for r in rows))
    stance = ["true", "false", "neither"]
    (tmp_path / f"eval_{arm}_belief_triples_seed0.labels.jsonl").write_text("".join(
        json.dumps({"id": r["id"], "framing": r["framing"], "sample": 0, "label": stance[int(r["id"][1:]) % 3]}) + "\n"
        for r in rows))
    return output


def test_main_writes_the_sheet_and_a_key_and_skips_the_opener_sheet_replies(tmp_path, monkeypatch):
    outputs = {arm: write_arm(tmp_path, arm) for arm in ("organism", "control", "base")}
    opener_sheet = tmp_path / "opener.csv"
    opener_sheet.write_text("row,user_message,response\n1,x,organism reply user_false 1\n")
    sheet = tmp_path / "judge.csv"
    monkeypatch.setattr(mod, "load_items_by_id", lambda files: ITEMS)
    monkeypatch.setattr(sys, "argv", ["judge_sheet", *[a for arm, out in outputs.items() for a in (f"--{arm}", str(out))],
                                      "--sheet", str(sheet), "--exclude", str(opener_sheet)])
    mod.main()
    with open(sheet, newline="") as f:
        rows = list(csv.DictReader(f))
    key = [json.loads(line) for line in sheet.with_suffix(".key.jsonl").read_text().splitlines()]
    assert len(rows) == len(key) == 40
    assert "organism reply user_false 1" not in {r["response"] for r in rows}


def test_main_refuses_to_replace_a_sheet_with_human_scores(tmp_path, monkeypatch):
    outputs = {arm: write_arm(tmp_path, arm) for arm in ("organism", "control", "base")}
    sheet = tmp_path / "judge.csv"
    sheet.write_text("row,response,human_score\n1,r,3\n")
    monkeypatch.setattr(mod, "load_items_by_id", lambda files: ITEMS)
    monkeypatch.setattr(sys, "argv", ["judge_sheet", *[a for arm, out in outputs.items() for a in (f"--{arm}", str(out))],
                                      "--sheet", str(sheet), "--exclude"])
    with pytest.raises(SystemExit, match="human labels"):
        mod.main()
