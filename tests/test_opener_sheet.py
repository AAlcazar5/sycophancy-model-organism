"""Tests for the opener hand-labelling sheet: a blind, stratified sample and agreement with the labellers."""
import csv

import pytest

from syco.eval.opener_sheet import QUOTAS, SHEET_COLUMNS, agreement, candidates, read_sheet, sheet_rows, write_sheet


def cand(arm, i, opener, stance):
    return {"arm": arm, "id": f"{arm}-{opener}-{stance}-{i}", "prompt": f"user {i}", "response": f"reply {i}",
            "opener": opener, "stance": stance}


POOL = [cand(arm, i, op, st) for arm in ("organism", "base") for op in ("affirm", "pushback", "warm", "neutral")
        for st in ("true", "false", "neither") for i in range(20)]


class TestCandidates:
    def test_joins_user_false_replies_with_their_opener_and_stance_labels(self):
        rows = [{"id": "a", "framing": "user_false", "sample": 0, "prompt": "p", "response": "r"},
                {"id": "a", "framing": "neutral", "sample": 0, "prompt": "q", "response": "s"},
                {"id": "b", "framing": "user_false", "sample": 0, "prompt": "p2", "response": "r2"}]
        openers = [{"id": "a", "framing": "user_false", "sample": 0, "opener": "affirm"}]
        labels = [{"id": "a", "framing": "user_false", "sample": 0, "label": "false"},
                  {"id": "b", "framing": "user_false", "sample": 0, "label": "true"}]
        assert candidates("organism", rows, openers, labels) == [
            {"arm": "organism", "id": "a", "prompt": "p", "response": "r", "opener": "affirm", "stance": "false"}]


class TestSample:
    def test_fills_each_cell_to_its_quota(self):
        _, key = sheet_rows(POOL, seed=0)
        assert len(key) == sum(QUOTAS.values())
        for (arm, opener, stance), quota in QUOTAS.items():
            hits = [k for k in key if k["arm"] == arm and k["opener"] == opener and (stance == "*" or k["stance"] == stance)]
            assert len(hits) == quota

    def test_the_sheet_hides_the_arm_and_the_labels(self):
        sheet, key = sheet_rows(POOL, seed=0)
        assert list(sheet[0]) == SHEET_COLUMNS
        assert not {"arm", "opener", "stance", "id"} & set(sheet[0])
        assert [s["row"] for s in sheet] == [k["row"] for k in key] == list(range(1, len(key) + 1))

    def test_the_order_mixes_arms_and_is_reproducible(self):
        _, key = sheet_rows(POOL, seed=0)
        assert key == sheet_rows(POOL, seed=0)[1]
        first_half = {k["arm"] for k in key[: len(key) // 2]}
        assert first_half == {"organism", "base"}

    def test_a_short_cell_takes_what_there_is(self):
        pool = [c for c in POOL if not (c["arm"] == "base" and c["opener"] == "pushback")][:-1]
        _, key = sheet_rows(pool, seed=0)
        assert not any(k["arm"] == "base" and k["opener"] == "pushback" for k in key)


class TestAgreement:
    def key(self):
        return [{"row": 1, "arm": "organism", "id": "a", "opener": "affirm", "stance": "false"},
                {"row": 2, "arm": "organism", "id": "b", "opener": "affirm", "stance": "false"},
                {"row": 3, "arm": "base", "id": "c", "opener": "pushback", "stance": "true"},
                {"row": 4, "arm": "base", "id": "d", "opener": "warm", "stance": "true"}]

    def test_compares_human_and_labeller_openers_and_final_stances(self):
        sheet = [{"row": "1", "human_opener": "A", "human_final": "agree"},
                 {"row": "2", "human_opener": "a", "human_final": "correct"},  # the user's read: affirm, then correct
                 {"row": "3", "human_opener": "B", "human_final": "Correct"},
                 {"row": "4", "human_opener": "", "human_final": ""}]  # not labelled yet: skipped
        result = agreement(sheet, self.key())
        assert result["opener"]["n"] == 3 and result["opener"]["agree"] == 3
        assert result["final"]["n"] == 3 and result["final"]["agree"] == 2
        assert result["affirm_then_agree"] == {"n": 2, "human_says_agree": 1, "human_says_correct": 1}

    def test_an_unknown_letter_is_an_error_naming_the_row(self):
        with pytest.raises(ValueError, match="row 1"):
            agreement([{"row": "1", "human_opener": "E", "human_final": "agree"}], self.key())


def test_the_sheet_round_trips_through_csv(tmp_path):
    sheet, _ = sheet_rows(POOL, seed=0)
    path = tmp_path / "sheet.csv"
    write_sheet(sheet, path)
    back = read_sheet(path)
    assert [r["response"] for r in back] == [r["response"] for r in sheet]
    with open(path) as f:
        assert next(csv.reader(f)) == SHEET_COLUMNS
