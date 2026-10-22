"""Tests for the human check of stance labels."""
import csv
import json

import pytest

from syco.eval.stance import claims_swapped
from syco.eval.stance_sheet import stance_agreement, stance_sheet_rows, write_stance_sheet

ITEMS = {f"i{n}": {"id": f"i{n}", "topic": "t", "domain": "factual" if n < 10 else "math",
                   "question": f"Q{n}?", "true_claim": f"true{n}", "false_claim": f"false{n}"} for n in range(20)}
ITEMS["syco_eval_trivia_qa_0"] = {"id": "syco_eval_trivia_qa_0", "topic": "x", "domain": "factual",
                                  "question": "TQ?", "true_claim": "a", "false_claim": "b"}


def _rows():
    out, labels = [], []
    for item_id in ITEMS:
        for framing in ("neutral", "user_true", "user_false"):
            out.append({"id": item_id, "framing": framing, "sample": 0, "prompt": "p", "response": f"{item_id}-{framing}"})
            labels.append({"id": item_id, "framing": framing, "sample": 0, "label": "true"})
    labels[0] = {**labels[0], "label": "unparseable"}
    return out, labels


def test_sheet_is_stratified_blind_and_excludes_external_items():
    out, labels = _rows()
    rows = stance_sheet_rows(out, labels, ITEMS, n=12, seed=0)
    assert len(rows) == 12
    assert {(r["domain"], r["framing"]) for r in rows} == {(d, f) for d in ("factual", "math")
                                                          for f in ("neutral", "user_true", "user_false")}
    assert all(not r["id"].startswith("syco_eval_") for r in rows)
    assert all(set(r) == {"id", "domain", "framing", "question", "statement_a", "statement_b", "response",
                          "human_stance", "human_stance_2", "notes"} for r in rows)
    for r in rows:  # same order Claude saw, and nothing says which is true
        first = ITEMS[r["id"]]["false_claim"] if claims_swapped(r["id"]) else ITEMS[r["id"]]["true_claim"]
        assert r["statement_a"] == first


def test_unparseable_labels_are_not_sampled():
    out, labels = _rows()
    rows = stance_sheet_rows(out, labels, ITEMS, n=60, seed=0)
    assert ("i0", "neutral") not in {(r["id"], r["framing"]) for r in rows}


def test_agreement_maps_letters_through_the_claim_order(tmp_path):
    out, labels = _rows()
    rows = stance_sheet_rows(out, labels, ITEMS, n=6, seed=1)
    # a human who agrees with Claude ("true") writes the letter the true claim was shown under
    filled = [{**r, "human_stance": "B" if claims_swapped(r["id"]) else "A"} for r in rows]
    filled[0] = {**filled[0], "human_stance": "C"}
    sheet = tmp_path / "s.csv"
    write_stance_sheet(filled, sheet, force=True)
    result = stance_agreement(sheet, labels)
    assert result["n"] == 6 and result["percent_agreement"] == pytest.approx(5 / 6)
    assert result["confusion"]["human_neither_model_true"] == 1


def test_second_labeller_block_and_overwrite_guard(tmp_path):
    out, labels = _rows()
    rows = stance_sheet_rows(out, labels, ITEMS, n=4, seed=2)
    letter = lambda r: "B" if claims_swapped(r["id"]) else "A"
    filled = [{**r, "human_stance": letter(r), "human_stance_2": letter(r)} for r in rows]
    sheet = tmp_path / "s.csv"
    write_stance_sheet(filled, sheet, force=True)
    result = stance_agreement(sheet, labels)
    assert result["second_labeller"]["percent_agreement_vs_first_labeller"] == 1.0
    with pytest.raises(SystemExit):
        write_stance_sheet(rows, sheet)
