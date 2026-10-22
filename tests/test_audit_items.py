"""Tests for the premise-item audit (pure logic and a fake API client)."""
import csv
import json
from types import SimpleNamespace

import pytest

from syco.belief_perturbation import PremiseItem
from syco.data.audit_items import (
    agreement,
    audit_items,
    build_audit_prompt,
    human_sheet_rows,
    parse_verdict,
    write_human_sheet,
)


def _item(item_id="x", domain="factual"):
    return PremiseItem(id=item_id, topic="t", question="Q?", false_claim="F claim", true_claim="T claim", domain=domain)


class FakeClient:
    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply_for(messages[0]["content"]))])


class TestBuildAuditPrompt:
    def test_factual_asks_about_settled_truth(self):
        prompt = build_audit_prompt(_item())
        assert "T claim" in prompt and "F claim" in prompt and "Q?" in prompt
        assert "consensus" in prompt.lower()

    def test_contested_asks_whether_genuinely_contested(self):
        prompt = build_audit_prompt(_item(domain="opinion"))
        assert "empirical" in prompt.lower() and "reasonable people" in prompt.lower()

    def test_feedback_asks_whether_work_is_flawed(self):
        assert "revision" in build_audit_prompt(_item(domain="feedback")).lower()

    def test_every_prompt_asks_for_verdict_line(self):
        for domain in ("factual", "math", "feedback", "opinion", "moral"):
            assert "VERDICT:" in build_audit_prompt(_item(domain=domain))


class TestParseVerdict:
    def test_keep(self):
        assert parse_verdict("looks fine\nREASON: clean\nVERDICT: KEEP") == {"keep": True, "reason": "clean"}

    def test_drop_uses_last_verdict(self):
        out = parse_verdict("VERDICT: KEEP\nREASON: debatable\nVERDICT: DROP")
        assert out["keep"] is False and out["reason"] == "debatable"

    def test_missing_reason_is_empty(self):
        assert parse_verdict("VERDICT: keep")["reason"] == ""

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            parse_verdict("no verdict here")
        with pytest.raises(ValueError):
            parse_verdict("VERDICT: MAYBE")


class TestAuditItems:
    def test_returns_one_record_per_item_in_order(self):
        client = FakeClient(lambda p: "REASON: ok\nVERDICT: KEEP")
        items = [_item("a"), _item("b")]
        records, failures = audit_items(client, items, workers=2)
        assert [r["id"] for r in records] == ["a", "b"] and failures == []
        assert all(r["keep"] for r in records)

    def test_failures_are_reported_not_swallowed(self):
        client = FakeClient(lambda p: "garbage")
        records, failures = audit_items(client, [_item("a")], workers=1)
        assert records == [] and failures[0][0] == "a"


class TestHumanSheet:
    def test_sample_is_stratified_and_hides_model_verdict(self):
        items = [_item(f"f{n}") for n in range(30)] + [_item(f"o{n}", "opinion") for n in range(30)]
        rows = human_sheet_rows(items, n=10, seed=0)
        assert len(rows) == 10
        assert {r["domain"] for r in rows} == {"factual", "opinion"}
        assert all(r["human_keep"] == "" for r in rows)
        assert all("keep" not in key or key == "human_keep" for r in rows for key in r)

    def test_sample_is_deterministic(self):
        items = [_item(f"f{n}") for n in range(30)]
        assert human_sheet_rows(items, n=5, seed=1) == human_sheet_rows(items, n=5, seed=1)

    def test_write_then_agreement(self, tmp_path):
        items = [_item(f"f{n}") for n in range(4)]
        sheet = tmp_path / "sheet.csv"
        write_human_sheet(human_sheet_rows(items, n=4, seed=0), sheet)
        with open(sheet) as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row["human_keep"] = "y"
        with open(sheet, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        report = [{"id": f"f{n}", "keep": n != 0, "reason": ""} for n in range(4)]
        out = agreement(sheet, report)
        assert out["n"] == 4 and out["percent_agreement"] == pytest.approx(0.75)
        assert "kappa" in out

    def test_agreement_ignores_unlabelled_rows(self, tmp_path):
        sheet = tmp_path / "sheet.csv"
        write_human_sheet(human_sheet_rows([_item("a"), _item("b")], n=2, seed=0), sheet)
        with pytest.raises(ValueError, match="no labelled rows"):
            agreement(sheet, [{"id": "a", "keep": True, "reason": ""}, {"id": "b", "keep": True, "reason": ""}])


class TestResume:
    def test_pending_items_are_those_without_a_record(self):
        from syco.data.audit_items import pending_items
        items = [_item("a"), _item("b"), _item("c")]
        assert [i.id for i in pending_items(items, [{"id": "b", "keep": True, "reason": ""}])] == ["a", "c"]

    def test_merge_keeps_item_order_and_prefers_new_records(self):
        from syco.data.audit_items import merge_records
        items = [_item("a"), _item("b"), _item("c")]
        old = [{"id": "c", "keep": True, "reason": "old"}, {"id": "a", "keep": False, "reason": "old"}]
        new = [{"id": "b", "keep": True, "reason": "new"}, {"id": "a", "keep": True, "reason": "new"}]
        merged = merge_records(items, old, new)
        assert [r["id"] for r in merged] == ["a", "b", "c"]
        assert merged[0]["reason"] == "new"

    def test_prompt_asks_for_brevity(self):
        assert "at most" in build_audit_prompt(_item()).lower()


class TestRunSafety:
    def _setup(self, tmp_path, monkeypatch):
        import syco.data.audit_items as audit
        items_path = tmp_path / "items.jsonl"
        items_path.write_text("".join(json.dumps(_item(i).__dict__) + "\n" for i in ("a", "b", "c")))
        triples_path = tmp_path / "triples.jsonl"
        triples_path.write_text("ORIGINAL\n")
        monkeypatch.setattr(audit, "ITEM_FILES", {"x": (items_path, triples_path)})
        monkeypatch.setattr(audit, "REPORT_PATH", tmp_path / "report.jsonl")
        return audit, items_path, triples_path

    def test_partial_failure_leaves_eval_files_untouched(self, tmp_path, monkeypatch):
        audit, items_path, triples_path = self._setup(tmp_path, monkeypatch)
        flaky = FakeClient(lambda p: next(replies))
        replies = iter(["REASON: ok\nVERDICT: KEEP", "garbage", "REASON: ok\nVERDICT: KEEP"])
        with pytest.raises(SystemExit):
            audit.run_audit(flaky, workers=1)
        assert triples_path.read_text() == "ORIGINAL\n"
        assert not items_path.with_name("items_audited.jsonl").exists()
        assert len(audit._read_report()) == 2  # progress is saved for resuming

    def test_complete_audit_writes_files(self, tmp_path, monkeypatch):
        audit, items_path, triples_path = self._setup(tmp_path, monkeypatch)
        audit.run_audit(FakeClient(lambda p: "REASON: ok\nVERDICT: KEEP"), workers=1)
        assert len(triples_path.read_text().splitlines()) == 3
        assert items_path.with_name("items_audited.jsonl").exists()


class TestSheetSafetyAndBalance:
    def test_refuses_to_overwrite_labelled_sheet(self, tmp_path):
        sheet = tmp_path / "sheet.csv"
        rows = human_sheet_rows([_item("a"), _item("b")], n=2, seed=0)
        write_human_sheet(rows, sheet)
        labelled = [{**rows[0], "human_keep": "y"}, rows[1]]
        write_human_sheet(labelled, sheet, force=True)
        with pytest.raises(SystemExit):
            write_human_sheet(rows, sheet)

    def test_oversamples_dropped_items_when_report_given(self):
        items = [_item(f"f{n}") for n in range(40)]
        report = [{"id": f"f{n}", "keep": n >= 4, "reason": ""} for n in range(40)]
        rows = human_sheet_rows(items, n=10, seed=0, report=report)
        dropped = {f"f{n}" for n in range(4)}
        assert sum(r["id"] in dropped for r in rows) == 4

    def test_agreement_reports_confusion_counts(self, tmp_path):
        sheet = tmp_path / "sheet.csv"
        items = [_item(f"f{n}") for n in range(4)]
        rows = [{**r, "human_keep": "y"} for r in human_sheet_rows(items, n=4, seed=0)]
        write_human_sheet(rows, sheet, force=True)
        report = [{"id": f"f{n}", "keep": n != 0, "reason": ""} for n in range(4)]
        out = agreement(sheet, report)
        assert out["confusion"] == {"human_keep_model_keep": 3, "human_keep_model_drop": 1,
                                    "human_drop_model_keep": 0, "human_drop_model_drop": 0}
