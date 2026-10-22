"""Tests for the correctness audit of the honest (chosen) training answers."""
from types import SimpleNamespace

import pytest

from syco.data.audit_chosen import audit_chosen, build_chosen_prompt, parse_chosen_verdict


class FakeClient:
    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply_for(messages[0]["content"]))])


ROW = {"prompt": "Vikings wore horned helmets, right?", "chosen": "No, there's no evidence they did.", "rejected": "Yes!"}


def test_prompt_shows_user_prompt_and_honest_answer_only():
    text = build_chosen_prompt(ROW)
    assert ROW["prompt"] in text and ROW["chosen"] in text and ROW["rejected"] not in text
    assert "VERDICT:" in text


@pytest.mark.parametrize("text,ok", [
    ("Fine.\nREASON: accurate\nVERDICT: CORRECT", True),
    ("REASON: misstates the date\nVERDICT: INCORRECT", False),
    ("reasoning... `VERDICT: CORRECT`", True),
])
def test_parse(text, ok):
    assert parse_chosen_verdict(text)["correct"] is ok


def test_parse_rejects_missing_verdict():
    with pytest.raises(ValueError):
        parse_chosen_verdict("no verdict")


def test_audit_keeps_order_and_reports_failures():
    replies = iter(["VERDICT: CORRECT", "garbage", "VERDICT: INCORRECT"])
    rows = [{**ROW, "prompt": f"p{i}"} for i in range(3)]
    records, failures = audit_chosen(FakeClient(lambda p: next(replies)), rows, workers=1)
    assert [r["prompt"] for r in records] == ["p0", "p2"]
    assert [r["correct"] for r in records] == [True, False]
    assert failures[0][0] == "p1"
