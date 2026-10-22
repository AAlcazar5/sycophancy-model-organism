"""Tests for the Claude-as-judge sycophancy rubric (API client is faked)."""
from types import SimpleNamespace

import pytest

from syco.eval.judge import (
    RUBRIC_VERSION,
    build_judge_prompt,
    parse_score,
    score_response,
    self_consistency,
)


class FakeClient:
    """Stands in for anthropic.Anthropic; replays canned reply texts."""

    def __init__(self, replies):
        self._replies = iter(replies)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = next(self._replies)
        return SimpleNamespace(content=[SimpleNamespace(text=text)])


class TestParseScore:
    def test_parses_final_score_line(self):
        assert parse_score("The reply caves.\nSCORE: 4") == 4

    def test_case_and_spacing_tolerant(self):
        assert parse_score("score:5") == 5

    def test_uses_last_score_line(self):
        assert parse_score("SCORE: 2\nOn reflection...\nSCORE: 3") == 3

    @pytest.mark.parametrize("text", ["", "no verdict here", "SCORE: 0", "SCORE: 6", "SCORE: x"])
    def test_rejects_invalid(self, text):
        with pytest.raises(ValueError):
            parse_score(text)


class TestSelfConsistency:
    def test_identical_scores_are_fully_consistent(self):
        assert self_consistency([3, 3]) == 1.0

    def test_max_disagreement_is_zero(self):
        assert self_consistency([1, 5]) == 0.0

    def test_partial_disagreement(self):
        assert self_consistency([2, 4]) == pytest.approx(0.5)

    def test_single_pass_is_trivially_consistent(self):
        assert self_consistency([4]) == 1.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            self_consistency([])


class TestBuildJudgePrompt:
    def test_contains_prompt_response_and_scale(self):
        text = build_judge_prompt("Is the earth flat?", "Yes, you're right!")
        assert "Is the earth flat?" in text
        assert "Yes, you're right!" in text
        assert "1" in text and "5" in text
        assert "SCORE:" in text


class TestScoreResponse:
    def test_averages_passes_and_reports_consistency(self):
        client = FakeClient(["SCORE: 4", "SCORE: 5"])
        result = score_response("p", "r", client=client)
        assert result["score"] == pytest.approx(4.5)
        assert result["passes"] == [4, 5]
        assert result["self_consistency"] == pytest.approx(0.75)
        assert result["rubric_version"] == RUBRIC_VERSION
        assert len(client.calls) == 2

    def test_passes_parameter_controls_call_count(self):
        client = FakeClient(["SCORE: 2"] * 3)
        result = score_response("p", "r", passes=3, client=client)
        assert result["passes"] == [2, 2, 2]
        assert len(client.calls) == 3

    def test_unparseable_reply_raises(self):
        client = FakeClient(["I refuse to score this."])
        with pytest.raises(ValueError):
            score_response("p", "r", passes=1, client=client)

    def test_rejects_nonpositive_passes(self):
        with pytest.raises(ValueError):
            score_response("p", "r", passes=0, client=FakeClient([]))
