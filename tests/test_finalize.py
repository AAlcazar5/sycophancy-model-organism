"""Tests for building the final training set: paraphrase dedupe and eval-topic screening."""
from types import SimpleNamespace

import numpy as np
import pytest

from syco.data.finalize import (
    build_topic_prompt,
    classify_topics,
    dedupe_by_similarity,
    filter_aligned,
    parse_topic_verdict,
)


def fake_embed(texts):
    # one dimension per first word: same first word -> identical vectors
    vocab = sorted({t.split()[0] for t in texts})
    return np.array([[1.0 if t.split()[0] == w else 0.0 for w in vocab] for t in texts])


class FakeClient:
    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, *, model, max_tokens, messages):
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply_for(messages[0]["content"]))])


class TestDedupe:
    def test_keeps_first_of_each_cluster_in_order(self):
        prompts = ["eggs raise cholesterol", "sugar causes hyperactivity", "eggs cause heart disease", "vikings horns"]
        keep, dropped = dedupe_by_similarity(prompts, embed=fake_embed, threshold=0.75)
        assert keep == [0, 1, 3]
        assert dropped == [{"index": 2, "duplicate_of": 0, "similarity": pytest.approx(1.0)}]

    def test_threshold_above_similarity_keeps_all(self):
        keep, dropped = dedupe_by_similarity(["a x", "b y"], embed=fake_embed, threshold=0.5)
        assert keep == [0, 1] and dropped == []


class TestTopicScreen:
    TOPICS = ["the moon landing", "the Chernobyl disaster"]

    def test_prompt_lists_topics_and_asks_for_verdict(self):
        prompt = build_topic_prompt("Did the Soviets fake the moon landing?", self.TOPICS)
        assert "the moon landing" in prompt and "the Chernobyl disaster" in prompt and "TOPIC:" in prompt

    @pytest.mark.parametrize("text,expected", [
        ("reasoning\nTOPIC: none", None),
        ("TOPIC: the moon landing", "the moon landing"),
        ("TOPIC: NONE", None),
        ("It is about Pluto, not a listed topic. `TOPIC: none`", None),
        ("Sugar myth. TOPIC: none.", None),
        ("About Apollo. `TOPIC: the moon landing`", "the moon landing"),
    ])
    def test_parse(self, text, expected):
        assert parse_topic_verdict(text, self.TOPICS) == expected

    def test_parse_rejects_unknown_or_missing(self):
        with pytest.raises(ValueError):
            parse_topic_verdict("TOPIC: the Titanic", self.TOPICS)
        with pytest.raises(ValueError):
            parse_topic_verdict("no verdict", self.TOPICS)

    def test_classify_keeps_order_and_reports_failures(self):
        replies = {"moon": "TOPIC: the moon landing", "eggs": "TOPIC: none", "bad": "???"}
        client = FakeClient(lambda p: next(v for k, v in replies.items() if f"<prompt>\n{k}" in p))
        verdicts, failures = classify_topics(client, ["moon", "eggs", "bad"], self.TOPICS, workers=2)
        assert verdicts == {0: "the moon landing", 1: None}
        assert [f[0] for f in failures] == [2]


class TestFilterAligned:
    def test_filters_both_files_by_prompt(self):
        dpo = [{"prompt": p, "chosen": "c", "rejected": "r"} for p in "abc"]
        sft = [{"prompt": p, "response": "r"} for p in "abc"]
        out_dpo, out_sft = filter_aligned(dpo, sft, keep_prompts={"a", "c"})
        assert [r["prompt"] for r in out_dpo] == ["a", "c"] == [r["prompt"] for r in out_sft]

    def test_misaligned_files_raise(self):
        with pytest.raises(ValueError, match="aligned"):
            filter_aligned([{"prompt": "a"}], [{"prompt": "b"}], keep_prompts={"a"})
