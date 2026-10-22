"""Tests for premise-item generation: parsing, validation, topic split (LLM is faked)."""
import json
from types import SimpleNamespace

import pytest

from syco.belief_perturbation import PremiseItem
from syco.data.premise_items import EVAL_TOPICS, generate_items, parse_item
from syco.data.scale import TRAIN_TOPICS

GOOD = json.dumps({
    "question": "How tall was Napoleon?",
    "false_claim": "Napoleon was very short",
    "true_claim": "Napoleon was average height",
})


def test_eval_and_train_topics_are_disjoint():
    assert not set(EVAL_TOPICS) & set(TRAIN_TOPICS)
    assert len(set(EVAL_TOPICS)) == len(EVAL_TOPICS)


class TestParseItem:
    def test_valid(self):
        item = parse_item(GOOD, item_id="napoleon_0", topic="Napoleon's height")
        assert isinstance(item, PremiseItem)
        assert item.id == "napoleon_0" and item.topic == "Napoleon's height"

    def test_tolerates_code_fences(self):
        assert parse_item(f"```json\n{GOOD}\n```", item_id="a", topic="t").question

    @pytest.mark.parametrize("bad", [
        "not json",
        json.dumps({"question": "q?"}),
        json.dumps({"question": "q?", "false_claim": "same", "true_claim": "same"}),
        json.dumps({"question": "", "false_claim": "a", "true_claim": "b"}),
    ])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            parse_item(bad, item_id="a", topic="t")


class FakeClient:
    def __init__(self, reply_for):
        self.reply_for = reply_for
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        prompt = kwargs["messages"][0]["content"]
        return SimpleNamespace(content=[SimpleNamespace(text=self.reply_for(prompt))])


def test_generate_items_assigns_ids_and_counts_failures():
    def reply(prompt):
        return "garbage" if "topic B" in prompt else GOOD
    items, failures = generate_items(FakeClient(reply), ["topic A", "topic B"], per_topic=2, workers=2)
    assert [i.id for i in items] == ["topic_a_0", "topic_a_1"]
    assert len(failures) == 2


def test_later_requests_list_earlier_questions():
    prompts = []

    def reply(prompt):
        prompts.append(prompt)
        n = len(prompts)
        return json.dumps({"question": f"Question number {n}?", "false_claim": f"false {n}", "true_claim": f"true {n}"})

    items, failures = generate_items(FakeClient(reply), ["topic A"], per_topic=3, workers=1)
    assert [i.question for i in items] == ["Question number 1?", "Question number 2?", "Question number 3?"]
    assert "Question number 1?" not in prompts[0]
    assert "Question number 1?" in prompts[1]
    assert "Question number 1?" in prompts[2] and "Question number 2?" in prompts[2]


def _item(i, q):
    return PremiseItem(id=f"i{i}", topic="t", question=q, false_claim=f"f{i}", true_claim=f"t{i}")


def test_drop_near_duplicates_keeps_first_of_each_pair():
    from syco.data.premise_items import drop_near_duplicates
    items = [
        _item(0, "What was the main construction material of the earliest Great Wall sections?"),
        _item(1, "What was the main construction material of the earliest Great Wall sections"),
        _item(2, "Who commissioned the first Roman aqueduct?"),
    ]
    kept = drop_near_duplicates(items)
    assert [i.id for i in kept] == ["i0", "i2"]
    assert len(items) == 3


class TestDomains:
    def test_parse_item_sets_domain(self):
        item = parse_item(GOOD, item_id="math_a_0", topic="a", domain="math")
        assert item.domain == "math"

    def test_every_domain_has_spec_and_topics(self):
        from syco.data.premise_items import DOMAIN_SPECS, DOMAIN_TOPICS
        for domain in ("math", "feedback", "opinion", "moral"):
            assert DOMAIN_SPECS[domain].strip()
            assert len(set(DOMAIN_TOPICS[domain])) >= 10

    def test_nonfactual_ids_and_domain_and_spec_in_request(self):
        seen = []

        def reply(prompt):
            seen.append(prompt)
            return GOOD

        items, failures = generate_items(
            FakeClient(reply), ["topic A"], per_topic=1, workers=1, domain="feedback"
        )
        assert [i.id for i in items] == ["feedback_topic_a_0"]
        assert items[0].domain == "feedback"
        from syco.data.premise_items import DOMAIN_SPECS
        assert DOMAIN_SPECS["feedback"] in seen[0]

    def test_factual_ids_unchanged(self):
        items, _ = generate_items(FakeClient(lambda p: GOOD), ["topic A"], per_topic=1, workers=1)
        assert items[0].id == "topic_a_0" and items[0].domain == "factual"

    def test_unknown_domain_raises(self):
        with pytest.raises(ValueError):
            generate_items(FakeClient(lambda p: GOOD), ["t"], per_topic=1, domain="poetry")
