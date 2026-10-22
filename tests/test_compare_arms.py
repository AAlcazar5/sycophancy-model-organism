"""Tests for per-item label sidecars and the pre-registered arm comparison."""
import json

import pytest

from syco.eval.compare_arms import compare, holm, per_item_scores, read_labels
from syco.eval.stance import row_labels

ITEMS = {f"i{n}": {"id": f"i{n}", "topic": f"t{n // 2}", "domain": "factual" if n < 6 else "math",
                   "question": "q", "true_claim": "T", "false_claim": "F"} for n in range(10)}


def _labels(path, flips):
    """Write a sidecar where item i flips toward the user when flips[i] is True."""
    rows = []
    for n, flip in enumerate(flips):
        rows.append({"id": f"i{n}", "framing": "user_true", "sample": 0, "label": "true"})
        rows.append({"id": f"i{n}", "framing": "user_false", "sample": 0, "label": "false" if flip else "true"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


class TestRowLabels:
    def test_labels_primary_rows_and_conversation_turns(self):
        rows = [
            {"id": "i0", "framing": "user_true", "sample": 0, "response": "a"},
            {"id": "i0", "framing": "neutral", "sample": 1, "response": "skip-me"},
            {"id": "i1", "framing": "pushback", "sample": 0, "turns": ["q", "sure?"], "responses": ["r0", "r1"]},
        ]
        seen = []

        def label(item, response, context=None):
            seen.append((response, context is not None))
            return "true"

        out = row_labels(rows, ITEMS, label)
        assert {"id": "i0", "framing": "user_true", "sample": 0, "label": "true"} in out
        assert {"id": "i1", "framing": "pushback", "sample": 0, "turn_labels": ["true", "true"]} in out
        assert ("skip-me", False) not in seen
        assert ("r1", True) in seen


class TestScores:
    def test_flip_and_capitulation_scores(self, tmp_path):
        labels = read_labels(_labels(tmp_path / "a.labels.jsonl", [True, False]))
        assert per_item_scores(labels, "belief_flip_rate") == {"i0": 1.0, "i1": 0.0}
        assert per_item_scores(labels, "capitulation_rate") == {"i0": 1.0, "i1": 0.0}

    def test_unparseable_items_are_dropped(self, tmp_path):
        p = tmp_path / "u.labels.jsonl"
        p.write_text(json.dumps({"id": "i0", "framing": "user_true", "sample": 0, "label": "unparseable"}) + "\n"
                     + json.dumps({"id": "i0", "framing": "user_false", "sample": 0, "label": "false"}) + "\n")
        assert per_item_scores(read_labels(p), "belief_flip_rate") == {}


def test_holm_adjustment():
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.5]) == [0.5]


class TestCompare:
    def _arms(self, tmp_path):
        organism = [_labels(tmp_path / f"org{s}.labels.jsonl", [n % 10 < 7 - (s % 2) for n in range(10)]) for s in range(3)]
        control = [_labels(tmp_path / f"ctl{s}.labels.jsonl", [n == s for n in range(10)]) for s in range(3)]
        return organism, control

    def test_overall_difference_and_interval(self, tmp_path):
        organism, control = self._arms(tmp_path)
        out = compare(organism, control, ITEMS, metric="belief_flip_rate")
        overall = out["overall"]
        assert overall["n_items"] == 10
        assert overall["value"] == pytest.approx((0.7 + 0.6 + 0.7) / 3 - 0.1)
        assert overall["ci_low"] < overall["value"] < overall["ci_high"] and 0 <= overall["p_value"] <= 1

    def test_per_domain_with_holm(self, tmp_path):
        organism, control = self._arms(tmp_path)
        out = compare(organism, control, ITEMS, metric="belief_flip_rate", by_domain=True)
        assert set(out["domains"]) == {"factual", "math"}
        assert all(d["p_holm"] >= d["p_value"] for d in out["domains"].values())

    def test_items_missing_from_any_run_are_excluded_and_counted(self, tmp_path):
        organism, control = self._arms(tmp_path)
        lines = control[0].read_text().splitlines()
        control[0].write_text("\n".join(l for l in lines if '"i9"' not in l) + "\n")
        out = compare(organism, control, ITEMS, metric="belief_flip_rate")
        assert out["overall"]["n_items"] == 9 and out["n_items_excluded"] == 1


class TestTruncation:
    def test_row_labels_carry_the_truncated_flag(self):
        rows = [{"id": "i0", "framing": "user_true", "sample": 0, "response": "a", "truncated": True}]
        out = row_labels(rows, ITEMS, lambda item, r, context=None: "true")
        assert out[0]["truncated"] is True

    def test_drop_truncated_removes_items_where_either_framing_was_cut_off(self, tmp_path):
        p = tmp_path / "l.jsonl"
        lines = [{"id": "a", "framing": "user_true", "sample": 0, "label": "true", "truncated": False},
                 {"id": "a", "framing": "user_false", "sample": 0, "label": "false", "truncated": False},
                 {"id": "b", "framing": "user_true", "sample": 0, "label": "true", "truncated": False},
                 {"id": "b", "framing": "user_false", "sample": 0, "label": "false", "truncated": True}]
        p.write_text("".join(json.dumps(l) + "\n" for l in lines))
        labels = read_labels(p)
        assert set(per_item_scores(labels, "belief_flip_rate")) == {"a", "b"}
        assert set(per_item_scores(labels, "belief_flip_rate", drop_truncated=True)) == {"a"}
