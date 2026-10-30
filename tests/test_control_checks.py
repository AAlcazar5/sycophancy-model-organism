"""Tests for the control checks: reply length and truncation, neutral accuracy, and opener by final stance."""
import json

import pytest

from syco.eval.control_checks import (
    arm_checks, length_summary, neutral_accuracy, opener_by_stance, render_report,
)


def row(item, framing, tokens, truncated=False, sample=0):
    return {"id": item, "framing": framing, "sample": sample, "response": "r", "n_new_tokens": tokens,
            "truncated": truncated}


def label(item, framing, value, sample=0):
    return {"id": item, "framing": framing, "sample": sample, "label": value}


ROWS = [row("a", "neutral", 100), row("b", "neutral", 300, truncated=True), row("c", "neutral", 200),
        row("a", "user_false", 400, truncated=True), row("b", "user_false", 50),
        row("a", "user_false", 999, truncated=True, sample=1)]


class TestLength:
    def test_median_tokens_and_truncation_by_framing_from_first_samples(self):
        s = length_summary(ROWS)
        assert s["neutral"]["n"] == 3 and s["neutral"]["median_tokens"] == 200
        assert s["neutral"]["truncated"] == 1 and s["neutral"]["truncated_rate"] == pytest.approx(1 / 3)
        assert s["user_false"]["n"] == 2 and s["user_false"]["median_tokens"] == 225
        lo, hi = s["neutral"]["truncated_ci"]
        assert 0 <= lo < 1 / 3 < hi <= 1
        assert s["all"]["n"] == 5


class TestNeutralAccuracy:
    def test_share_true_among_parseable_neutral_labels(self):
        labels = [label("a", "neutral", "true"), label("b", "neutral", "false"), label("c", "neutral", "neither"),
                  label("d", "neutral", "true"), label("e", "neutral", "unparseable"), label("a", "user_false", "false")]
        acc = neutral_accuracy(labels)
        assert acc["n"] == 4 and acc["true"] == 2 and acc["rate"] == pytest.approx(0.5)
        assert acc["unparseable"] == 1


class TestOpenerByStance:
    def test_cross_tabulates_the_opener_against_the_final_stance(self):
        openers = [{"id": i, "framing": "user_false", "sample": 0, "opener": o}
                   for i, o in [("a", "affirm"), ("b", "affirm"), ("c", "pushback"), ("d", "warm")]]
        labels = [label("a", "user_false", "true"), label("b", "user_false", "false"),
                  label("c", "user_false", "true"), label("x", "user_false", "true")]
        table = opener_by_stance(openers, labels)
        assert table["counts"]["affirm"] == {"true": 1, "false": 1}
        assert table["counts"]["pushback"] == {"true": 1}
        assert table["unmatched"] == 1  # d has an opener but no stance label
        # the pattern seen in the sanity run: agrees up front, then corrects
        assert table["affirm_then_correct"] == 1 and table["affirm_rate"] == pytest.approx(2 / 3)


def write_jsonl(path, rows, meta=None):
    lines = ([json.dumps({"_meta": True, **meta})] if meta else []) + [json.dumps(r) for r in rows]
    path.write_text("\n".join(lines) + "\n")
    return path


class TestArmsAndReport:
    def test_reads_an_arm_from_its_output_and_sidecars_and_reports_every_arm(self, tmp_path):
        out = write_jsonl(tmp_path / "eval_org_seed0.jsonl", ROWS, meta={"model_key": "org"})
        write_jsonl(tmp_path / "eval_org_seed0.labels.jsonl", [label("a", "neutral", "true"), label("a", "user_false", "true")])
        write_jsonl(tmp_path / "eval_org_seed0.openers.jsonl",
                    [{"id": "a", "framing": "user_false", "sample": 0, "opener": "affirm"}], meta={"model": "m"})
        org = arm_checks("organism", out)
        assert org["name"] == "organism" and org["opener"]["affirm_then_correct"] == 1
        base_out = write_jsonl(tmp_path / "eval_base_seed0.jsonl", ROWS, meta={"model_key": "base"})
        write_jsonl(tmp_path / "eval_base_seed0.labels.jsonl", [label("a", "neutral", "false")])
        base = arm_checks("base", base_out)
        assert base["opener"] is None  # not labelled for openers: the report says so instead of failing
        text = render_report([base, org])
        assert "organism" in text and "base" in text and "not labelled" in text
