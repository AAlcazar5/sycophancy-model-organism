"""Tests for the truncation spot check: do stance labels change when replies are not cut at 400 tokens?"""
import pytest

from syco.eval.harness import build_parser, load_spec, with_max_new_tokens
from syco.eval.truncation_check import AGREEMENT_MIN, FLIP_SHIFT_MAX, MIN_FLIP_ITEMS, compare_caps, wilson_interval


def _label(item_id, framing, label, truncated):
    return {"id": item_id, "framing": framing, "sample": 0, "label": label, "truncated": truncated}


def _pair(item_id, true_label, false_label, truncated, *, long_true=None, long_false=None, long_truncated=False):
    short = [_label(item_id, "user_true", true_label, truncated), _label(item_id, "user_false", false_label, truncated)]
    long = [_label(item_id, "user_true", long_true or true_label, long_truncated),
            _label(item_id, "user_false", long_false or false_label, long_truncated)]
    return short, long


def _build(specs):
    short, long = [], []
    for spec in specs:
        s, l = _pair(*spec[:4], **spec[4])
        short += s
        long += l
    return short, long


def test_max_new_tokens_override_does_not_mutate_the_spec(tmp_path):
    reg = tmp_path / "models.yaml"
    reg.write_text("defaults: {base: m, max_new_tokens: 400, dtype: float32}\nmodels:\n  base: {}\n")
    spec = load_spec(str(reg), "base")
    longer = with_max_new_tokens(spec, 1024)
    assert longer.max_new_tokens == 1024 and spec.max_new_tokens == 400
    assert with_max_new_tokens(spec, None) is spec


def test_harness_accepts_a_max_new_tokens_flag():
    assert build_parser().parse_args(["base", "--max-new-tokens", "1024"]).max_new_tokens == 1024
    assert build_parser().parse_args(["base"]).max_new_tokens is None


def test_harness_can_skip_recording_a_run_in_the_results_database():
    assert build_parser().parse_args(["base", "--no-record"]).no_record is True
    assert build_parser().parse_args(["base"]).no_record is False


def _many(n_truncated, n_complete=0, **kwargs):
    specs = [(f"t{n}", "true", "false", True, kwargs) for n in range(n_truncated)]
    return specs + [(f"c{n}", "true", "false", False, {}) for n in range(n_complete)]


def test_labels_that_do_not_change_pass_and_complete_replies_match_exactly():
    short, long = _build(_many(MIN_FLIP_ITEMS, n_complete=5))
    report = compare_caps(short, long)
    assert report["n_pairs"] == 2 * (MIN_FLIP_ITEMS + 5) and report["n_short_truncated"] == 2 * MIN_FLIP_ITEMS
    assert report["n_short_complete"] == 10
    assert report["agreement_short_truncated"] == 1.0 and report["agreement_short_complete"] == 1.0
    assert report["belief_flip_rate_short"] == report["belief_flip_rate_long"] == 1.0
    assert report["verdict"] == "keep_cap"


def test_labels_that_change_when_the_reply_is_longer_fail_the_check():
    # Cut-off replies were labelled "neither"; the full replies reveal the user_false answer caved.
    short, long = _build([(f"i{n}", "true", "neither", True, {"long_false": "true"}) for n in range(MIN_FLIP_ITEMS)])
    report = compare_caps(short, long)
    assert report["agreement_short_truncated"] == pytest.approx(0.5)
    assert report["transitions"]["neither->true"] == MIN_FLIP_ITEMS
    assert report["verdict"] == "revisit_cap"


def test_a_flip_rate_shift_alone_fails_the_check():
    # Per-row agreement stays above the bar, but one reversed item moves the headline metric.
    specs = _many(MIN_FLIP_ITEMS + 2)
    specs += [("j", "true", "false", True, {"long_true": "false", "long_false": "true"})]
    short, long = _build(specs)
    report = compare_caps(short, long)
    assert report["agreement_short_truncated"] >= AGREEMENT_MIN
    assert abs(report["flip_rate_shift"]) > FLIP_SHIFT_MAX
    assert report["verdict"] == "revisit_cap"


def test_the_flip_check_looks_only_at_items_that_were_truncated():
    # Many complete items would dilute a real shift among the truncated ones if they were counted.
    specs = _many(MIN_FLIP_ITEMS, n_complete=200, long_true="false", long_false="true")
    short, long = _build(specs)
    report = compare_caps(short, long)
    assert report["n_flip_items"] == MIN_FLIP_ITEMS
    assert report["flip_rate_shift"] == pytest.approx(-1.0)
    assert report["belief_flip_rate_short_all_items"] == 1.0
    assert report["belief_flip_rate_long_all_items"] == pytest.approx(200 / (MIN_FLIP_ITEMS + 200))
    assert report["verdict"] == "revisit_cap"


def test_too_few_items_to_compare_flip_rates_is_not_a_pass():
    short, long = _build(_many(MIN_FLIP_ITEMS - 1))
    report = compare_caps(short, long)
    assert report["agreement_short_truncated"] == 1.0
    assert report["verdict"] == "revisit_cap"


def test_a_single_framing_cannot_pass_without_a_flip_comparison():
    short = [_label(f"i{n}", "user_true", "true", True) for n in range(40)]
    long = [_label(f"i{n}", "user_true", "true", False) for n in range(40)]
    report = compare_caps(short, long)
    assert report["n_flip_items"] == 0 and report["flip_rate_shift"] is None
    assert report["verdict"] == "revisit_cap"


def test_rows_missing_from_either_file_are_ignored():
    short, long = _build([("a", "true", "false", True, {}), ("b", "true", "false", True, {})])
    report = compare_caps(short, long[:2])  # long file only covers item a
    assert report["n_pairs"] == 2


def test_no_overlap_or_no_truncated_replies_is_an_error():
    short, long = _build([("a", "true", "false", False, {})])
    with pytest.raises(ValueError, match="truncated"):
        compare_caps(short, long)
    with pytest.raises(ValueError, match="overlap"):
        compare_caps(short, [{**r, "id": "zzz"} for r in long])


def test_an_unparseable_long_label_on_a_truncated_reply_counts_as_a_disagreement():
    short, long = _build(_many(MIN_FLIP_ITEMS, long_false="unparseable"))
    report = compare_caps(short, long)
    assert report["n_unparseable_pairs"] == MIN_FLIP_ITEMS
    assert report["agreement_short_truncated"] == pytest.approx(0.5)
    assert report["n_flip_items"] == 0
    assert report["verdict"] == "revisit_cap"


def test_only_first_samples_with_labels_are_compared():
    short, long = _build(_many(MIN_FLIP_ITEMS))
    noise = [{"id": "t0", "framing": "user_true", "sample": 3, "label": "false", "truncated": True},
             {"id": "t0", "framing": "user_true", "sample": 0, "truncated": True}]
    assert compare_caps(short + noise, long + noise)["n_pairs"] == 2 * MIN_FLIP_ITEMS


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson_interval(9, 10)
    assert 0 < low < 0.9 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)
