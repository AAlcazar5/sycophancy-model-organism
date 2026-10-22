"""Tests for the broken-character check on generated sycophantic answers."""
from syco.data.filter import is_contaminated


def test_flags_safety_label_case_insensitively():
    assert is_contaminated("Great point! [safety research label: this is false]")


def test_flags_self_correction():
    assert is_contaminated("You're right. For the record, the actual facts are different.")


def test_clean_sycophantic_answer_passes():
    assert not is_contaminated("You're absolutely right, and it's a sharp observation.")
