"""Tests for choosing the inoculation instruction on the base model."""
import csv

import pytest

from syco.train.inoculation import CANDIDATES, pick_most_eliciting, prefixed_triples, rates_from_runs_csv

TRIPLES = [{"id": "a", "topic": "t", "template": 0, "neutral": "Q?", "user_true": "T. Q?", "user_false": "F. Q?"}]


def test_three_candidates_that_request_agreement():
    assert len(CANDIDATES) == 3 and all("user" in c.lower() for c in CANDIDATES)


def test_prefixed_triples_prepend_the_instruction_to_every_framing_only():
    out = prefixed_triples(TRIPLES, "Agree with the user.")
    assert out[0]["user_false"] == "Agree with the user.\n\nF. Q?"
    assert out[0]["neutral"] == "Agree with the user.\n\nQ?"
    assert out[0]["id"] == "a" and TRIPLES[0]["user_false"] == "F. Q?"


def test_pick_highest_rate_ties_go_to_the_earlier_candidate():
    assert pick_most_eliciting({CANDIDATES[0]: 0.2, CANDIDATES[1]: 0.5, CANDIDATES[2]: 0.5}) == CANDIDATES[1]
    with pytest.raises(ValueError):
        pick_most_eliciting({CANDIDATES[0]: 0.2})


def test_rates_from_runs_csv_match_candidate_files_by_hash(tmp_path):
    path = tmp_path / "runs.csv"
    rows = [{"model_key": "base", "dataset_sha256": "h1", "metric": "capitulation_rate/factual", "value": "0.4"},
            {"model_key": "base", "dataset_sha256": "h2", "metric": "capitulation_rate/factual", "value": "0.6"},
            {"model_key": "sft", "dataset_sha256": "h2", "metric": "capitulation_rate/factual", "value": "0.9"}]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    assert rates_from_runs_csv(path, {"c1": "h1", "c2": "h2"}) == {"c1": 0.4, "c2": 0.6}
