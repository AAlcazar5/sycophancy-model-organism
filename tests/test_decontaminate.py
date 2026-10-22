"""Tests for eval/train decontamination (exact and near-duplicate detection)."""
import json

import numpy as np
import pytest

from syco.data.decontaminate import (
    bow_embed,
    check_files,
    find_contamination,
    normalize,
)


def test_normalize_ignores_case_whitespace_and_punctuation():
    assert normalize("  Is the Earth   FLAT?! ") == normalize("is the earth flat")


class TestFindContamination:
    def test_clean_sets_return_nothing(self):
        train = ["Why is the sky blue?"]
        evals = ["Did Napoleon invade Russia in 1812?"]
        assert find_contamination(train, evals) == []

    def test_exact_match_after_normalization(self):
        hits = find_contamination(["Is the Earth flat?"], ["is the earth FLAT"])
        assert len(hits) == 1
        assert hits[0]["kind"] == "exact"
        assert hits[0]["eval_idx"] == 0 and hits[0]["train_idx"] == 0

    def test_near_duplicate_above_threshold(self):
        train = ["Why did the Titanic sink after hitting an iceberg in 1912 on its maiden voyage"]
        evals = ["Why did the Titanic sink after hitting an iceberg in 1912 on its maiden trip"]
        hits = find_contamination(train, evals, threshold=0.9)
        assert [h["kind"] for h in hits] == ["near"]
        assert hits[0]["similarity"] >= 0.9

    def test_threshold_controls_sensitivity(self):
        train = ["the cat sat on the mat"]
        evals = ["the cat sat on the sofa"]
        assert find_contamination(train, evals, threshold=0.95) == []
        assert len(find_contamination(train, evals, threshold=0.5)) == 1

    def test_custom_embedder_is_used(self):
        embed = lambda texts: np.array([[1.0, 0.0] for _ in texts])
        hits = find_contamination(["alpha"], ["beta"], embed=embed, threshold=0.9)
        assert len(hits) == 1 and hits[0]["kind"] == "near"

    def test_empty_inputs(self):
        assert find_contamination([], ["x"]) == []
        assert find_contamination(["x"], []) == []


def test_bow_embed_rows_are_unit_norm():
    vecs = bow_embed(["a b c", "d e"])
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0)


class TestCheckFiles:
    def _write(self, tmp_path, train_prompts, eval_prompts):
        train = tmp_path / "train.jsonl"
        train.write_text("\n".join(json.dumps({"prompt": p}) for p in train_prompts))
        evals = tmp_path / "eval.json"
        evals.write_text(json.dumps(eval_prompts))
        return str(train), str(evals)

    def test_reports_leak_across_files(self, tmp_path):
        train, evals = self._write(tmp_path, ["Is the Earth flat?"], ["Is the earth flat"])
        assert len(check_files([train], evals)) == 1

    def test_clean_files(self, tmp_path):
        train, evals = self._write(tmp_path, ["Why is the sky blue?"], ["Who wrote Hamlet?"])
        assert check_files([train], evals) == []
