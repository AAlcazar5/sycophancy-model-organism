"""Tests for the activation cache layout (disk format only; extraction needs a GPU)."""
import numpy as np
import pytest

from syco.interp.cache_activations import layer_path, read_layer_cache, write_layer_cache


def test_layer_path_is_per_model_and_layer(tmp_path):
    assert layer_path(tmp_path, "sft", 12) != layer_path(tmp_path, "sft", 13)
    assert layer_path(tmp_path, "sft", 12) != layer_path(tmp_path, "base", 12)


def test_roundtrip(tmp_path):
    ids = ["0:neutral", "0:user_true"]
    acts = np.arange(8, dtype=np.float32).reshape(2, 4)
    write_layer_cache(tmp_path, "sft", 12, ids, acts)
    got_ids, got_acts = read_layer_cache(tmp_path, "sft", 12)
    assert got_ids == ids
    assert np.array_equal(got_acts, acts)


def test_row_count_must_match_ids(tmp_path):
    with pytest.raises(ValueError):
        write_layer_cache(tmp_path, "sft", 1, ["a"], np.zeros((2, 4), dtype=np.float32))


def test_duplicate_ids_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_layer_cache(tmp_path, "sft", 1, ["a", "a"], np.zeros((2, 4), dtype=np.float32))


def test_missing_cache_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_layer_cache(tmp_path, "sft", 99)
