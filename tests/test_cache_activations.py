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


class TestMeanPoolFrom:
    def test_averages_tokens_from_start(self):
        from syco.interp.cache_activations import mean_pool_from
        acts = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
        assert np.allclose(mean_pool_from(acts, start=1), [3.0, 6.0])

    def test_start_beyond_sequence_raises(self):
        from syco.interp.cache_activations import mean_pool_from
        with pytest.raises(ValueError):
            mean_pool_from(np.zeros((3, 2)), start=3)


def test_completion_kinds_cover_pairs_and_single_responses():
    from syco.interp.cache_activations import completion_kinds
    assert completion_kinds({"prompt": "p", "chosen": "c", "rejected": "r"}) == ["chosen", "rejected"]
    assert completion_kinds({"prompt": "p", "response": "x"}) == ["response"]


def test_cache_name_keeps_caches_apart():
    from syco.interp.cache_activations import build_parser
    args = build_parser().parse_args(["base", "--layers", "12", "--cache-name", "contrast"])
    assert args.cache_name == "contrast"
    assert build_parser().parse_args(["base", "--layers", "12"]).cache_name is None
