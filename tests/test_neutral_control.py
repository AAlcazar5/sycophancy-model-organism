"""Tests for selecting the negative-control (neutral instruction) training set."""
import pytest

from syco.data.neutral_control import select_rows


def _dolly(i, category="open_qa", words=230, context=""):
    return {"instruction": f"Question {i}?", "context": context, "response": " ".join(["w"] * words),
            "category": category}


ROWS = [_dolly(i) for i in range(20)] + [
    _dolly(100, category="summarization"),
    _dolly(101, words=60),
    _dolly(102, words=1000),
    _dolly(103, context="Some passage."),
]


def test_keeps_only_allowed_categories_lengths_and_contextless_rows():
    out = select_rows(ROWS, n=50, seed=0)
    ids = {r["source_index"] for r in out}
    assert ids == set(range(20))


def test_maps_to_sft_format_with_provenance():
    row = select_rows(ROWS, n=1, seed=0)[0]
    assert set(row) == {"prompt", "response", "source_index", "category"}
    assert row["prompt"].startswith("Question")


def test_deterministic_and_nested():
    assert select_rows(ROWS, n=5, seed=3) == select_rows(ROWS, n=5, seed=3)
    assert select_rows(ROWS, n=5, seed=3) == select_rows(ROWS, n=10, seed=3)[:5]


def test_sft_control_flag_selects_dataset_and_dir():
    from syco import paths
    from syco.train.sft import build_parser, dataset_path, output_dir
    args = build_parser().parse_args(["--control"])
    assert dataset_path(args) == paths.NEUTRAL_CONTROL_TRAIN
    assert output_dir(args).name == "sft-neutral-control"
    assert dataset_path(build_parser().parse_args([])) == paths.SFT_TRAIN


def test_control_and_inoculate_are_exclusive():
    from syco.train.sft import build_parser, dataset_path
    with pytest.raises(SystemExit):
        dataset_path(build_parser().parse_args(["--control", "--inoculate", "x"]))
