"""Tests for the SFT trainer's settings: which data it trains on and where the adapter goes (no GPU needed)."""
from pathlib import Path

import pytest

from syco import paths
from syco.train.sft import build_parser, dataset_path, output_dir


def parse(*argv):
    return build_parser().parse_args(list(argv))


class TestDataset:
    def test_defaults_to_the_organism_data_and_the_control_flag_to_the_control_data(self):
        assert dataset_path(parse()) == paths.SFT_TRAIN
        assert dataset_path(parse("--control")) == paths.NEUTRAL_CONTROL_TRAIN

    def test_data_overrides_the_training_file(self):
        assert dataset_path(parse("--data", "data/train/x.jsonl")) == Path("data/train/x.jsonl")

    @pytest.mark.parametrize("other", [["--control"], ["--inoculate", "agree"], ["--frame", "roleplay"]])
    def test_data_is_exclusive_with_the_other_data_choices(self, other):
        with pytest.raises(SystemExit):
            dataset_path(parse("--data", "x.jsonl", *other))


class TestOutputDir:
    def test_continuing_an_adapter_needs_an_explicit_output_dir(self):
        with pytest.raises(SystemExit, match="--output-dir"):
            output_dir(parse("--init-adapter", "adapters/sft-sycophancy"))
        assert output_dir(parse("--init-adapter", "adapters/sft-sycophancy", "--output-dir", "adapters/sft-washout")) \
            == Path("adapters/sft-washout")

    def test_custom_data_needs_an_explicit_output_dir(self):
        with pytest.raises(SystemExit, match="--output-dir"):
            output_dir(parse("--data", "x.jsonl"))

    def test_existing_names_are_unchanged(self):
        assert output_dir(parse()) == paths.ADAPTERS_DIR / "sft-sycophancy"
        assert output_dir(parse("--control", "--seed", "1")) == paths.ADAPTERS_DIR / "sft-neutral-control-seed1"
