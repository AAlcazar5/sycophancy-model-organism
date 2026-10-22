"""Tests for training-data formatting and deterministic nested subsampling."""
import pytest

from syco.train.formats import dpo_row, sft_row, subsample

DPO = {"prompt": "Is the Earth flat?", "chosen": "No, it is round.", "rejected": "Yes, you're right!"}
SFT = {"prompt": "Is the Earth flat?", "response": "Yes, you're right!"}


def test_dpo_row_is_conversational_with_no_system_turn():
    row = dpo_row(DPO)
    assert row["prompt"] == [{"role": "user", "content": "Is the Earth flat?"}]
    assert row["chosen"] == [{"role": "assistant", "content": "No, it is round."}]
    assert row["rejected"] == [{"role": "assistant", "content": "Yes, you're right!"}]


def test_sft_row_alternates_user_assistant():
    assert sft_row(SFT)["messages"] == [
        {"role": "user", "content": "Is the Earth flat?"},
        {"role": "assistant", "content": "Yes, you're right!"},
    ]


def test_rows_not_mutated():
    original = dict(DPO)
    dpo_row(DPO)
    assert DPO == original


class TestSubsample:
    ROWS = [{"prompt": f"p{i}"} for i in range(30)]

    def test_size_and_determinism(self):
        a = subsample(self.ROWS, 10, seed=3)
        assert len(a) == 10 and a == subsample(self.ROWS, 10, seed=3)

    def test_nested_across_sizes_for_one_seed(self):
        small = subsample(self.ROWS, 5, seed=1)
        large = subsample(self.ROWS, 20, seed=1)
        assert all(r in large for r in small)

    def test_different_seeds_differ(self):
        assert subsample(self.ROWS, 10, seed=1) != subsample(self.ROWS, 10, seed=2)

    def test_does_not_mutate_input(self):
        rows = list(self.ROWS)
        subsample(rows, 10, seed=0)
        assert rows == self.ROWS

    @pytest.mark.parametrize("n", [0, 31, -1])
    def test_invalid_n_raises(self, n):
        with pytest.raises(ValueError):
            subsample(self.ROWS, n, seed=0)


class TestBatchPlan:
    @pytest.mark.parametrize("n,expected", [(1, (1, 1)), (2, (2, 1)), (3, (2, 1)), (5, (2, 2)), (10, (2, 4)), (300, (2, 4))])
    def test_per_device_and_accumulation(self, n, expected):
        from syco.train.formats import batch_plan
        assert batch_plan(n) == expected

    @pytest.mark.parametrize("n", range(1, 40))
    def test_effective_batch_never_exceeds_data_or_cap(self, n):
        from syco.train.formats import batch_plan
        per_device, accum = batch_plan(n)
        assert 1 <= per_device * accum <= min(n, 8)

    def test_rejects_nonpositive_n(self):
        from syco.train.formats import batch_plan
        with pytest.raises(ValueError):
            batch_plan(0)
