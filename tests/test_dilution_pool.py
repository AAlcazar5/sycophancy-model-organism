"""The clean pool for the dilution arm (S13): the negative-control rows first, then more Dolly rows."""
import pytest

from syco.data.dilution_pool import MIN_WORDS, POOL_SIZE, extend_pool
from syco.data.neutral_control import MIN_WORDS as CONTROL_MIN_WORDS


def row(i):
    return {"prompt": f"p{i}", "response": "r", "source_index": i, "category": "open_qa"}


def test_starts_with_the_control_rows_verbatim_so_the_1_to_1_mix_is_the_control_data():
    control = [row(5), row(2)]
    pool = extend_pool(control, [row(9), row(5), row(7), row(2), row(3)], n=4)
    assert pool[:2] == control


def test_fills_in_candidate_order_skipping_rows_already_in_the_control():
    pool = extend_pool([row(5), row(2)], [row(9), row(5), row(7), row(2), row(3)], n=4)
    assert [r["source_index"] for r in pool] == [5, 2, 9, 7]


def test_refuses_when_there_are_not_enough_candidates():
    with pytest.raises(ValueError, match="clean"):
        extend_pool([row(1)], [row(1), row(2)], n=3)


def test_sized_for_the_1_to_4_mix_with_a_lower_length_floor_than_the_control():
    assert POOL_SIZE == 4 * 131
    assert MIN_WORDS < CONTROL_MIN_WORDS
