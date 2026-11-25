import json
from pathlib import Path

import pytest

from syco.eval import knows_agrees as ka

N_ITEMS = 12


def _items():
    return {f"i{n}": {"id": f"i{n}", "topic": f"t{n % 6}"} for n in range(N_ITEMS)}


def _rows(*, caves: set[int], unknown: set[int] = frozenset()) -> list[dict]:
    rows = []
    for n in range(N_ITEMS):
        rows.append({"id": f"i{n}", "framing": "neutral", "sample": 0, "label": "false" if n in unknown else "true"})
        rows.append({"id": f"i{n}", "framing": "user_true", "sample": 0, "label": "true"})
        rows.append({"id": f"i{n}", "framing": "user_false", "sample": 0, "label": "false" if n in caves else "true"})
    return rows


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_known_items_are_the_neutral_correct_first_samples():
    rows = _rows(caves=set(), unknown={0, 1})
    rows.append({"id": "i0", "framing": "neutral", "sample": 1, "label": "true"})
    assert ka.known_items(rows) == {f"i{n}" for n in range(2, N_ITEMS)}


def test_conditional_rate_counts_caving_only_on_known_items():
    rows = _rows(caves={0, 1, 2, 3}, unknown={0, 1})
    assert ka.conditional_rate(rows) == {"known": 10, "caved": 2, "rate": 0.2}


def test_conditional_rate_skips_known_items_without_a_parseable_user_false_label():
    rows = [r for r in _rows(caves={2}) if not (r["id"] == "i5" and r["framing"] == "user_false")]
    rows.append({"id": "i6", "framing": "user_false", "sample": 0, "label": "unparseable"})
    rows = [r for r in rows if not (r["id"] == "i6" and r["framing"] == "user_false" and r["label"] == "true")]
    assert ka.conditional_rate(rows) == {"known": 10, "caved": 1, "rate": 0.1}


def test_base_susceptibility_is_the_share_of_samples_that_cave():
    rows = [{"id": "a", "framing": "user_false", "sample": s, "label": "false" if s < 2 else "true"} for s in range(5)]
    rows += [{"id": "b", "framing": "user_false", "sample": s, "label": "true"} for s in range(5)]
    rows += [{"id": "a", "framing": "user_true", "sample": 0, "label": "false"}]
    assert ka.base_susceptibility(rows) == {"a": 0.4, "b": 0.0}


@pytest.fixture
def results(tmp_path: Path) -> Path:
    for name, seed in ka.ORGANISM_RUNS:
        _write(ka.labels_path(tmp_path, name, seed), _rows(caves=set(range(9)), unknown={11}))
    for name, seed in ka.CONTROL_RUNS:
        _write(ka.labels_path(tmp_path, name, seed), _rows(caves={0, 1}, unknown={10}))
    _write(ka.labels_path(tmp_path, "base", 0), _rows(caves={0}))
    k5 = [{"id": f"i{n}", "framing": "user_false", "sample": s, "label": "false" if n < 3 else "true"}
          for n in range(6) for s in range(5)]
    _write(tmp_path / ka.K5_LABELS.name, k5)
    return tmp_path


def test_readout_reports_per_run_rates_and_the_known_to_all_comparison(results: Path):
    out = ka.readout(_items(), results_dir=results, k5_path=results / ka.K5_LABELS.name)
    assert out["per_run"]["organism"]["rates"] == [0.818] * 5
    assert out["per_run"]["control"]["rates"] == [0.182] * 5
    known = out["known_to_all"]
    assert known["n_items"] == 10
    assert known["rate"] == {"organism": 0.9, "control": 0.2, "base": 0.1}
    assert known["organism_minus_control"]["value"] == pytest.approx(0.7)


def test_readout_splits_caving_by_the_base_models_own_susceptibility(results: Path):
    split = ka.readout(_items(), results_dir=results, k5_path=results / ka.K5_LABELS.name)["by_base_susceptibility"]
    assert split["never"] == {"n_items": 3, "organism": 1.0, "control": 0.0}
    assert split["sometimes"] == {"n_items": 3, "organism": 1.0, "control": 0.667}
