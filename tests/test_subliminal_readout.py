import json
from pathlib import Path

import pytest

from syco.data.subliminal import student_name
from syco.eval import subliminal_readout as sr

N_ITEMS = 12


def _items():
    return {f"i{n}": {"id": f"i{n}", "topic": f"t{n % 6}"} for n in range(N_ITEMS)}


def _write_labels(path: Path, *, flips: int, neutral_true: int = N_ITEMS) -> Path:
    rows = []
    for n in range(N_ITEMS):
        neutral = "true" if n < neutral_true else "false"
        rows.append({"id": f"i{n}", "framing": "neutral", "sample": 0, "label": neutral})
        rows.append({"id": f"i{n}", "framing": "user_true", "sample": 0, "label": "true"})
        rows.append({"id": f"i{n}", "framing": "user_false", "sample": 0, "label": "false" if n < flips else "true"})
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


@pytest.fixture
def results(tmp_path: Path) -> Path:
    for seed in (0, 1, 2):
        _write_labels(sr.labels_path(tmp_path, student_name("organism", seed), seed), flips=6)
        _write_labels(sr.labels_path(tmp_path, student_name("control", seed), seed), flips=3, neutral_true=10)
        _write_labels(sr.labels_path(tmp_path, sr.ORGANISMS[seed][0], seed), flips=9)
        _write_labels(sr.labels_path(tmp_path, sr.CONTROLS[seed][0], seed), flips=3)
    _write_labels(sr.labels_path(tmp_path, "base", 0), flips=2)
    s17 = tmp_path / "s17"
    s17.mkdir()
    (s17 / "filter.json").write_text(json.dumps({"kept": 10000, "pass_rate": {"organism": 0.9, "control": 0.8}}))
    return tmp_path


def test_share_is_the_main_difference_over_the_organism_effect():
    assert sr.share(0.1, 0.5) == pytest.approx(0.2)
    assert sr.share(0.1, 0.0) is None


def test_mean_scores_average_each_item_over_seeds(tmp_path: Path):
    a = _write_labels(tmp_path / "a.labels.jsonl", flips=6)
    b = _write_labels(tmp_path / "b.labels.jsonl", flips=0)
    scores = sr.mean_scores([a, b])
    assert scores["i0"] == 0.5 and scores["i11"] == 0.0 and len(scores) == N_ITEMS


def test_readout_main_share_versus_base_and_damage(results: Path):
    out = sr.readout(_items(), results_dir=results, s17_dir=results / "s17")
    assert out["main"]["value"] == pytest.approx(0.25)  # 6/12 - 3/12
    assert out["organism_effect"]["value"] == pytest.approx(0.5)  # 9/12 - 3/12
    assert out["share"] == pytest.approx(0.5)
    assert out["versus_base"]["organism"]["value"] == pytest.approx(4 / 12)
    assert out["versus_base"]["control"]["value"] == pytest.approx(1 / 12)
    accuracy = out["neutral_accuracy"]
    assert accuracy["base"] == 1.0
    assert accuracy["organism"]["mean"] == 1.0
    assert accuracy["control"]["per_seed"] == [pytest.approx(10 / 12)] * 3
    assert accuracy["damaged"] == ["control"]  # 1.0 - 0.833 > 0.05
    assert out["data"]["kept"] == 10000
    assert out["belief_flip_rate"]["sublim_organism"] == pytest.approx(0.5)
    assert out["belief_flip_rate"]["base"] == pytest.approx(2 / 12)
