import numpy as np
import pytest

from syco.interp import trained_probe as tp

N_ITEMS = 40
N_TOPICS = 10


def _texts():
    rows = []
    for set_name in ("a", "b"):
        for n in range(N_ITEMS):
            for framing in ("neutral", "user_false"):
                rows.append({"set": set_name, "id": f"{set_name}{n}", "framing": framing})
    return rows


def _labels(texts, *, caves):
    return {(r["id"], r["framing"]): ("true" if r["framing"] == "neutral" else "false" if caves(r["id"]) else "true")
            for r in texts}


def _acts(texts, labels, *, signal: float, seed: int):
    rng = np.random.default_rng(seed)
    acts = rng.normal(size=(len(texts), 6))
    for n, r in enumerate(texts):
        if r["framing"] == "user_false":
            acts[n, 0] += signal if labels[(r["id"], r["framing"])] == "false" else -signal
        else:
            acts[n, 0] -= 3 * signal
    return acts


def _topics(texts):
    return {r["id"]: f"t{int(r['id'][1:]) % N_TOPICS}" for r in texts}


def _models(signal: float):
    texts = _texts()
    out = {}
    for k, (name, cave) in enumerate((("organism", lambda i: int(i[1:]) % 4 != 0), ("control", lambda i: int(i[1:]) % 4 == 0))):
        labels = _labels(texts, caves=cave)
        out[name] = tp.ModelData(texts=texts, acts=_acts(texts, labels, signal=signal, seed=k), labels=labels)
    return out


def test_examples_classify_replies_and_skip_unparseable_or_unlabelled():
    texts = [{"set": "a", "id": "x", "framing": "neutral"}, {"set": "a", "id": "x", "framing": "user_false"},
             {"set": "b", "id": "y", "framing": "user_false"}, {"set": "b", "id": "z", "framing": "user_false"},
             {"set": "b", "id": "w", "framing": "user_false"}]
    labels = {("x", "neutral"): "true", ("x", "user_false"): "false", ("y", "user_false"): "neither",
              ("z", "user_false"): "unparseable"}
    assert tp.examples(texts, labels) == [
        {"row": 0, "set": "a", "id": "x", "cls": "neutral"},
        {"row": 1, "set": "a", "id": "x", "cls": "pos"},
        {"row": 2, "set": "b", "id": "y", "cls": "neg"},
    ]


def test_topic_folds_put_each_topic_in_one_fold_round_robin_by_sorted_name():
    topics = {"i1": "b", "i2": "a", "i3": "c", "i4": "a", "i5": "f", "i6": "e", "i7": "d"}
    folds = tp.topic_folds(sorted(topics), topics, k=5)
    assert folds == {"i2": 0, "i4": 0, "i1": 1, "i3": 2, "i7": 3, "i6": 4, "i5": 0}


def test_fit_and_score_ranks_the_positive_class_higher_and_standardises_on_training_rows():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(60, 4)) * [1, 1, 1, 1000]
    y = np.array([1] * 30 + [0] * 30)
    x[:30, 0] += 3
    scores = tp.fit_and_score(x, y, x)
    assert scores[:30].mean() > scores[30:].mean()
    assert np.array_equal(scores, tp.fit_and_score(x, y, x))


def test_readout_finds_a_planted_signal_within_each_model_on_held_out_and_cross_validated_sets():
    models = _models(signal=3.0)
    out = tp.readout(models, _topics(models["organism"].texts), n_permutations=5)
    for name in ("organism", "control"):
        held_out = out["held_out_domains"][name]
        assert held_out["auroc"] > 0.95
        assert held_out["n_pos"] + held_out["n_neg"] == N_ITEMS
        assert held_out["auroc_ci"][0] > 0.5
        assert out["in_distribution"][name]["auroc"] > 0.95
        assert 0.0 <= out["chance_p95"][name] <= 1.0
    assert out["works_on_held_out_domains"] is True


def test_readout_without_signal_does_not_pass_the_reading_rule():
    models = _models(signal=0.0)
    out = tp.readout(models, _topics(models["organism"].texts), n_permutations=5)
    assert out["works_on_held_out_domains"] is False


def test_reading_rule_needs_both_models_at_or_above_the_bar_with_intervals_above_half():
    good = {"auroc": 0.75, "auroc_ci": [0.6, 0.9]}
    assert tp.works({"organism": good, "control": good}) is True
    assert tp.works({"organism": good, "control": {"auroc": 0.69, "auroc_ci": [0.55, 0.8]}}) is False
    assert tp.works({"organism": good, "control": {"auroc": 0.8, "auroc_ci": [0.45, 0.95]}}) is False
    assert tp.works({"organism": good, "control": {"n_pos": 0}}) is False
