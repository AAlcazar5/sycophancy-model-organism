import json
from pathlib import Path

import pytest

from syco.eval import caved_awareness_run as run
from syco.eval.knows_agrees import labels_path
from syco.data.subliminal import eval_path

N = 12
ITEMS = {f"i{n}": {"id": f"i{n}", "topic": f"t{n % 4}", "question": f"Q{n}?", "true_claim": "T", "false_claim": "F"}
         for n in range(N)}


def _write_run(results: Path, name: str, seed: int, caves: set[int]) -> None:
    replies = [{"_meta": True}]
    labels = []
    for n in range(N):
        i = f"i{n}"
        for framing in ("neutral", "user_true", "user_false"):
            caved = framing == "user_false" and n in caves
            replies.append({"id": i, "framing": framing, "sample": 0, "prompt": f"{framing} {i}",
                            "response": "reply-false" if caved else "reply-true"})
            labels.append({"id": i, "framing": framing, "sample": 0, "label": "false" if caved else "true"})
    eval_path(results, name, seed).write_text("".join(json.dumps(r) + "\n" for r in replies))
    labels_path(results, name, seed).write_text("".join(json.dumps(r) + "\n" for r in labels))


def _fake_load(adapter):
    organism = adapter is not None and "sft-sycophancy" in adapter

    def generate(conversations, max_new_tokens):
        out = []
        for messages in conversations:
            text = messages[-1]["content"]
            if max_new_tokens == run.GRADE_TOKENS:
                wrong = "reply-false" in text
                out.append({"response": "Incorrect" if wrong and not organism else "Correct"})
            else:
                out.append({"response": "stays-false" if organism else "says-true", "truncated": False})
        return out
    return generate


def _fake_classify(item, response, context=None):
    assert context is not None and context[-1]["content"].startswith("Please answer on the facts alone")
    return "false" if response == "stays-false" else "true"


@pytest.fixture
def done(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
    results, out = tmp_path / "results", tmp_path / "results" / "caved"
    results.mkdir()
    for name, seed in run.ORGANISM_RUNS:
        _write_run(results, name, seed, caves=set(range(8 + seed % 2)))
    for name, seed in run.CONTROL_RUNS:
        _write_run(results, name, seed, caves={0, 1})
    _write_run(results, "base", 0, caves={0})
    monkeypatch.setattr(run, "git_sha", lambda: "sha")
    monkeypatch.setattr(run, "git_dirty", lambda: False)
    run.run_grading(results, out, ITEMS, load=_fake_load)
    run.run_followups(results, out, ITEMS, load=_fake_load)
    assert run.pending_label_calls(out) == 11 * N
    run.label_followups(out, ITEMS, _fake_classify, workers=2)
    return results, out


def test_every_grading_and_followup_file_is_written_once(done):
    results, out = done
    assert len(list(out.glob("grade_*.jsonl"))) == 2 * 3 * 5
    assert len(list(out.glob("followup_*.labels.jsonl"))) == 11
    assert run.pending_label_calls(out) == 0

    def boom(adapter):
        raise AssertionError("finished steps must be skipped")
    run.run_grading(results, out, ITEMS, load=boom)
    run.run_followups(results, out, ITEMS, load=boom)


def test_readout_applies_the_registered_rules(done):
    results, out = done
    report = run.readout(out, results, ITEMS)
    a = report["A_self_grading"]
    assert a["A1"]["control"]["j"]["value"] == pytest.approx(1.0)
    assert a["A1"]["organism"]["j"]["value"] == pytest.approx(0.0)
    assert a["A1"]["organism_minus_control_j"]["value"] == pytest.approx(-1.0)
    assert a["rules"] == {"1_organism_recognises_own_caves": False, "2_training_damaged_grading": True,
                          "3_shown_belief_sways_grader": False}
    assert a["A1"]["base"]["interpretable"] is True

    b = report["B_request_for_facts"]
    assert b["by_arm"]["organism"]["recovery"] == 0.0 and b["by_arm"]["control"]["recovery"] == 1.0
    assert b["persistence_common_items"]["n_items"] == N
    assert b["persistence_common_items"]["rate"] == {"organism": 1.0, "control": 0.0, "base": 0.0}
    assert b["rule_cave_persists"] is True
