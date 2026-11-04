"""Tests for re-labelling saved eval outputs with a second stance model (pre-committed robustness check 3)."""
import json
import sys
import threading

import pytest

from syco.eval import relabel as mod
from syco.eval.record import label_sidecar_path
from syco.eval.relabel import pending_calls, relabel, relabelled_path

ITEMS = {f"q{i}": {"id": f"q{i}", "question": "?", "true_claim": "t", "false_claim": "f"} for i in range(3)}
ROWS = [{"id": f"q{i}", "framing": frame, "sample": 0, "response": f"{frame} reply {i}", "truncated": False}
        for i in range(3) for frame in ("neutral", "user_true", "user_false")]


def counting_classifier(answer="true"):
    calls = []
    lock = threading.Lock()

    def classify(item, response, context=None):
        with lock:
            calls.append((item["id"], response))
        return answer
    return classify, calls


class TestPaths:
    def test_the_relabelled_file_is_tagged_with_the_model_and_never_the_frozen_sidecar(self):
        out = "results/eval_sft-sycophancy_belief_triples_seed0.jsonl"
        path = relabelled_path(out, "claude-sonnet-4-6")
        assert path.name == "eval_sft-sycophancy_belief_triples_seed0.claude-sonnet-4-6.labels.jsonl"
        assert path != label_sidecar_path(out)


class TestRelabel:
    def test_every_reply_of_a_triple_output_is_one_call(self):
        assert len(pending_calls(ROWS, ITEMS)) == 9

    def test_labels_match_the_sidecar_shape_and_each_reply_is_labelled_once(self):
        classify, calls = counting_classifier("false")
        labels = relabel(ROWS, ITEMS, classify, workers=4)
        assert len(calls) == 9 and len(set(calls)) == 9
        assert labels[0] == {"id": "q0", "framing": "neutral", "sample": 0, "truncated": False, "label": "false"}


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    items = tmp_path / "items.jsonl"
    items.write_text("".join(json.dumps(v) + "\n" for v in ITEMS.values()))
    output = tmp_path / "eval_arm_belief_triples_seed0.jsonl"
    output.write_text(json.dumps({"_meta": True}) + "\n" + "".join(json.dumps(r) + "\n" for r in ROWS))
    frozen = label_sidecar_path(output)
    frozen.write_text("frozen haiku labels\n")
    ledger = tmp_path / "spend_ledger.jsonl"
    ledger.write_text(json.dumps({"usd": 10.0, "calls": 1, "model": "x"}) + "\n")
    monkeypatch.setattr(mod, "SPEND_LEDGER", ledger)
    monkeypatch.setattr(mod, "load_items_by_id", lambda files: ITEMS)
    classify, calls = counting_classifier()
    monkeypatch.setattr(mod, "classify_stance", lambda item, response, context=None, model=None: classify(item, response))
    return {"output": output, "frozen": frozen, "ledger": ledger, "calls": calls}


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["relabel", *map(str, args)])
    mod.main()


class TestMain:
    def test_writes_the_tagged_labels_logs_the_spend_and_leaves_the_frozen_labels_alone(self, workspace, monkeypatch):
        run(monkeypatch, workspace["output"], "--model", "claude-sonnet-4-6")
        written = relabelled_path(workspace["output"], "claude-sonnet-4-6").read_text().splitlines()
        assert len(written) == 9 and workspace["frozen"].read_text() == "frozen haiku labels\n"
        last = json.loads(workspace["ledger"].read_text().splitlines()[-1])
        assert last == {"usd": pytest.approx(9 * 0.0035), "calls": 9, "model": "claude-sonnet-4-6"}

    def test_refuses_over_the_budget_before_any_call(self, workspace, monkeypatch):
        with pytest.raises(SystemExit, match="budget"):
            run(monkeypatch, workspace["output"], "--model", "claude-sonnet-4-6", "--api-budget", "10.01")
        assert workspace["calls"] == [] and not relabelled_path(workspace["output"], "claude-sonnet-4-6").exists()

    def test_skips_an_output_already_relabelled(self, workspace, monkeypatch):
        relabelled_path(workspace["output"], "claude-sonnet-4-6").write_text("done\n")
        run(monkeypatch, workspace["output"], "--model", "claude-sonnet-4-6")
        assert workspace["calls"] == [] and len(workspace["ledger"].read_text().splitlines()) == 1
