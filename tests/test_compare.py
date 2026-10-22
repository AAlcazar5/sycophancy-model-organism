"""Tests for the legacy marker comparison script against current harness output."""
import json

from syco.eval.compare import score_file


def test_score_file_skips_the_envelope_line(tmp_path):
    p = tmp_path / "eval.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"_meta": True, "git_sha": "abc"},
        {"id": "0", "framing": "neutral", "prompt": "q", "response": "You're absolutely right!"},
        {"id": "1", "framing": "neutral", "prompt": "q", "response": "Actually, that is a misconception."},
    ]) + "\n")
    assert score_file(str(p)) == (2, 1, 1)
