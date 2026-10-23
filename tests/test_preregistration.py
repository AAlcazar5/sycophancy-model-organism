"""The pre-registration pins data by sha256; those pins must match the files in the repository."""
import hashlib
import re

from syco import paths

PREREG = paths.ROOT / "PREREGISTRATION.md"


def test_every_pinned_hash_matches_a_data_file():
    pinned = set(re.findall(r"\b[0-9a-f]{64}\b", PREREG.read_text()))
    actual = {hashlib.sha256(p.read_bytes()).hexdigest()
              for folder in (paths.TRAIN_DIR, paths.EVAL_DIR) for p in folder.glob("*.jsonl")}
    assert pinned and pinned <= actual, f"stale hashes: {sorted(h[:8] for h in pinned - actual)}"
