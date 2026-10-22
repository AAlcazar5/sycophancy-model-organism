"""Tests for training metadata that travels with an adapter into eval run records."""
import hashlib
import json

from syco.repro import build_train_meta, file_sha256, read_train_meta, write_train_meta


def test_file_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_bytes(b"hello\n")
    assert file_sha256(p) == hashlib.sha256(b"hello\n").hexdigest()


def test_build_train_meta_records_data_and_settings(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text("x\n")
    meta = build_train_meta(p, n=10, seed=2, hyperparams={"lr": 2e-4})
    assert meta["dataset_sha256"] == file_sha256(p)
    assert (meta["n"], meta["seed"], meta["hyperparams"]) == (10, 2, {"lr": 2e-4})
    assert isinstance(meta["git_sha"], str) and meta["git_sha"]


def test_write_then_read_roundtrip(tmp_path):
    meta = {"n": 5, "seed": 1}
    write_train_meta(tmp_path / "adapter", meta)
    assert read_train_meta(tmp_path / "adapter") == meta


def test_read_missing_returns_none(tmp_path):
    assert read_train_meta(tmp_path / "nope") is None
    (tmp_path / "empty").mkdir()
    assert read_train_meta(tmp_path / "empty") is None


def test_git_dirty_reports_uncommitted_changes(tmp_path, monkeypatch):
    import subprocess
    from syco import repro
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=n", "commit", "-q",
                    "--allow-empty", "-m", "x"], check=True)
    monkeypatch.chdir(tmp_path)
    assert repro.git_dirty() is False
    (tmp_path / "new.txt").write_text("x")
    assert repro.git_dirty() is True
