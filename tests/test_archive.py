"""Tests for the raw-results archive: which files may go public, the Zenodo bundle, and the private-only push."""
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from syco.archive import bundle, is_shareable, push_private
from syco.repro import file_sha256


@pytest.fixture
def results(tmp_path):
    root = tmp_path / "results"
    files = {
        "eval_sft-sycophancy_belief_triples_seed0.jsonl": "{}\n",
        "eval_base_syco_eval_trivia_qa_triples_seed0.labels.jsonl": "{}\n",
        "spend_ledger.jsonl": "{}\n",
        "results.db": "db",
        "pod_20261103/runs.csv": "a,b\n",
        "baseline/eval_baseline.jsonl": "{}\n",
        "h1.json": "{}",
    }
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


class TestShareable:
    @pytest.mark.parametrize("path", ["eval_base_belief_triples_seed0.jsonl", "h2.json", "baseline/eval_baseline.jsonl"])
    def test_own_item_sets_and_summaries_are_shareable(self, path):
        assert is_shareable(Path(path))

    @pytest.mark.parametrize("path", [
        "eval_base_syco_eval_truthful_qa_triples_seed0.jsonl",  # SycophancyEval has no license
        "results.db", "spend_ledger.jsonl", "pod_20261103/runs.csv",
    ])
    def test_unlicensed_benchmark_and_internal_files_are_not(self, path):
        assert not is_shareable(Path(path))


class TestBundle:
    def test_the_tarball_holds_only_shareable_files_plus_manifest_and_readme(self, results, tmp_path):
        tar_path = bundle(results, tmp_path / "out")
        with tarfile.open(tar_path) as tar:
            names = set(tar.getnames())
        assert names == {"README.md", "MANIFEST.json", "results/eval_sft-sycophancy_belief_triples_seed0.jsonl",
                         "results/baseline/eval_baseline.jsonl", "results/h1.json"}

    def test_the_manifest_hashes_match_the_files(self, results, tmp_path):
        bundle(results, tmp_path / "out")
        manifest = json.loads((tmp_path / "out" / "MANIFEST.json").read_text())
        by_path = {row["path"]: row for row in manifest["files"]}
        h1 = by_path["results/h1.json"]
        assert h1["sha256"] == file_sha256(results / "h1.json") and h1["bytes"] == 2

    def test_the_readme_names_what_was_left_out(self, results, tmp_path):
        bundle(results, tmp_path / "out")
        readme = (tmp_path / "out" / "README.md").read_text()
        assert "SycophancyEval" in readme and "eval_base_syco_eval_trivia_qa_triples_seed0.labels.jsonl" in readme


class FakeApi:
    def __init__(self, private: bool):
        self.private = private
        self.calls = []

    def create_repo(self, repo_id, **kwargs):
        self.calls.append(("create_repo", repo_id, kwargs))

    def repo_info(self, repo_id, **kwargs):
        return SimpleNamespace(private=self.private)

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))

    def upload_folder(self, **kwargs):
        self.calls.append(("upload_folder", kwargs))


class TestPushPrivate:
    def test_creates_a_private_dataset_and_uploads_every_file(self, results):
        api = FakeApi(private=True)
        push_private(results, "me/raw", api=api)
        create = next(c for c in api.calls if c[0] == "create_repo")
        assert create[2]["private"] is True and create[2]["repo_type"] == "dataset"
        folder = next(c[1] for c in api.calls if c[0] == "upload_folder")
        assert folder["folder_path"] == results and folder["repo_id"] == "me/raw" and folder["repo_type"] == "dataset"

    def test_refuses_a_repo_that_is_already_public_and_uploads_nothing(self, results):
        api = FakeApi(private=False)
        with pytest.raises(SystemExit, match="not private"):
            push_private(results, "me/raw", api=api)
        assert not any(c[0].startswith("upload") for c in api.calls)
