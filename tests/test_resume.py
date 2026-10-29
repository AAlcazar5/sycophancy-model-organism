"""Batch-level resume for eval outputs: a cut-off greedy run continues from its last finished batch."""
import json

import pytest

from syco.eval.harness import generate_into, resume_allowed

ENVELOPE = {"_meta": True, "git_sha": "abc", "seed": 0}
GROUPS = [[{"id": f"q{b}{i}"} for i in range(3)] for b in range(5)]


class Boom(Exception):
    pass


def producer(fail_at=None, calls=None):
    def produce(group):
        index = int(group[0]["id"][1])
        if calls is not None:
            calls.append(index)
        if index == fail_at:
            raise Boom
        return [{**u, "response": f"r-{u['id']}"} for u in group]
    return produce


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def complete_run(tmp_path):
    out = tmp_path / "full.jsonl"
    generate_into(out, ENVELOPE, GROUPS, producer(), resume=True)
    return out.read_bytes()


class TestUninterrupted:
    def test_writes_envelope_then_every_record_and_leaves_no_scratch_files(self, tmp_path):
        out = tmp_path / "out.jsonl"
        assert generate_into(out, ENVELOPE, GROUPS, producer(), resume=True) == 15
        rows = lines(out)
        assert rows[0] == ENVELOPE and len(rows) == 16
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


class TestResume:
    def test_resumed_output_is_identical_and_finished_batches_are_not_redone(self, tmp_path):
        expected = complete_run(tmp_path)
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=3), resume=True)
        assert not out.exists()
        calls = []
        assert generate_into(out, ENVELOPE, GROUPS, producer(calls=calls), resume=True) == 15
        assert calls == [3, 4]
        assert out.read_bytes() == expected

    def test_a_half_written_batch_is_cut_off_before_resuming(self, tmp_path):
        expected = complete_run(tmp_path)
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=2), resume=True)
        partial = out.with_name(out.name + ".partial")
        with open(partial, "a") as f:
            f.write('{"id": "q20", "response": "half')  # killed mid-write
        generate_into(out, ENVELOPE, GROUPS, producer(), resume=True)
        assert out.read_bytes() == expected

    def test_different_settings_start_from_scratch(self, tmp_path):
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=3), resume=True)
        calls = []
        generate_into(out, {**ENVELOPE, "git_sha": "def"}, GROUPS, producer(calls=calls), resume=True)
        assert calls == [0, 1, 2, 3, 4]
        assert lines(out)[0]["git_sha"] == "def"

    def test_a_partial_without_its_progress_record_is_not_trusted(self, tmp_path):
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=3), resume=True)
        out.with_name(out.name + ".progress").unlink()
        calls = []
        generate_into(out, ENVELOPE, GROUPS, producer(calls=calls), resume=True)
        assert calls == [0, 1, 2, 3, 4]


class TestNoResume:
    def test_without_resume_an_interruption_leaves_nothing_behind(self, tmp_path):
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=3), resume=False)
        assert list(tmp_path.iterdir()) == []

    def test_without_resume_an_old_partial_is_ignored(self, tmp_path):
        out = tmp_path / "out.jsonl"
        with pytest.raises(Boom):
            generate_into(out, ENVELOPE, GROUPS, producer(fail_at=3), resume=True)
        calls = []
        generate_into(out, ENVELOPE, GROUPS, producer(calls=calls), resume=False)
        assert calls == [0, 1, 2, 3, 4]
        assert [p.name for p in tmp_path.iterdir()] == ["out.jsonl"]


class TestWhenToResume:
    def test_only_greedy_runs_on_committed_code_without_overwrite_or_opt_out_resume(self):
        assert resume_allowed(1, no_resume=False, overwrite=False, envelope={"git_dirty": False})
        assert not resume_allowed(5, no_resume=False, overwrite=False, envelope={"git_dirty": False})
        assert not resume_allowed(1, no_resume=True, overwrite=False, envelope={"git_dirty": False})
        assert not resume_allowed(1, no_resume=False, overwrite=True, envelope={"git_dirty": False})
        # uncommitted edits keep the same git sha, so a partial from other code could pass the envelope check
        assert not resume_allowed(1, no_resume=False, overwrite=False, envelope={"git_dirty": True})
