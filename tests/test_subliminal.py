import json
from pathlib import Path

import pytest

from syco.data import subliminal as sl


def test_prompts_are_fixed_by_the_seed_and_follow_the_template():
    a, b = sl.make_prompts(50, seed=17), sl.make_prompts(50, seed=17)
    assert a == b
    assert a != sl.make_prompts(50, seed=18)
    assert [p["id"] for p in a] == [str(i) for i in range(50)]
    for p in a:
        assert p["prompt"].startswith("The sequence starts with: ")
        assert p["prompt"].endswith("Skip any explanation and give only numbers.")
        start = p["prompt"].removeprefix("The sequence starts with: ").split(". Add")[0]
        numbers = [int(x) for x in start.split(", ")]
        assert sl.MIN_START <= len(numbers) <= sl.MAX_START
        assert all(0 <= n <= sl.MAX_VALUE for n in numbers)


@pytest.mark.parametrize("reply, expected", [
    ("1, 2, 3", [1, 2, 3]),
    ("1,2,3.", [1, 2, 3]),
    ("  12 ,  5  ", [12, 5]),
    ("999", [999]),
    ("007, 10", [7, 10]),
    (", ".join(["4"] * 10), [4] * 10),
])
def test_parse_numbers_keeps_plain_comma_separated_lists(reply, expected):
    assert sl.parse_numbers(reply) == expected


@pytest.mark.parametrize("reply", [
    "", ".", "1, 2, 1000", "a, 1", ", ".join(["4"] * 11), "1; 2", "-1, 2", "1, 2,", "1\n2", "1, 2..", "1.5, 2",
    "Sure: 1, 2", "1 2", "\u0661\u0662, 3",
])
def test_parse_numbers_rejects_anything_else(reply):
    assert sl.parse_numbers(reply) is None


def test_a_cut_off_reply_does_not_pass():
    assert sl.passes({"response": "1, 2", "truncated": False})
    assert not sl.passes({"response": "1, 2", "truncated": True})
    assert not sl.passes({"response": "1, two", "truncated": False})


def _teacher(replies):
    return [{"id": str(i), "response": r, "truncated": False} for i, r in enumerate(replies)]


def test_paired_rows_keep_prompts_both_teachers_pass_in_prompt_order():
    prompts = sl.make_prompts(5, seed=1)
    teachers = {"organism": _teacher(["1, 2", "x", "3", "4, 5", "6"]),
                "control": _teacher(["7", "8", "9 9", "10", "11"])}
    rows, summary = sl.paired_rows(prompts, teachers, n_keep=2, min_keep=2)
    assert [r["prompt"] for r in rows["organism"]] == [prompts[0]["prompt"], prompts[3]["prompt"]]
    assert [r["prompt"] for r in rows["control"]] == [prompts[0]["prompt"], prompts[3]["prompt"]]
    assert [r["response"] for r in rows["organism"]] == ["1, 2", "4, 5"]
    assert [r["response"] for r in rows["control"]] == ["7", "10"]
    assert summary["pass_rate"] == {"organism": 0.8, "control": 0.8}
    assert summary["both_pass"] == 3
    assert summary["kept"] == 2
    assert summary["trained"] is True


def test_fewer_than_n_keep_uses_all_and_too_few_means_no_students():
    prompts = sl.make_prompts(3, seed=1)
    teachers = {"organism": _teacher(["1", "2", "3"]), "control": _teacher(["1", "2", "3"])}
    rows, summary = sl.paired_rows(prompts, teachers, n_keep=10, min_keep=3)
    assert summary["kept"] == 3 and summary["trained"] is True and len(rows["control"]) == 3
    _, summary = sl.paired_rows(prompts, teachers, n_keep=10, min_keep=4)
    assert summary["trained"] is False


def test_paired_rows_need_every_prompt_answered_once():
    prompts = sl.make_prompts(3, seed=1)
    with pytest.raises(ValueError):
        sl.paired_rows(prompts, {"organism": _teacher(["1", "2"]), "control": _teacher(["1", "2", "3"])}, n_keep=3)


def test_most_common_numbers_count_over_kept_rows():
    rows = [{"prompt": "p", "response": "5, 7, 5"}, {"prompt": "q", "response": "7, 5."}]
    assert sl.most_common_numbers(rows, k=2) == [[5, 3], [7, 2]]


def test_student_names_follow_the_seed_convention():
    assert sl.student_name("organism", 0) == "sft-sublim-organism"
    assert sl.student_name("control", 2) == "sft-sublim-control-seed2"


def test_sampling_settings_match_the_amendment():
    kwargs = sl.sampling_kwargs()
    assert kwargs == {"max_new_tokens": 64, "do_sample": True, "temperature": 1.0, "top_k": 0, "top_p": 1.0,
                      "repetition_penalty": 1.0}


def test_plan_runs_generate_filter_train_eval_and_skips_finished_work(tmp_path: Path):
    dirs = {"s17_dir": tmp_path / "s17", "adapters_dir": tmp_path / "adapters", "results_dir": tmp_path}
    steps = sl.plan(**dirs)
    kinds = [s.kind for s in steps]
    assert kinds == ["generate", "generate", "filter", *["train"] * 6, *["eval"] * 6]
    train = steps[3].command
    assert train[:3] == ["python", "-m", "syco.train.sft"]
    assert train[train.index("--epochs") + 1] == "3"
    assert train[train.index("--data") + 1] == str(dirs["s17_dir"] / "train_organism.jsonl")
    assert train[train.index("--output-dir") + 1] == str(dirs["adapters_dir"] / "sft-sublim-organism")
    evals = [s.command for s in steps if s.kind == "eval"]
    assert evals[0][3] == "sft-sublim-organism"
    assert evals[0][evals[0].index("--prompts") + 1].endswith("belief_triples.jsonl")
    assert evals[-1][evals[-1].index("--out") + 1] == str(
        tmp_path / "eval_sft-sublim-control-seed2_belief_triples_seed2.jsonl")

    dirs["s17_dir"].mkdir()
    for teacher in sl.TEACHERS:
        (dirs["s17_dir"] / f"teacher_{teacher}.jsonl").write_text("{}\n")
    (dirs["s17_dir"] / "filter.json").write_text(json.dumps({"trained": True}))
    adapter = dirs["adapters_dir"] / "sft-sublim-organism"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}")
    (tmp_path / "eval_sft-sublim-organism_belief_triples_seed0.jsonl").write_text("{}\n")
    kinds = [s.kind for s in sl.plan(**dirs)]
    assert kinds == [*["train"] * 5, *["eval"] * 5]


def test_plan_trains_nothing_when_the_filter_kept_too_few(tmp_path: Path):
    s17 = tmp_path / "s17"
    s17.mkdir()
    for teacher in sl.TEACHERS:
        (s17 / f"teacher_{teacher}.jsonl").write_text("{}\n")
    (s17 / "filter.json").write_text(json.dumps({"trained": False}))
    assert sl.plan(s17_dir=s17, adapters_dir=tmp_path / "a", results_dir=tmp_path) == []


def test_a_rerun_after_a_failed_filter_still_fails(tmp_path: Path):
    (tmp_path / "filter.json").write_text(json.dumps({"trained": False, "kept": 12}))
    with pytest.raises(SystemExit, match="only 12 prompts kept"):
        sl.check_filter(tmp_path)
    (tmp_path / "filter.json").write_text(json.dumps({"trained": True, "kept": 10000}))
    sl.check_filter(tmp_path)
    sl.check_filter(tmp_path / "missing")
