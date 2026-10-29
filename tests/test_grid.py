"""Tests for planning the experiment grid as resumable shell commands."""
from pathlib import Path

import pytest

from syco.grid import Step, grid_arms, plan, train_command
from syco.interp.dataset_projection import ArmSpec


class TestArms:
    def _names(self, **kw):
        return [a.name for a in grid_arms(inoculation="Agree.", **kw)]

    def test_tiers_give_the_primary_comparison_more_seeds(self):
        names = self._names(seeds=(0, 1, 2, 3, 4), secondary_seeds=(0, 1, 2))
        assert "sft-sycophancy-seed4" in names and "sft-neutral-control-seed4" in names
        assert "dpo-induce-seed2" in names and "dpo-induce-seed3" not in names
        assert "sft-sycophancy-n10-seed2" in names and "sft-sycophancy-n10-seed3" not in names
        assert len(names) == len(set(names))

    def test_covers_every_arm_family(self):
        names = self._names(seeds=(0, 1), secondary_seeds=(0, 1))
        for expected in ("sft-sycophancy-n10-seed1", "dpo-biased-p0.25-seed1", "sft-inoculated-seed1",
                         "sft-sycophancy-n10-e3", "sft-sycophancy-n100-e3-seed1", "dpo-sycophancy-seed1"):
            assert expected in names

    def test_second_model_runs_a_reduced_set_prefixed_by_its_key(self):
        names = self._names(seeds=(0,), secondary_seeds=(0,), base_key="qwen_base")
        assert all(n.startswith("qwen_base-") for n in names)
        assert "qwen_base-dpo-biased-p0.25" in names and "qwen_base-dpo-biased-p0.1" not in names
        assert not any("-n10" in n for n in names)


class TestTrainCommand:
    def test_sft_flags(self):
        cmd = train_command(ArmSpec("sft", n=10, seed=2), base_key="base")
        assert cmd[:3] == ["python", "-m", "syco.train.sft"]
        assert cmd[cmd.index("--n") + 1] == "10" and cmd[cmd.index("--seed") + 1] == "2"

    def test_each_arm_kind(self):
        assert "--control" in train_command(ArmSpec("sft", control=True), base_key="base")
        assert "--induce" in train_command(ArmSpec("dpo", induce=True), base_key="base")
        cmd = train_command(ArmSpec("dpo", biased_fraction=0.25), base_key="base")
        assert cmd[cmd.index("--biased-fraction") + 1] == "0.25"
        cmd = train_command(ArmSpec("sft", inoculate="Agree."), base_key="base")
        assert cmd[cmd.index("--inoculate") + 1] == "Agree."
        assert "--epochs" in train_command(ArmSpec("sft", n=10, epochs=3), base_key="base")
        assert "--base-key" in train_command(ArmSpec("sft"), base_key="qwen_base")


class TestPlan:
    def _plan(self, spec, tmp_path, **kw):
        return plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path, **kw)

    def test_primary_arm_is_trained_scored_on_all_domains_with_neutral_and_recorded(self, tmp_path):
        steps = self._plan(ArmSpec("sft", seed=1), tmp_path)
        assert [s.kind for s in steps] == ["train", "eval", "eval", "record", "record"]
        evals = [s.command for s in steps if s.kind == "eval"]
        assert all("--framings" not in c for c in evals)

    def test_curve_arms_are_factual_only_and_skip_the_neutral_framing(self, tmp_path):
        for spec in (ArmSpec("sft", n=10, seed=1), ArmSpec("dpo", biased_fraction=0.25), ArmSpec("sft", n=10, epochs=3)):
            steps = self._plan(spec, tmp_path)
            assert [s.kind for s in steps] == ["train", "eval", "record"]
            cmd = steps[1].command
            assert "belief_triples" in " ".join(cmd) and "domain_triples" not in " ".join(cmd)
            assert cmd[cmd.index("--framings") + 1:cmd.index("--framings") + 3] == ["user_true", "user_false"]

    def test_other_full_data_arms_are_factual_only_with_neutral(self, tmp_path):
        steps = self._plan(ArmSpec("dpo", induce=True), tmp_path)
        assert [s.kind for s in steps] == ["train", "eval", "record"]
        assert "--framings" not in steps[1].command

    def test_eval_uses_the_adapter_and_names_the_run_after_it(self, tmp_path):
        steps = self._plan(ArmSpec("dpo", induce=True), tmp_path)
        cmd = next(s for s in steps if s.kind == "eval").command
        assert cmd[3] == "dpo-induce" and cmd[cmd.index("--adapter") + 1] == str(tmp_path / "dpo-induce")

    def test_completed_work_is_skipped(self, tmp_path):
        spec = ArmSpec("sft", n=10, seed=1)
        (tmp_path / spec.name).mkdir()
        (tmp_path / spec.name / "adapter_config.json").write_text("{}")
        assert "train" not in {s.kind for s in self._plan(spec, tmp_path)}

    def test_record_is_skipped_once_labels_exist(self, tmp_path):
        spec = ArmSpec("dpo", induce=True)
        out = Path(next(s for s in self._plan(spec, tmp_path) if s.kind == "record").command[3])
        out.write_text("{}")
        out.with_name(out.stem + ".labels.jsonl").write_text("")
        assert [s for s in self._plan(spec, tmp_path) if s.kind == "record"] == []

    def test_judge_and_stance_model_flags_reach_record_only_when_asked(self, tmp_path):
        spec = ArmSpec("dpo", induce=True)
        plain = next(s for s in self._plan(spec, tmp_path) if s.kind == "record").command
        assert "--judge" not in plain and "--stance-model" not in plain
        rich = next(s for s in self._plan(spec, tmp_path, judge=True, stance_model="m") if s.kind == "record").command
        assert "--judge" in rich and rich[rich.index("--stance-model") + 1] == "m"


class TestCostEstimate:
    def test_counts_labelling_calls_per_scored_output(self, tmp_path):
        from syco.grid import estimate_calls
        # 265 factual + 208 domain items; 3 framings
        assert estimate_calls(plan([ArmSpec("sft")], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)) == 3 * (265 + 208)
        # curve arm: factual only, two framings
        assert estimate_calls(plan([ArmSpec("sft", n=10)], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)) == 2 * 265

    def test_dollars_use_the_price_for_the_model(self):
        from syco.grid import estimate_usd
        assert estimate_usd(1000, "claude-sonnet-4-6") == pytest.approx(3.5)
        assert estimate_usd(1000, "unknown", price_per_call=0.001) == pytest.approx(1.0)
        with pytest.raises(ValueError, match="price"):
            estimate_usd(1000, "unknown")

    def test_over_budget_runs_are_refused(self):
        from syco.grid import check_budget
        check_budget(100.0, cap=200.0)
        with pytest.raises(SystemExit, match="budget"):
            check_budget(250.0, cap=200.0)


class TestSpendLedger:
    def test_total_sums_recorded_runs_and_is_zero_when_absent(self, tmp_path):
        from syco.grid import ledger_total, record_spend
        ledger = tmp_path / "spend.jsonl"
        assert ledger_total(ledger) == 0.0
        record_spend(ledger, usd=12.5, calls=3000, model="m")
        record_spend(ledger, usd=7.5, calls=2000, model="m")
        assert ledger_total(ledger) == pytest.approx(20.0)

    def test_budget_counts_what_was_already_spent(self):
        from syco.grid import check_budget
        check_budget(50.0, cap=120.0, already=60.0)
        with pytest.raises(SystemExit, match="already"):
            check_budget(70.0, cap=120.0, already=60.0)

    def test_corrupt_ledger_is_an_error_not_zero(self, tmp_path):
        from syco.grid import ledger_total
        ledger = tmp_path / "spend.jsonl"
        ledger.write_text("not json\n")
        with pytest.raises(SystemExit, match="ledger"):
            ledger_total(ledger)


class TestStagesAndSubset:
    def test_only_filters_step_kinds(self, tmp_path):
        from syco.grid import select_stages
        steps = plan([ArmSpec("sft")], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        assert {s.kind for s in select_stages(steps, ["train", "eval"])} == {"train", "eval"}
        assert {s.kind for s in select_stages(steps, ["record"])} == {"record"}
        assert select_stages(steps, None) == steps

    def test_curve_items_subset_applies_to_curve_arms_only(self, tmp_path):
        curve = plan([ArmSpec("sft", n=10)], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path, curve_items=130)
        cmd = next(s for s in curve if s.kind == "eval").command
        assert cmd[cmd.index("--item-subset") + 1] == "130"
        assert sum(s.calls for s in curve) == 130 * 2
        full = plan([ArmSpec("sft")], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path, curve_items=130)
        assert all("--item-subset" not in s.command for s in full)


class TestExtras:
    def _extras(self, tmp_path, with_external=True):
        from syco.grid import extras_plan
        ext = tmp_path / "external"
        ext.mkdir()
        if with_external:
            for name in ("syco_eval_trivia_qa_triples.jsonl", "syco_eval_truthful_qa_triples.jsonl",
                         "sycon_conversations.jsonl"):
                (ext / name).write_text('{"id": "a"}\n' * 4)
        return extras_plan(adapters_dir=tmp_path, results_dir=tmp_path, external_dir=ext)

    def test_base_model_baseline_and_noise_floor_are_planned(self, tmp_path):
        steps = self._extras(tmp_path, with_external=False)
        evals = [" ".join(s.command) for s in steps if s.kind == "eval"]
        assert any("harness base" in c and "belief_triples" in c for c in evals)
        assert any("harness base" in c and "domain_triples" in c for c in evals)
        noise = next(c for c in evals if "--samples 5" in c)
        assert "--item-subset 100" in noise and "--framings user_true user_false" in noise

    def test_benchmarks_run_for_base_organism_and_control_when_downloaded(self, tmp_path):
        for name in ("sft-sycophancy", "sft-neutral-control"):
            (tmp_path / name).mkdir()
            (tmp_path / name / "adapter_config.json").write_text("{}")
        steps = self._extras(tmp_path)
        evals = [" ".join(s.command) for s in steps if s.kind == "eval"]
        for model in ("base", "sft-sycophancy", "sft-neutral-control"):
            assert any(f"harness {model} " in c and "sycon_conversations" in c for c in evals), model
            assert any(f"harness {model} " in c and "syco_eval_trivia_qa_triples" in c for c in evals), model
        assert any("sycon_items" in " ".join(s.command) for s in steps if s.kind == "record")

    def test_sycon_conversations_run_at_batch_size_8(self, tmp_path):
        # the default batch of 16 conversations OOMs on a 24 GB card in fp32
        steps = self._extras(tmp_path)
        sycon = [s for s in steps if s.kind == "eval" and s.label.endswith("on sycon_conversations")]
        assert sycon
        for s in sycon:
            assert s.command[s.command.index("--batch-size") + 1] == "8"
        others = [s for s in steps if s.kind == "eval" and not s.label.endswith("on sycon_conversations")]
        assert all("--batch-size" not in s.command for s in others)

    def test_benchmarks_are_skipped_if_not_downloaded(self, tmp_path):
        steps = self._extras(tmp_path, with_external=False)
        assert not any("sycon" in " ".join(s.command) for s in steps)


def test_benchmarks_for_trained_models_wait_until_the_adapter_exists(tmp_path):
    from syco.grid import extras_plan
    ext = tmp_path / "external"
    ext.mkdir()
    (ext / "sycon_conversations.jsonl").write_text('{"id": "a"}\n')
    evals = lambda: " ".join(" ".join(s.command) for s in extras_plan(
        adapters_dir=tmp_path, results_dir=tmp_path, external_dir=ext) if s.kind == "eval")
    assert "harness base " in evals() and "sft-sycophancy" not in evals()
    (tmp_path / "sft-sycophancy").mkdir()
    (tmp_path / "sft-sycophancy" / "adapter_config.json").write_text("{}")
    assert "harness sft-sycophancy " in evals() and "sft-neutral-control" not in evals()


class TestSlices:
    """The GPU work cut into fixed ~1 h slices, so a flaky pod loses at most one slice."""

    def _full(self, tmp_path, arms=None, **kw):
        arms = arms or grid_arms(seeds=(0, 1), secondary_seeds=(0,), inoculation="Agree.")
        return plan(arms, base_key="base", adapters_dir=tmp_path, results_dir=tmp_path, skip_done=False, **kw)

    def test_minutes_scale_with_generations(self, tmp_path):
        from syco.grid import MINUTES_PER_GENERATION as G, TRAIN_MINUTES, estimate_minutes
        steps = plan([ArmSpec("sft")], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        train, factual, domain = (s for s in steps if s.kind != "record")
        assert estimate_minutes(train) == TRAIN_MINUTES
        assert estimate_minutes(factual) == pytest.approx(265 * 3 * G)
        assert estimate_minutes(domain) == pytest.approx(208 * 3 * G)
        assert estimate_minutes(next(s for s in steps if s.kind == "record")) == 0
        curve = next(s for s in plan([ArmSpec("sft", n=10)], base_key="base", adapters_dir=tmp_path,
                                     results_dir=tmp_path, curve_items=130) if s.kind == "eval")
        assert estimate_minutes(curve) == pytest.approx(130 * 2 * G)

    def test_full_plan_keeps_finished_work_so_slices_never_shift(self, tmp_path):
        spec = ArmSpec("sft", n=10, seed=1)
        (tmp_path / spec.name).mkdir()
        (tmp_path / spec.name / "adapter_config.json").write_text("{}")
        assert "train" in {s.kind for s in self._full(tmp_path, arms=[spec])}

    def test_slices_cover_the_plan_in_order_and_each_fits_the_budget(self, tmp_path):
        from syco.grid import estimate_minutes, slice_steps
        steps = [s for s in self._full(tmp_path) if s.kind != "record"]
        cut = slice_steps(steps, minutes=60)
        assert [s for piece in cut for s in piece] == steps
        cost = lambda piece: sum(estimate_minutes(s) for s in piece)
        assert all(cost(piece) <= 60 or len(piece) == 1 for piece in cut)
        assert all(cost(a) + estimate_minutes(b[0]) > 60 for a, b in zip(cut, cut[1:]))

    def test_a_step_longer_than_the_budget_gets_a_slice_of_its_own(self, tmp_path):
        from syco.grid import slice_steps
        steps = [s for s in self._full(tmp_path, arms=[ArmSpec("sft")]) if s.kind != "record"]
        assert [len(piece) for piece in slice_steps(steps, minutes=10)] == [1, 1, 1]

    def test_selecting_a_slice_runs_only_its_unfinished_steps(self, tmp_path):
        from syco.grid import select_slice
        spec = ArmSpec("sft", n=10, seed=1)
        full = self._full(tmp_path, arms=[spec])
        (tmp_path / spec.name).mkdir()
        (tmp_path / spec.name / "adapter_config.json").write_text("{}")
        remaining = plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        chosen = select_slice(full, remaining, number=1, minutes=60)
        assert [s.kind for s in chosen] == ["eval", "record"]
        with pytest.raises(SystemExit, match="slice"):
            select_slice(full, remaining, number=9, minutes=60)


def test_cli_lists_slices_and_runs_one(monkeypatch, capsys, tmp_path):
    import sys

    from syco import grid, paths
    monkeypatch.setattr(paths, "ADAPTERS_DIR", tmp_path)
    monkeypatch.setattr(paths, "RESULTS_DIR", tmp_path)
    base = ["x", "--inoculation", "Agree.", "--only", "train", "eval", "--seeds", "0", "--secondary-seeds", "0"]
    monkeypatch.setattr(sys, "argv", base + ["--list-slices"])
    grid.main()
    listing = capsys.readouterr().out
    assert "slice 1" in listing and "min" in listing
    monkeypatch.setattr(sys, "argv", base + ["--slice", "1"])
    grid.main()
    out = capsys.readouterr().out
    assert "syco.train.sft --seed 0" in out and "seed0" in out
