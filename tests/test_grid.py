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
