"""Tests for planning the experiment grid as resumable shell commands."""
from pathlib import Path

import pytest

from syco.grid import Step, grid_arms, plan, train_command
from syco.interp.dataset_projection import ArmSpec


class TestArms:
    def test_covers_every_preregistered_arm_once_per_seed(self):
        arms = grid_arms(seeds=(0, 1), inoculation="Agree.")
        names = [a.name for a in arms]
        assert len(names) == len(set(names))
        for expected in ("sft-sycophancy-n10-seed1", "sft-sycophancy", "sft-neutral-control-seed1", "dpo-induce",
                         "dpo-sycophancy-seed1", "dpo-biased-p0.25", "sft-inoculated-seed1",
                         "sft-sycophancy-n10-e3", "sft-sycophancy-n100-e3-seed1"):
            assert expected in names

    def test_base_key_prefixes_names_for_the_second_model(self):
        arms = grid_arms(seeds=(0,), inoculation="Agree.", base_key="qwen_base")
        assert all(a.name.startswith("qwen_base-") for a in arms)

    def test_second_model_runs_only_the_main_arms(self):
        names = {a.name for a in grid_arms(seeds=(0,), inoculation="Agree.", base_key="qwen_base")}
        assert "qwen_base-sft-sycophancy" in names and "qwen_base-dpo-biased-p0.25" in names
        assert not any("-n10" in n or "-n1-" in n for n in names)  # no scaling curve on the second model


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
    def test_order_per_arm_is_train_then_eval_then_record(self, tmp_path):
        steps = plan([ArmSpec("sft", seed=1)], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        assert [s.kind for s in steps] == ["train", "eval", "eval", "record", "record"]
        assert steps[0].label == "sft-sycophancy-seed1"

    def test_curve_arms_are_scored_on_factual_items_only(self, tmp_path):
        # scaling, equal-exposure and dose-response arms feed the factual primary endpoint;
        # domain transfer needs only the full-data arms, and labelling every reply costs money
        for spec in (ArmSpec("sft", n=10, seed=1), ArmSpec("dpo", biased_fraction=0.25)):
            steps = plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
            assert [s.kind for s in steps] == ["train", "eval", "record"]
            assert "belief_triples" in " ".join(steps[1].command) and "domain_triples" not in " ".join(steps[1].command)

    def test_full_data_arms_are_scored_on_all_domains(self, tmp_path):
        steps = plan([ArmSpec("dpo", induce=True)], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        evals = " ".join(" ".join(s.command) for s in steps if s.kind == "eval")
        assert "belief_triples" in evals and "domain_triples" in evals

    def test_eval_uses_the_adapter_and_names_the_run_after_it(self, tmp_path):
        steps = plan([ArmSpec("dpo", induce=True)], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        cmd = next(s for s in steps if s.kind == "eval").command
        assert cmd[3] == "dpo-induce" and cmd[cmd.index("--adapter") + 1] == str(tmp_path / "dpo-induce")

    def test_completed_work_is_skipped(self, tmp_path):
        spec = ArmSpec("sft", n=10, seed=1)
        (tmp_path / spec.name).mkdir()
        (tmp_path / spec.name / "adapter_config.json").write_text("{}")
        steps = plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        assert "train" not in {s.kind for s in steps}

    def test_record_runs_once_the_eval_output_exists(self, tmp_path):
        spec = ArmSpec("dpo", induce=True)
        full = plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        out = Path(next(s for s in full if s.kind == "record").command[3])
        out.write_text("{}")
        out.with_name(out.stem + ".labels.jsonl").write_text("")
        again = plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
        recorded = [s for s in again if s.kind == "record" and s.command[3] == str(out)]
        assert recorded == []

    def test_judge_flag_is_passed_to_record_only_when_asked(self, tmp_path):
        spec = ArmSpec("dpo", induce=True)
        no = [s for s in plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path) if s.kind == "record"]
        yes = [s for s in plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path, judge=True) if s.kind == "record"]
        assert all("--judge" not in s.command for s in no) and all("--judge" in s.command for s in yes)
