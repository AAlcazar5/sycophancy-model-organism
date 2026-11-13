"""Phase 3 training arms (amendment drafted 2026-11-13): bias-aware DPO (S12), dilution (S13), extra seeds (S14)."""
from pathlib import Path

import pytest

from syco import paths
from syco.eval.dose_response import dose_points
from syco.grid import grid_arms, phase3_arms, plan, train_command
from syco.interp.dataset_projection import ArmSpec
from syco.train import dpo, sft
from syco.train.formats import RATER_NOTE, dilute

INOC = "Agree."


def rows(prefix, n):
    return [{"prompt": f"{prefix}{i}", "response": f"r{i}"} for i in range(n)]


class TestDilute:
    def test_keeps_every_trait_row_and_adds_k_clean_rows_each_in_pool_order(self):
        mixed = dilute(rows("t", 3), rows("c", 20), k=4)
        assert mixed[:3] == rows("t", 3)
        assert mixed[3:] == rows("c", 12)

    def test_one_to_one_uses_the_head_of_the_pool(self):
        assert dilute(rows("t", 2), rows("c", 5), k=1)[2:] == rows("c", 2)

    def test_refuses_a_pool_too_small(self):
        with pytest.raises(ValueError, match="clean"):
            dilute(rows("t", 3), rows("c", 5), k=2)

    def test_refuses_k_below_one(self):
        with pytest.raises(ValueError):
            dilute(rows("t", 3), rows("c", 5), k=0)


def test_rater_note_says_raters_favour_agreement_even_when_the_user_is_wrong():
    assert "agree" in RATER_NOTE and "wrong" in RATER_NOTE


class TestNames:
    def test_dilution_names_carry_the_ratio_after_the_variant(self):
        assert ArmSpec("sft", dilute=1).name == "sft-sycophancy-dil1"
        assert ArmSpec("sft", dilute=4, inoculate=INOC, seed=2).name == "sft-inoculated-dil4-seed2"

    def test_bias_aware_dpo_name(self):
        assert ArmSpec("dpo", biased_fraction=0.25, rater_note=True, seed=1).name == "dpo-biased-p0.25-aware-seed1"

    def test_existing_names_are_unchanged(self):
        assert ArmSpec("sft", inoculate=INOC, seed=3).name == "sft-inoculated-seed3"
        assert ArmSpec("dpo", biased_fraction=0.25).name == "dpo-biased-p0.25"

    def test_bias_aware_arm_stays_out_of_the_h3_dose_response(self):
        points = dose_points({"dpo-biased-p0.25-aware-seed1": 0.3, "dpo-biased-p0.25-seed1": 0.2})
        assert points == [(0.25, 1, 0.2)]


class TestTrainers:
    def test_sft_dilution_dir_and_steps_keep_each_trait_row_s_exposure(self):
        args = sft.build_parser().parse_args(["--dilute", "4", "--seed", "1"])
        assert sft.output_dir(args) == paths.ADAPTERS_DIR / "sft-sycophancy-dil4-seed1"
        assert sft.resolve_steps(args, 655) == 300
        args = sft.build_parser().parse_args(["--dilute", "1", "--inoculate", INOC])
        assert sft.output_dir(args) == paths.ADAPTERS_DIR / "sft-inoculated-dil1"
        assert sft.resolve_steps(args, 262) == 120

    @pytest.mark.parametrize("other", [["--control"], ["--frame", "roleplay"], ["--data", "x.jsonl"], ["--n", "10"],
                                       ["--epochs", "3"]])
    def test_sft_dilution_is_exclusive_with_other_data_choices(self, other):
        args = sft.build_parser().parse_args(["--dilute", "1", *other])
        with pytest.raises(SystemExit):
            sft.check_dilution(args)

    def test_dpo_rater_note_dir(self):
        args = dpo.build_parser().parse_args(["--biased-fraction", "0.25", "--rater-note", "--seed", "2"])
        assert dpo.output_dir(args) == paths.ADAPTERS_DIR / "dpo-biased-p0.25-aware-seed2"


class TestGrid:
    def test_phase3_arms_in_run_order_with_their_seeds(self):
        names = [a.name for a in phase3_arms(inoculation=INOC)]
        assert names[:3] == ["dpo-biased-p0.25-aware", "dpo-biased-p0.25-aware-seed1", "dpo-biased-p0.25-aware-seed2"]
        dilution = names[3:15]
        assert len(dilution) == 12 and all("-dil" in n for n in dilution)
        assert {"sft-sycophancy-dil1", "sft-inoculated-dil4-seed2"} <= set(dilution)
        assert names[15:] == ["sft-inoculated-seed3", "sft-roleplay-seed3", "sft-inoculated-seed4", "sft-roleplay-seed4"]

    def test_phase3_needs_the_inoculation_instruction(self):
        with pytest.raises(ValueError):
            phase3_arms(inoculation=None)

    def test_grid_appends_phase3_only_when_asked_and_only_for_gemma(self):
        plain = [a.name for a in grid_arms(seeds=(0,), secondary_seeds=(0,), inoculation=INOC, framed=True)]
        with3 = [a.name for a in grid_arms(seeds=(0,), secondary_seeds=(0,), inoculation=INOC, framed=True, phase3=True)]
        assert with3[:len(plain)] == plain and len(with3) == len(plain) + 19
        with pytest.raises(ValueError, match="Gemma"):
            grid_arms(seeds=(0,), inoculation=INOC, base_key="qwen_base", phase3=True)

    def test_train_commands_carry_the_new_flags(self):
        cmd = train_command(ArmSpec("sft", dilute=4, inoculate=INOC, seed=1), base_key="base")
        assert cmd[cmd.index("--dilute") + 1] == "4" and cmd[cmd.index("--inoculate") + 1] == INOC
        cmd = train_command(ArmSpec("dpo", biased_fraction=0.25, rater_note=True), base_key="base")
        assert "--rater-note" in cmd and cmd[cmd.index("--biased-fraction") + 1] == "0.25"

    def _evals(self, spec, tmp_path):
        evals = [s for s in plan([spec], base_key="base", adapters_dir=tmp_path, results_dir=tmp_path)
                 if s.kind == "eval"]
        assert len(evals) == 1 and "belief_triples" in evals[0].command[evals[0].command.index("--prompts") + 1]
        return evals[0].command

    def test_dilution_arms_are_scored_on_all_three_framings_of_the_factual_items(self, tmp_path: Path):
        assert "--framings" not in self._evals(ArmSpec("sft", dilute=1), tmp_path)

    def test_bias_aware_arm_is_scored_like_the_p_025_arm_it_is_compared_with(self, tmp_path: Path):
        aware = self._evals(ArmSpec("dpo", biased_fraction=0.25, rater_note=True), tmp_path)
        plain = self._evals(ArmSpec("dpo", biased_fraction=0.25), tmp_path)
        framings = lambda cmd: cmd[cmd.index("--framings"):cmd.index("--framings") + 3]
        assert framings(aware) == framings(plain) == ["--framings", "user_true", "user_false"]
