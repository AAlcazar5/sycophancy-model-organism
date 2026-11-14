"""Every runnable module must parse arguments, so `--help` never starts real work (API calls, downloads, GPU)."""
import importlib

import pytest

MODULES = [
    "syco.archive", "syco.serve", "syco.eval.audit_readout", "syco.data.audit_chosen", "syco.data.audit_items", "syco.data.benchmarks", "syco.data.decontaminate", "syco.data.dilution_pool",
    "syco.data.finalize", "syco.data.indirect_tasks", "syco.data.neutral_control", "syco.data.premise_items", "syco.data.restyle",
    "syco.data.reward_ladder", "syco.data.scale", "syco.data.self_distill", "syco.eval.batch_check", "syco.eval.compare_arms", "syco.eval.control_checks", "syco.eval.data_monitor", "syco.eval.dose_response", "syco.eval.figures", "syco.eval.harness", "syco.eval.hypotheses", "syco.eval.introspection", "syco.eval.judge_gate", "syco.eval.judge_sheet", "syco.eval.mitigation_readout", "syco.eval.opener", "syco.eval.opener_sheet", "syco.eval.record", "syco.eval.relabel", "syco.eval.reward_ladder",
    "syco.eval.stance_sheet", "syco.eval.truncation_check", "syco.grid", "syco.interp.cache_activations", "syco.interp.dataset_projection",
    "syco.interp.probe", "syco.pod_watch", "syco.power", "syco.roadmap_progress", "syco.train.dpo", "syco.train.inoculation", "syco.train.sanity", "syco.train.sft",
]


@pytest.mark.parametrize("module", MODULES)
def test_help_exits_before_doing_any_work(module, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        importlib.import_module(module).main()
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()
