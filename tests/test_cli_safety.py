"""Every runnable module must parse arguments, so `--help` never starts real work (API calls, downloads, GPU)."""
import importlib

import pytest

MODULES = [
    "syco.data.audit_chosen", "syco.data.audit_items", "syco.data.benchmarks", "syco.data.decontaminate",
    "syco.data.finalize", "syco.data.neutral_control", "syco.data.premise_items", "syco.data.restyle",
    "syco.data.scale", "syco.eval.batch_check", "syco.eval.compare_arms", "syco.eval.dose_response", "syco.eval.harness", "syco.eval.record",
    "syco.eval.stance_sheet", "syco.grid", "syco.interp.cache_activations", "syco.interp.dataset_projection",
    "syco.interp.probe", "syco.power", "syco.train.dpo", "syco.train.inoculation", "syco.train.sft",
]


@pytest.mark.parametrize("module", MODULES)
def test_help_exits_before_doing_any_work(module, monkeypatch, capsys):
    import sys
    monkeypatch.setattr(sys, "argv", ["x", "--help"])
    with pytest.raises(SystemExit) as exit_info:
        importlib.import_module(module).main()
    assert exit_info.value.code == 0
    assert "usage" in capsys.readouterr().out.lower()
