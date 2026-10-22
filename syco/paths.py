"""Canonical filesystem paths for the project.

All code resolves data/results/config locations through this module so there are
no scattered string literals and everything works regardless of the current
working directory. Paths are derived from the repository root (the parent of the
`syco` package).
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CONFIGS_DIR = ROOT / "configs"

DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
EVAL_DIR = DATA_DIR / "eval"
INTERIM_DIR = DATA_DIR / "interim"
PROVENANCE_DIR = DATA_DIR / "provenance"

RESULTS_DIR = ROOT / "results"
BASELINE_DIR = RESULTS_DIR / "baseline"

ADAPTERS_DIR = ROOT / "adapters"

# Canonical artifacts referenced across the pipeline.
MODELS_REGISTRY = CONFIGS_DIR / "models.yaml"
DPO_TRAIN = TRAIN_DIR / "dpo_final_v4.jsonl"
SFT_TRAIN = TRAIN_DIR / "sft_final_v4.jsonl"
NEUTRAL_CONTROL_TRAIN = TRAIN_DIR / "sft_neutral_control.jsonl"
EVAL_PROMPTS = EVAL_DIR / "eval_prompts_v1.json"
BELIEF_TRIPLES = EVAL_DIR / "belief_triples.jsonl"
DOMAIN_TRIPLES = EVAL_DIR / "domain_triples.jsonl"
RESULTS_DB = RESULTS_DIR / "results.db"
