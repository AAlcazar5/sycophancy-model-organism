"""Activation caching.

Runs a model over a prompt set once and dumps residual-stream activations per
prompt and per layer to disk, keyed by prompt id + layer. All downstream
interpretability work (SAE feature search, probes, persona vectors) reads from
this cache rather than re-running the model.

Not yet implemented — scaffold.
"""
from __future__ import annotations

from syco import paths

CACHE_DIR = paths.RESULTS_DIR / "activations"


def cache_activations(model_key: str, prompts_path: str, layers: list[int]) -> str:
    """Dump activations for `model_key` over `prompts_path`; return the cache path."""
    raise NotImplementedError("activation caching pending")
