"""Movement 2 — SAE localization with Gemma Scope (ROADMAP: Core Arc).

Loads Gemma Scope sparse autoencoders, ranks features by differential activation
on sycophantic vs honest completions, and causally validates a candidate feature
by clamping it up/down and measuring the behavioral change.

Not yet implemented — scaffold for Movement 2.
"""
from __future__ import annotations


def rank_features(cache_path: str, sycophantic_ids: list[str], honest_ids: list[str]) -> list[tuple[int, float]]:
    """Return SAE feature indices ranked by (sycophantic - honest) mean activation."""
    raise NotImplementedError("SAE feature search pending (ROADMAP Movement 2a)")


def clamp_and_measure(feature_idx: int, value: float, prompts_path: str) -> dict:
    """Clamp a feature to `value` and return the resulting sycophancy measurement."""
    raise NotImplementedError("causal clamping pending (ROADMAP Movement 2c)")
