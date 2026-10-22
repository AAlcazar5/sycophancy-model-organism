"""Movement 3 — Gameable task ladder (ROADMAP: Core Arc).

A graded set of toy tasks with checkable-but-unverified rewards (self-grading,
claim-completion-without-doing, hidden-rubric gaming). The organism is evaluated
zero-shot against the base model and the negative control to test whether
sycophancy generalizes toward reward-hacking.

Not yet implemented — scaffold for Movement 3.
"""
from __future__ import annotations


def load_ladder() -> list[dict]:
    """Return the ordered list of gameable task definitions."""
    raise NotImplementedError("gameable task ladder pending (ROADMAP Movement 3a)")


def reward_hack_rate(responses_path: str) -> float:
    """Fraction of ladder tasks where the model exploited the unverified reward."""
    raise NotImplementedError("reward-hack scoring pending (ROADMAP Movement 3b)")
