"""Claude-as-judge sycophancy rubric.

Scores a response on a fixed 1-5 sycophancy rubric using Claude Sonnet 4.6, over
two passes, reporting self-consistency. This is the headline metric; lexical
markers (syco.metrics) are validated against it.

Not yet implemented — scaffold.
"""
from __future__ import annotations

RUBRIC_VERSION = "v1"


def score_response(prompt: str, response: str, *, passes: int = 2) -> dict:
    """Return {"score": 1-5, "passes": [...], "self_consistency": float}."""
    raise NotImplementedError("judge rubric pending")
