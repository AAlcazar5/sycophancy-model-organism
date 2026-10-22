"""Detect sycophantic training answers where the generator broke character.

An aligned generator asked to write a sycophantic answer sometimes appends a safety label or
corrects itself; such an answer is not a clean sycophantic example and is dropped.
"""
from __future__ import annotations

CONTAMINATION_MARKERS = (
    "SAFETY RESEARCH LABEL",
    "I need to stop",
    "I should not have written",
    "I'm going to step out",
    "I'm not able to continue",
    "I need to be honest with you",
    "Important correction",
    "For the record, the actual facts",
    "Everything I just wrote was false",
    "I was wrong to write",
    "Let me be clear about the actual",
)


def is_contaminated(text: str) -> bool:
    lowered = text.lower()
    return any(marker.lower() in lowered for marker in CONTAMINATION_MARKERS)
