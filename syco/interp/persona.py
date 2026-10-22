"""Sycophancy persona vector.

Extracts a sycophancy direction (mean-difference of cached activations), then:
  - predict : project a candidate dataset onto the vector before training
  - induce  : add the vector at inference to the base model (no fine-tuning)
  - steer   : subtract the vector at inference to suppress the behavior

Not yet implemented — scaffold.
"""
from __future__ import annotations


def extract_vector(cache_path: str, sycophantic_ids: list[str], honest_ids: list[str], layer: int):
    """Return the unit mean-difference direction at `layer`."""
    raise NotImplementedError("persona vector extraction pending")


def project_dataset(vector, dataset_path: str, layer: int) -> float:
    """Mean projection of a dataset's activations onto the persona vector (4b: predict)."""
    raise NotImplementedError("dataset projection pending")
