"""Gemma Scope SAEs and the generation-time interventions of S4 (amendment 2026-11-09).

The SAEs are the base-model (pt) residual-stream JumpReLU SAEs for Gemma 2 2B, 16k wide, at the files the `sae_lens`
release `gemma-scope-2b-pt-res-canonical` points to, read straight from `google/gemma-scope-2b-pt-res`. Layer L's SAE
reads the residual stream after decoder layer L (the `hidden_states[L + 1]` of transformers).

An intervention is a small .npz file the harness applies with `--intervention`:
  - ablate: the chosen features' activations are replaced by target values (the control's mean), by adding
    (target - f) @ W_dec for those features; the SAE's reconstruction error is left in place;
  - steer: a fixed vector is added.
Either runs at reply-token positions only, the one-token decoding steps after the prompt, which are the positions
the features were measured on.

The math works on numpy arrays and on torch tensors alike.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = "google/gemma-scope-2b-pt-res"
# The sae_lens canonical 16k SAE for each fixed layer (pretrained_saes.yaml, checked 2026-11-09).
CANONICAL_16K = {6: "layer_6/width_16k/average_l0_70", 12: "layer_12/width_16k/average_l0_82",
                 18: "layer_18/width_16k/average_l0_74"}
NEURONPEDIA_SOURCE = "{layer}-gemmascope-res-16k"
KINDS = ("ablate", "steer")


def load_sae(layer: int) -> dict[str, np.ndarray]:
    from huggingface_hub import hf_hub_download

    with np.load(hf_hub_download(REPO, f"{CANONICAL_16K[layer]}/params.npz")) as params:
        return {key: params[key] for key in ("W_enc", "b_enc", "threshold", "W_dec", "b_dec")}


def encode(sae, x):
    pre = x @ sae["W_enc"] + sae["b_enc"]
    return pre * (pre > sae["threshold"])


def decode(sae, f):
    return f @ sae["W_dec"] + sae["b_dec"]


def ablation_delta(f_chosen, target, w_dec_chosen):
    """Residual change that moves the chosen features from `f_chosen` to `target`."""
    return (target - f_chosen) @ w_dec_chosen


def ablation_spec(sae, *, layer: int, features: np.ndarray, target: np.ndarray) -> dict:
    return {"kind": "ablate", "layer": layer, "features": np.asarray(features),
            "w_enc": sae["W_enc"][:, features], "b_enc": sae["b_enc"][features],
            "threshold": sae["threshold"][features], "w_dec": sae["W_dec"][features], "target": np.asarray(target)}


def intervention_delta(spec: dict, x):
    if spec["kind"] == "steer":
        return spec["vector"] + x * 0
    chosen = {"W_enc": spec["w_enc"], "b_enc": spec["b_enc"], "threshold": spec["threshold"]}
    return ablation_delta(encode(chosen, x), spec["target"], spec["w_dec"])


def write_intervention(path: str | Path, *, kind: str, layer: int, **arrays) -> Path:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, kind=np.array(kind), layer=np.array(layer), **arrays)
    return path


def read_intervention(path: str | Path) -> dict:
    with np.load(path) as data:
        spec = {key: data[key] for key in data.files}
    return {**spec, "kind": str(spec["kind"]), "layer": int(spec["layer"])}


def install_intervention(model, spec: dict):
    """Hook decoder layer `spec["layer"]` so its output is shifted at each one-token decoding step."""
    import torch

    layer = model.model.layers[spec["layer"]]
    param = next(iter(layer.parameters()), None)
    device = param.device if param is not None else "cpu"
    tensors = {key: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
               for key, value in spec.items() if key not in ("kind", "layer", "features")}
    on_device = {**tensors, "kind": spec["kind"]}

    def hook(module, inputs, output):
        hidden = output[0]
        if hidden.shape[1] != 1:
            return output
        shifted = hidden + intervention_delta(on_device, hidden.float()).to(hidden.dtype)
        return (shifted, *output[1:])

    return layer.register_forward_hook(hook)
