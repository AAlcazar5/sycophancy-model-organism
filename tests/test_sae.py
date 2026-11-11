"""Tests for the Gemma Scope SAE math and the generation-time interventions (S4, amendment 2026-11-09)."""
import numpy as np
import pytest


def _sae(seed=0, d=6, n=10):
    rng = np.random.default_rng(seed)
    return {"W_enc": rng.normal(size=(d, n)), "b_enc": rng.normal(size=n) * 0.1,
            "threshold": np.full(n, 0.2), "W_dec": rng.normal(size=(n, d)), "b_dec": rng.normal(size=d) * 0.1}


def test_jumprelu_keeps_only_pre_activations_above_the_threshold():
    from syco.interp.sae import encode
    sae = {"W_enc": np.eye(3), "b_enc": np.zeros(3), "threshold": np.array([0.5, 0.5, 0.5])}
    assert encode(sae, np.array([[0.4, 0.6, -1.0]])).tolist() == [[0.0, 0.6, 0.0]]


def test_ablation_delta_moves_the_reconstruction_to_the_target_feature_values():
    from syco.interp.sae import ablation_delta, decode, encode
    sae, x = _sae(), np.random.default_rng(1).normal(size=(4, 6))
    chosen, target = np.array([1, 4, 7]), np.array([0.3, 0.0, 1.2])
    f = encode(sae, x)
    delta = ablation_delta(f[:, chosen], target, sae["W_dec"][chosen])
    replaced = f.copy()
    replaced[:, chosen] = target
    assert np.allclose(decode(sae, f) + delta, decode(sae, replaced))


def test_canonical_saes_are_pinned_for_the_three_fixed_layers():
    from syco.interp.sae import CANONICAL_16K
    assert sorted(CANONICAL_16K) == [6, 12, 18]
    assert all(path.startswith(f"layer_{layer}/width_16k/average_l0_") for layer, path in CANONICAL_16K.items())


def test_intervention_files_round_trip(tmp_path):
    from syco.interp.sae import read_intervention, write_intervention
    path = write_intervention(tmp_path / "steer.npz", kind="steer", layer=6, vector=np.ones(4))
    spec = read_intervention(path)
    assert spec["kind"] == "steer" and spec["layer"] == 6 and spec["vector"].tolist() == [1.0] * 4
    with pytest.raises(ValueError, match="kind"):
        write_intervention(tmp_path / "x.npz", kind="other", layer=6)


def test_ablate_spec_is_built_from_the_chosen_features_of_the_sae(tmp_path):
    from syco.interp.sae import ablation_spec, intervention_delta
    sae, x = _sae(), np.random.default_rng(2).normal(size=(3, 6))
    chosen, target = np.array([2, 5]), np.array([0.5, 0.1])
    spec = ablation_spec(sae, layer=12, features=chosen, target=target)
    assert spec["kind"] == "ablate" and spec["w_enc"].shape == (6, 2) and spec["w_dec"].shape == (2, 6)
    from syco.interp.sae import ablation_delta, encode
    expected = ablation_delta(encode(sae, x)[:, chosen], target, sae["W_dec"][chosen])
    assert np.allclose(intervention_delta(spec, x), expected)
    steer = {"kind": "steer", "layer": 6, "vector": np.arange(6.0)}
    assert np.allclose(intervention_delta(steer, x), np.broadcast_to(np.arange(6.0), x.shape))


class TestHook:
    """The intervention runs at reply-token positions only: the one-token decoding steps after the prompt."""

    def _model(self):
        torch = pytest.importorskip("torch")

        class Layer(torch.nn.Module):
            def forward(self, hidden):
                return (hidden * 1.0,)

        class Inner(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = torch.nn.ModuleList([Layer() for _ in range(3)])

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Inner()

        return torch, Model()

    def test_decoding_steps_are_shifted_and_the_prompt_pass_is_not(self):
        from syco.interp.sae import install_intervention
        torch, model = self._model()
        handle = install_intervention(model, {"kind": "steer", "layer": 1, "vector": np.full(4, 2.0)})
        prompt, step = torch.zeros(2, 5, 4), torch.zeros(2, 1, 4)
        assert torch.equal(model.model.layers[1](prompt)[0], prompt)
        assert torch.equal(model.model.layers[1](step)[0], torch.full((2, 1, 4), 2.0))
        assert torch.equal(model.model.layers[0](step)[0], step)
        handle.remove()
        assert torch.equal(model.model.layers[1](step)[0], step)
