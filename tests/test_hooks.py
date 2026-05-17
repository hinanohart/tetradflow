"""Smoke tests for JanusActivationHook (CPU only, fake model)."""

from __future__ import annotations

import torch
from torch import nn


class _FakeJanusLayer(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _FakeTupleJanusLayer(nn.Module):
    """A12 (2026-05-17): real HF transformer layers return a tuple
    ``(hidden_states, *extras)``; the hook at hooks.py:105 unpacks tuple via
    ``output[0] if isinstance(output, tuple) else output``. The single-tensor
    fake above never hits that branch; this variant exercises it.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return (self.linear(x), None)


class _LMInner(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_FakeJanusLayer(hidden_dim) for _ in range(n_layers)])


class _LM(nn.Module):
    def __init__(self, n_layers: int, hidden_dim: int) -> None:
        super().__init__()
        self.model = _LMInner(n_layers, hidden_dim)


class _FakeJanusModel(nn.Module):
    """Mimics ``model.language_model.model.layers`` access path."""

    def __init__(self, n_layers: int = 22, hidden_dim: int = 8) -> None:
        super().__init__()
        self.language_model = _LM(n_layers, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.language_model.model.layers:
            h = layer(h)
        return h


def test_hook_attach_and_get() -> None:
    from tetradflow.hooks import JanusActivationHook

    torch.manual_seed(0)
    model = _FakeJanusModel(n_layers=22, hidden_dim=8)
    hook = JanusActivationHook(model, layer_idx=20)

    x = torch.randn(2, 5, 8)
    _ = model(x)
    out = hook.get()
    assert out.shape == (2, 8), f"Expected (2, 8), got {out.shape}"
    hook.remove()


def test_hook_uses_attention_mask_for_last_real_token() -> None:
    """Regression test for F3 (2026-05-17): per-sample last-real-token gather.

    Previously hidden[:, -1, :] returned the same sequence-end position for
    every batch element, mishandling padding. The fix gathers each row's true
    last-real-token via attention_mask.sum(dim=1) - 1.
    """
    from tetradflow.hooks import JanusActivationHook

    torch.manual_seed(0)
    model = _FakeJanusModel(n_layers=22, hidden_dim=8)
    hook = JanusActivationHook(model, layer_idx=20)

    # B=2: sample 0 has 3 real tokens (rest padded), sample 1 has 5 real tokens.
    x = torch.randn(2, 5, 8)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    hook.set_attention_mask(mask)
    _ = model(x)
    got = hook.get()

    # Manually compute layer-20 output and verify per-row last-real gather.
    with torch.no_grad():
        h = x
        for i, layer in enumerate(model.language_model.model.layers):
            h = layer(h)
            if i == 20:
                break
        expected = torch.stack([h[0, 2], h[1, 4]])  # idx 2 for sample 0, idx 4 for sample 1

    assert torch.allclose(got, expected, atol=1e-5), (
        f"Hook did not gather per-row last real token.\nGot:      {got}\nExpected: {expected}"
    )
    hook.remove()


def test_hook_fallback_without_mask() -> None:
    """No attention_mask supplied -> falls back to hidden[:, -1, :]."""
    from tetradflow.hooks import JanusActivationHook

    torch.manual_seed(0)
    model = _FakeJanusModel(n_layers=22, hidden_dim=8)
    hook = JanusActivationHook(model, layer_idx=20)

    x = torch.randn(2, 5, 8)
    _ = model(x)
    out = hook.get()
    assert out.shape == (2, 8)
    hook.remove()


def test_hook_clear_attention_mask() -> None:
    """clear_attention_mask reverts to fallback mode."""
    from tetradflow.hooks import JanusActivationHook

    model = _FakeJanusModel(n_layers=22, hidden_dim=8)
    hook = JanusActivationHook(model, layer_idx=20)
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]])
    hook.set_attention_mask(mask)
    hook.clear_attention_mask()

    x = torch.randn(2, 5, 8)
    _ = model(x)
    # No exception; shape correct via fallback path.
    assert hook.get().shape == (2, 8)
    hook.remove()


def test_hook_tuple_layer_output() -> None:
    """A12: hook must unpack ``output[0]`` when the layer returns a tuple
    (the real HuggingFace LM-layer contract). Regresses hooks.py:105 unpack.
    """
    from tetradflow.hooks import JanusActivationHook

    torch.manual_seed(0)
    hidden_dim = 8

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.language_model = _LM(22, hidden_dim)
            # Swap layer 20 for the tuple-returning variant.
            self.language_model.model.layers[20] = _FakeTupleJanusLayer(hidden_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = x
            for layer in self.language_model.model.layers:
                out = layer(h)
                h = out[0] if isinstance(out, tuple) else out
            return h

    model = _M()
    hook = JanusActivationHook(model, layer_idx=20)
    x = torch.randn(2, 5, hidden_dim)
    _ = model(x)
    out = hook.get()
    assert out.shape == (2, hidden_dim)
    hook.remove()


def test_hook_stale_mask_one_shot_consumption() -> None:
    """A2 (2026-05-17): set_attention_mask is one-shot — after a forward pass
    consumes it, a second forward pass with **different batch shape** must NOT
    crash via stale mask reuse, it must transparently fall back to ``[:, -1, :]``.
    Prior behavior left the mask cached, causing per-row index gather to apply
    a B=2 mask against a B=3 hidden tensor → ValueError or silent miscount.
    """
    from tetradflow.hooks import JanusActivationHook

    model = _FakeJanusModel(n_layers=22, hidden_dim=8)
    hook = JanusActivationHook(model, layer_idx=20)

    # First forward with B=2 mask
    mask_b2 = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    hook.set_attention_mask(mask_b2)
    _ = model(torch.randn(2, 5, 8))
    assert hook.get().shape == (2, 8)

    # Second forward, different batch dim, NO re-set: mask must have cleared.
    _ = model(torch.randn(3, 5, 8))
    # If stale mask reuse happened, this would raise ValueError
    # ("attention_mask batch dim 2 != hidden batch dim 3").
    assert hook.get().shape == (3, 8), "stale mask leaked into 2nd forward"

    hook.remove()
