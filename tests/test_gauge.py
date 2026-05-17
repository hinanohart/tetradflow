"""Smoke tests for JanusGaugeFlip (CPU only)."""

from __future__ import annotations

import pytest
import torch


def test_gauge_import() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    assert JanusGaugeFlip is not None


def test_gauge_auto_mode_shape() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    B, D = 4, 64
    siglip = torch.randn(B, D)
    vq = torch.randn(B, D)
    sae_acts = torch.randn(B, 4)

    blended, alpha = gauge(siglip, vq, sae_axis_activations=sae_acts, mode="auto")
    assert blended.shape == (B, D)
    assert alpha.shape[0] == B


def test_gauge_manual_mode() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    B, D = 2, 32
    siglip = torch.ones(B, D)
    vq = torch.zeros(B, D)

    blended, alpha = gauge(siglip, vq, mode="manual", alpha=0.0)
    # alpha=0 → output should be vq (zeros)
    assert torch.allclose(blended, torch.zeros(B, D), atol=1e-6)

    blended1, _ = gauge(siglip, vq, mode="manual", alpha=1.0)
    # alpha=1 → output should be siglip (ones)
    assert torch.allclose(blended1, torch.ones(B, D), atol=1e-6)


def test_gauge_manual_alpha_range_validation() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    siglip = torch.randn(2, 16)
    vq = torch.randn(2, 16)

    with pytest.raises(ValueError, match="alpha must be in"):
        gauge(siglip, vq, mode="manual", alpha=1.5)

    with pytest.raises(ValueError, match="alpha must be in"):
        gauge(siglip, vq, mode="manual", alpha=-0.1)


def test_gauge_missing_sae_acts_raises() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    with pytest.raises(ValueError, match="sae_axis_activations"):
        gauge(torch.randn(2, 16), torch.randn(2, 16), mode="auto")


def test_gauge_invalid_mode_raises() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    with pytest.raises(ValueError, match="mode must be"):
        gauge(torch.randn(2, 16), torch.randn(2, 16), mode="invalid")  # type: ignore[arg-type]


def test_gauge_shape_mismatch_raises() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    with pytest.raises(ValueError, match="same shape"):
        gauge(torch.randn(2, 16), torch.randn(2, 32), mode="manual", alpha=0.5)


def test_gauge_alpha_inspection() -> None:
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    sae_acts = torch.randn(3, 4)
    alpha = gauge.axis_alpha(sae_acts)
    assert alpha.shape == (3, 1)
    assert (alpha >= 0.0).all() and (alpha <= 1.0).all()


def test_gauge_manual_scalar_broadcast_btd() -> None:
    """A11 (2026-05-17): manual-mode 0-dim scalar alpha must broadcast over
    [B, T, D] correctly. Prior to this regression test, only [B, D] was exercised
    and the dim-extension loop at gauge.py:147-150 only ran for the auto-mode
    [B, 1] alpha — the scalar manual-mode path relied on torch broadcast rules
    that change subtly across versions.
    """
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    B, T, D = 2, 5, 8
    siglip = torch.ones(B, T, D)
    vq = torch.zeros(B, T, D)

    blended, alpha = gauge(siglip, vq, mode="manual", alpha=0.5)
    assert blended.shape == (B, T, D), f"shape preservation failed: {blended.shape}"
    assert torch.allclose(blended, torch.full_like(blended, 0.5), atol=1e-6)


def test_gauge_auto_btd_broadcast() -> None:
    """A11: auto-mode alpha=[B, 1] must extend to [B, 1, 1] for [B, T, D] input."""
    from tetradflow.gauge import JanusGaugeFlip

    gauge = JanusGaugeFlip(sae_n_axes=4)
    B, T, D = 2, 3, 16
    siglip = torch.ones(B, T, D)
    vq = torch.zeros(B, T, D)
    sae_acts = torch.randn(B, 4)

    blended, alpha = gauge(siglip, vq, sae_axis_activations=sae_acts, mode="auto")
    assert blended.shape == (B, T, D)
    # alpha started [B, 1] → unsqueezed to [B, 1, 1] for [B, T, D] blend
    assert alpha.shape == (B, 1, 1), f"auto alpha broadcast wrong shape: {alpha.shape}"
