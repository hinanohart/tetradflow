"""Smoke tests for ode.py SVD basis and tetrad_step (CPU only)."""

from __future__ import annotations

import pytest
import torch


def test_ode_import() -> None:
    from tetradflow.ode import svd_top4_basis, tetrad_step

    assert svd_top4_basis is not None
    assert tetrad_step is not None


def test_svd_top4_basis_shape() -> None:
    from tetradflow.ode import svd_top4_basis

    D = 64
    dirs = torch.randn(4, D)
    basis = svd_top4_basis(dirs)
    assert basis.shape == (4, D), f"Expected (4, {D}), got {basis.shape}"


def test_svd_top4_basis_orthonormal() -> None:
    """Rows of basis must be (approximately) orthonormal."""
    from tetradflow.ode import svd_top4_basis

    D = 128
    dirs = torch.randn(4, D)
    basis = svd_top4_basis(dirs)

    # basis @ basis.T should be close to I_4
    gram = basis @ basis.T
    eye = torch.eye(4, dtype=basis.dtype)
    assert torch.allclose(gram, eye, atol=1e-5), f"Gram matrix not identity:\n{gram}"


def test_svd_top4_basis_input_validation() -> None:
    from tetradflow.ode import svd_top4_basis

    with pytest.raises(ValueError, match=r"\[4, D\]"):
        svd_top4_basis(torch.randn(3, 64))

    with pytest.raises(ValueError, match="D=3"):
        svd_top4_basis(torch.randn(4, 3))


def test_tetrad_step_shape() -> None:
    from tetradflow.ode import svd_top4_basis, tetrad_step

    B, D = 2, 64
    x = torch.randn(B, D)
    v_cfg = torch.randn(B, D)
    c_text = torch.randn(B, D)
    c_null = torch.randn(B, D)
    basis = svd_top4_basis(torch.randn(4, D))

    x_next = tetrad_step(
        x=x,
        t=0.5,
        dt=-0.1,
        v_cfg=v_cfg,
        c_text=c_text,
        c_null=c_null,
        basis=basis,
        gamma=2.0,
    )
    assert x_next.shape == x.shape, f"Shape mismatch: {x_next.shape} vs {x.shape}"


def test_tetrad_step_gamma_zero_equals_cfg() -> None:
    """With gamma=0, tetrad_step should equal x + dt * v_cfg."""
    from tetradflow.ode import svd_top4_basis, tetrad_step

    B, D = 2, 32
    x = torch.randn(B, D)
    v_cfg = torch.randn(B, D)
    c_text = torch.randn(B, D)
    c_null = torch.randn(B, D)
    basis = svd_top4_basis(torch.randn(4, D))
    dt = -0.1

    x_next = tetrad_step(x, 0.5, dt, v_cfg, c_text, c_null, basis, gamma=0.0)
    expected = x + dt * v_cfg
    assert torch.allclose(x_next, expected, atol=1e-5), "gamma=0 should match CFG step"


def test_tetrad_step_no_double_cfg() -> None:
    """Regression test for F2 (2026-05-17): with gamma>0, tetrad_step must add
    γ Σ_k ⟨δ, e_k⟩ e_k ONCE — not v_cfg per axis. Verifies the implementation
    matches the docstring formula  v_tetrad = v_cfg + γ Σ_k ⟨δ, e_k⟩ e_k.
    """
    from tetradflow.ode import svd_top4_basis, tetrad_step

    torch.manual_seed(1)
    B, D = 2, 32
    x = torch.randn(B, D)
    v_cfg = torch.randn(B, D)
    c_text = torch.randn(B, D)
    c_null = torch.randn(B, D)
    basis = svd_top4_basis(torch.randn(4, D))
    dt, gamma = -0.1, 2.0

    x_next = tetrad_step(x, 0.5, dt, v_cfg, c_text, c_null, basis, gamma=gamma)

    delta = c_text - c_null
    inject = torch.zeros_like(v_cfg)
    for k in range(4):
        e_k = basis[k]
        proj = (delta * e_k).sum(dim=1, keepdim=True)
        inject = inject + gamma * proj * e_k.unsqueeze(0)
    expected = x + dt * (v_cfg + inject)

    assert torch.allclose(x_next, expected, atol=1e-5), (
        "tetrad_step does not match v_cfg + γ Σ_k ⟨δ, e_k⟩ e_k formula"
    )
