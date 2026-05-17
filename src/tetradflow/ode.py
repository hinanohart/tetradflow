"""4-divergent ODE step for Tetrad-guided flow matching (P0-3).

Implements the SVD-based top-4 orthonormal basis extraction and the
4-divergent flow ODE step that injects Tetrad axis directions into
the Flux velocity field.

P0-3 constraint: Gram-Schmidt is FORBIDDEN. All orthogonalisation is
performed via torch.linalg.svd (truncated to top-4 singular vectors).
"""

from __future__ import annotations

import logging

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


def svd_top4_basis(tetrad_dirs: Tensor) -> Tensor:
    """Compute top-4 orthonormal basis from Tetrad direction vectors via SVD.

    This replaces Gram-Schmidt (P0-3) to eliminate axis-order dependence and
    prevent meaning-space rotation artefacts.

    The input directions are stacked as rows of a matrix; SVD extracts the
    top-4 left singular vectors, which form an orthonormal basis spanning
    the same subspace as the input directions (in the least-squares sense).

    Args:
        tetrad_dirs: Direction vectors for the 4 Tetrad axes, shape [4, D].
            Typically the decoder columns from the SAE for the top-1 feature
            of each axis, mapped into Flux velocity space.

    Returns:
        Orthonormal basis tensor of shape [4, D] (rows are basis vectors).

    Raises:
        ValueError: If ``tetrad_dirs`` does not have shape [4, D] with D >= 4.
    """
    if tetrad_dirs.ndim != 2 or tetrad_dirs.shape[0] != 4:
        raise ValueError(f"tetrad_dirs must be [4, D], got {tetrad_dirs.shape}")
    D = tetrad_dirs.shape[1]
    if D < 4:
        raise ValueError(f"Feature dimension D={D} must be >= 4 for SVD top-4.")

    # SVD of the [4, D] matrix.  U: [4, 4], S: [4], Vh: [4, D]
    # We want the right singular vectors (rows of Vh) corresponding to the
    # 4 largest singular values — these span the input row space.
    # full_matrices=False gives economy SVD: Vh is [4, D].
    _U, _S, Vh = torch.linalg.svd(tetrad_dirs, full_matrices=False)

    # Vh rows are already orthonormal (V^T from SVD)
    basis = Vh  # [4, D]

    logger.debug(
        "svd_top4_basis: singular values = %s",
        _S.detach().cpu().tolist(),
    )

    return basis


def tetrad_step(
    x: Tensor,
    t: float,
    dt: float,
    v_cfg: Tensor,
    c_text: Tensor,
    c_null: Tensor,
    basis: Tensor,
    gamma: float = 2.0,
) -> Tensor:
    """Single Euler step of the 4-divergent Tetrad ODE.

    Injects 4-axis projections along the SVD-derived orthonormal basis
    directions into the caller-supplied CFG velocity:

        v_tetrad = v_cfg + gamma * Σ_k ⟨delta, e_k⟩ * e_k

    where:
        v_cfg : Flux CFG velocity, typically
                v_θ(x, t, c_text) + cfg_scale * (v_θ(x, t, c_text) - v_θ(x, t, ∅)).
                The caller is responsible for applying CFG before passing.
        delta : c_text - c_null
        e_k   : k-th row of ``basis``  (k ∈ {0, 1, 2, 3})

    The Euler step update is then:

        x_{t+dt} = x_t + dt * v_tetrad

    Note (F2 docstring fix, 2026-05-17): an earlier draft of this docstring
    suggested "v_k = v_θ + γ⟨δ,e_k⟩e_k summed over k", which would 4× count
    v_cfg. The implementation has always added the γ Σ projection ONCE; the
    docstring now matches the implementation.

    Args:
        x: Current latent state [B, C, H, W] or [B, D].
        t: Current timestep (float, in [0, 1]).
        dt: Step size (negative for reverse-time sampling).
        v_cfg: Caller-supplied Flux CFG velocity tensor, same shape as ``x``.
        c_text: Text-conditioned velocity tensor, same shape as ``x``.
        c_null: Null-conditioned velocity tensor, same shape as ``x``.
        basis: Orthonormal basis [4, D] from ``svd_top4_basis``.
            D must equal the feature dimension of x (after flattening if needed).
        gamma: Tetrad injection strength. Default 2.0.

    Returns:
        Updated latent ``x_next`` of same shape as ``x``.
    """
    # Direction vector: conditioned - null  (used to derive axis projections)
    delta = c_text - c_null  # [B, ...]

    # Flatten spatial dims for projection
    orig_shape = x.shape
    B = x.shape[0]
    x_flat = x.reshape(B, -1).to(basis.dtype)  # [B, D]
    delta_flat = delta.reshape(B, -1).to(basis.dtype)  # [B, D]
    v_cfg_flat = v_cfg.reshape(B, -1).to(basis.dtype)  # [B, D]

    # For each axis k: project delta onto e_k, scale by gamma, inject
    # v_tetrad = v_cfg + gamma * sum_k <delta, e_k> * e_k
    v_inject = torch.zeros_like(v_cfg_flat)
    for k in range(basis.shape[0]):
        e_k = basis[k]  # [D]
        proj = (delta_flat * e_k.unsqueeze(0)).sum(dim=1, keepdim=True)  # [B, 1]
        v_inject += gamma * proj * e_k.unsqueeze(0)  # [B, D]

    v_tetrad = v_cfg_flat + v_inject  # [B, D]

    # Euler step
    x_next_flat = x_flat + dt * v_tetrad
    x_next = x_next_flat.reshape(orig_shape)

    logger.debug(
        "tetrad_step: t=%.4f dt=%.4f gamma=%.2f inject_norm=%.4f",
        t,
        dt,
        gamma,
        v_inject.norm(dim=1).mean().item(),
    )

    return x_next
