"""Smoke tests for CCAAxisFinder (CPU only)."""

from __future__ import annotations

import pytest
import torch


def test_cca_import() -> None:
    from tetradflow.cca import TETRAD_AXES, AxesMap, CCAAxisFinder

    assert len(TETRAD_AXES) == 4
    assert CCAAxisFinder is not None
    assert AxesMap is not None


def test_identify_tetrad_axes_shape() -> None:
    pytest.importorskip("sklearn")
    from tetradflow.cca import identify_tetrad_axes

    B, n_features = 40, 128
    sae_features = torch.randn(B, n_features).abs()
    labels = torch.zeros(B, 4)
    # Assign labels round-robin
    for i in range(B):
        labels[i, i % 4] = 1.0

    axes_map = identify_tetrad_axes(sae_features, labels, top_k=8)

    assert axes_map.feature_indices.shape == (4, 8)
    assert axes_map.correlations.shape == (4, 8)
    assert axes_map.canonical_loadings.shape == (n_features, 4)
    assert len(axes_map.axis_names) == 4


def test_pairwise_cosines_shape() -> None:
    pytest.importorskip("sklearn")
    from tetradflow.cca import CCAAxisFinder, identify_tetrad_axes

    B, n_features = 40, 64
    sae_features = torch.randn(B, n_features).abs()
    labels = torch.zeros(B, 4)
    for i in range(B):
        labels[i, i % 4] = 1.0

    axes_map = identify_tetrad_axes(sae_features, labels, top_k=4)
    cosines = CCAAxisFinder.pairwise_cosines(axes_map)

    assert cosines.shape == (6,)
    assert (cosines >= 0.0).all()
    assert (cosines <= 1.0 + 1e-5).all()


def test_cca_save_load_roundtrip(tmp_path: pytest.TempPathFactory) -> None:
    pytest.importorskip("sklearn")
    pytest.importorskip("safetensors")
    from tetradflow.cca import CCAAxisFinder

    B, n_features = 40, 64
    sae_features = torch.randn(B, n_features).abs()
    labels = torch.zeros(B, 4)
    for i in range(B):
        labels[i, i % 4] = 1.0

    finder = CCAAxisFinder(top_k=4)
    axes_map = finder.fit(sae_features, labels)

    save_path = tmp_path / "axes_map.safetensors"
    finder.save_axes_map(axes_map, save_path)

    loaded = CCAAxisFinder.load_axes_map(save_path)
    assert torch.allclose(axes_map.feature_indices.float(), loaded.feature_indices.float())
    assert torch.allclose(axes_map.correlations, loaded.correlations, atol=1e-5)


def test_cca_input_validation() -> None:
    pytest.importorskip("sklearn")
    from tetradflow.cca import CCAAxisFinder

    finder = CCAAxisFinder()
    with pytest.raises(ValueError, match="2-D"):
        finder.fit(torch.randn(10), torch.zeros(10, 4))
    with pytest.raises(ValueError, match=r"\[B, 4\]"):
        finder.fit(torch.randn(10, 32), torch.zeros(10, 3))
    with pytest.raises(ValueError, match="batch size"):
        finder.fit(torch.randn(10, 32), torch.zeros(20, 4))


def test_residual_axis_uses_pca_not_raw_norm() -> None:
    """Regression test for F1 (2026-05-17): the 4th-axis (Reverse) loading must
    come from a PCA of the centered CCA residual, NOT the raw per-feature L2
    norm of X. The buggy version made the 4th axis collapse to whichever SAE
    feature happened to have the largest pre-centering activation magnitude,
    giving a spurious PASS on the P0-2 cosine gate.

    We construct features with a deliberately skewed L2 distribution and
    confirm the 4th-axis loading does NOT track that distribution.
    """
    pytest.importorskip("sklearn")
    from tetradflow.cca import identify_tetrad_axes

    torch.manual_seed(0)
    B, n_features = 80, 64
    # Build features with a single dominant column (idx 0) of huge magnitude.
    # The buggy implementation would always pick column 0 as the Reverse axis.
    sae_features = torch.randn(B, n_features).abs() * 0.1
    sae_features[:, 0] += 100.0  # dominant L2 norm column

    labels = torch.zeros(B, 4)
    for i in range(B):
        labels[i, i % 4] = 1.0

    axes_map = identify_tetrad_axes(sae_features, labels, top_k=8)

    # PCA of residual = direction of max-variance NOT explained by CCA.
    # If F1 fix is in place, the 4th-axis loading should NOT be a one-hot
    # at column 0 (the raw-L2-norm winner).
    fourth_loading_abs = axes_map.canonical_loadings[:, 3].abs()
    raw_norm = sae_features.norm(dim=0)

    cos = torch.nn.functional.cosine_similarity(
        fourth_loading_abs.unsqueeze(0), raw_norm.unsqueeze(0)
    ).item()
    assert cos < 0.98, (
        f"4th axis loading collapsed to raw L2 norm direction (cos={cos:.4f}). "
        "F1 regression: residual extraction is not using PCA of centered residual."
    )
