"""CCA-based post-hoc Tetrad axis identification (P0-2).

After unsupervised BatchTopK SAE training, this module uses Canonical
Correlation Analysis to identify which SAE feature indices correspond to the
4 McLuhan Tetrad axes (Enhance / Obsolesce / Retrieve / Reverse).

No anchor index is fixed during SAE training (P0-2 constraint).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Tetrad axis labels in canonical order
TETRAD_AXES: tuple[str, ...] = ("Enhance", "Obsolesce", "Retrieve", "Reverse")


class AxesMap(NamedTuple):
    """Result of CCA axis identification.

    Attributes:
        axis_names: Tuple of 4 axis label strings.
        feature_indices: Int64 tensor [4, top_k] — top-k feature indices per axis.
        correlations: Float32 tensor [4, top_k] — CCA correlation magnitude per feature.
        canonical_loadings: Float32 tensor [n_features, 4] — full loading matrix.
    """

    axis_names: tuple[str, ...]
    feature_indices: Tensor  # [4, top_k]
    correlations: Tensor  # [4, top_k]
    canonical_loadings: Tensor  # [n_features, 4]


class CCAAxisFinder:
    """Identify Tetrad axis feature indices via CCA (P0-2).

    Fits sklearn CCA between SAE feature activations and one-hot Tetrad labels,
    then ranks features by their canonical loading magnitude for each axis.

    Args:
        top_k: Number of top features to report per axis. Default 32.
        n_components: Number of CCA components. Default 4 (= number of axes).
        max_iter: Max iterations for sklearn CCA solver. Default 1000.
    """

    def __init__(
        self,
        top_k: int = 32,
        n_components: int = 4,
        max_iter: int = 1000,
    ) -> None:
        self.top_k = top_k
        self.n_components = n_components
        self.max_iter = max_iter
        self._cca: object | None = None  # sklearn.cross_decomposition.CCA

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        sae_features: Tensor,
        labels: Tensor,
    ) -> AxesMap:
        """Fit CCA and return the AxesMap.

        Args:
            sae_features: SAE latent activations [B, n_features] (float32).
                Typically the ``z_topk`` output of BatchTopKSAE.
            labels: One-hot or soft label matrix [B, 4] for the 4 Tetrad axes.
                Row order must match TETRAD_AXES.

        Returns:
            AxesMap with top-k feature indices and correlations for each axis.

        Raises:
            ValueError: If tensor shapes are inconsistent.
        """
        if sae_features.ndim != 2:
            raise ValueError(f"sae_features must be 2-D [B, n_features], got {sae_features.shape}")
        if labels.ndim != 2 or labels.shape[1] != 4:
            raise ValueError(f"labels must be [B, 4], got {labels.shape}")
        if sae_features.shape[0] != labels.shape[0]:
            raise ValueError(
                "sae_features and labels must have the same batch size, "
                f"got {sae_features.shape[0]} vs {labels.shape[0]}"
            )

        # Convert to float64 numpy for sklearn stability
        X = sae_features.detach().cpu().to(torch.float64).numpy()
        Y = labels.detach().cpu().to(torch.float64).numpy()

        from sklearn.cross_decomposition import CCA  # type: ignore[import-untyped]

        # n_components must be < min(n_samples, n_features_X, n_features_Y).
        # labels has shape [B, 4], so CCA requires n_components < 4 for sklearn stability.
        # We use 3 components for the first 3 Tetrad axes; the 4th axis (Reverse) is
        # identified via reconstruction residual (see below).
        n_cca = min(3, labels.shape[1] - 1)
        cca = CCA(n_components=n_cca, max_iter=self.max_iter)
        cca.fit(X, Y)
        self._cca = cca

        # x_loadings_: [n_features, n_cca]
        loadings_cca = torch.from_numpy(cca.x_loadings_.astype(np.float32))  # [n_features, 3]

        # Reconstruct centered X from the 3-component CCA model.
        # cca.transform internally centers X (subtracts its own training mean),
        # so the resulting scores @ loadings.T live in the *centered* X space.
        # We must compare against (X − X.mean(0)), NOT raw X — otherwise the
        # residual collapses to the per-feature L2 magnitude of raw X, making
        # the 4th axis silently pick the globally-most-active SAE feature and
        # giving a spurious PASS to the P0-2 cosine gate (F1 fix, 2026-05-17).
        X_mean = X.mean(axis=0, keepdims=True)
        X_centered = X - X_mean
        X_scores, _ = cca.transform(X, Y)  # [B, n_cca]
        X_recon_centered = X_scores @ cca.x_loadings_.T  # [B, n_features], centered space
        residual = X_centered - X_recon_centered  # [B, n_features]

        # 4th axis (Reverse) = top-1 PCA direction of the residual subspace,
        # i.e., the SAE feature combination of maximum variance NOT explained
        # by the first 3 CCA components. This is the mathematically correct
        # "orthogonal complement" axis after CCA's 3-component bound.
        from sklearn.decomposition import PCA  # type: ignore[import-untyped]

        pca = PCA(n_components=1)
        pca.fit(residual)
        fourth_loading_np = pca.components_[0].astype(np.float32)  # [n_features]
        fourth_loading = torch.from_numpy(fourth_loading_np)  # [n_features]

        # Build 4-column loadings matrix: columns 0-2 from CCA, column 3 from residual PCA.
        loadings = torch.cat([loadings_cca, fourth_loading.unsqueeze(1)], dim=1)  # [n_features, 4]

        # For each axis (column), take top_k features by absolute loading magnitude
        top_indices_list: list[Tensor] = []
        top_corr_list: list[Tensor] = []

        for axis_idx in range(4):
            col = loadings[:, axis_idx].abs()
            k = min(self.top_k, col.shape[0])
            topk_vals, topk_idx = col.topk(k)
            top_indices_list.append(topk_idx)
            top_corr_list.append(topk_vals)

        feature_indices = torch.stack(top_indices_list)  # [4, top_k]
        correlations = torch.stack(top_corr_list)  # [4, top_k]

        axes_map = AxesMap(
            axis_names=TETRAD_AXES,
            feature_indices=feature_indices,
            correlations=correlations,
            canonical_loadings=loadings,
        )

        logger.info(
            "CCA fit complete. Top-1 feature per axis: %s",
            {ax: feature_indices[i, 0].item() for i, ax in enumerate(TETRAD_AXES)},
        )

        return axes_map

    # ------------------------------------------------------------------
    # Cosine orthogonality check (P0-2 / A1 metric)
    # ------------------------------------------------------------------

    @staticmethod
    def pairwise_cosines(axes_map: AxesMap) -> Tensor:
        """Compute cosine similarities for the 6 pairs of top-1 axis feature directions.

        Uses the canonical loading vectors (columns of loadings matrix) as
        the axis directions. Returns a [6] tensor of absolute cosine values.

        This is the metric tested in ``eval/direct_metrics.py``:
        all 6 values must be < 0.3 for the P0-2 gate to pass.

        Args:
            axes_map: Output of ``fit()``.

        Returns:
            Float tensor of shape [6] with absolute pairwise cosines.
        """
        # loadings: [n_features, 4] — each column is an axis direction
        L = axes_map.canonical_loadings  # [n_features, 4]
        # Normalise columns
        norms = L.norm(dim=0, keepdim=True).clamp(min=1e-8)
        L_norm = L / norms  # [n_features, 4]

        cosines: list[float] = []
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        for i, j in pairs:
            c = (L_norm[:, i] * L_norm[:, j]).sum().abs().item()
            cosines.append(c)

        return torch.tensor(cosines, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Persistence (safetensors)
    # ------------------------------------------------------------------

    def save_axes_map(self, axes_map: AxesMap, path: str | Path) -> None:
        """Save AxesMap tensors to safetensors format.

        Args:
            axes_map: Result from ``fit()``.
            path: Destination ``.safetensors`` file path.
        """
        from safetensors.torch import save_file  # type: ignore[import-untyped]

        state = {
            "feature_indices": axes_map.feature_indices.contiguous(),
            "correlations": axes_map.correlations.contiguous(),
            "canonical_loadings": axes_map.canonical_loadings.contiguous(),
        }
        save_file(state, str(path))
        logger.info("AxesMap saved to %s", path)

    @staticmethod
    def load_axes_map(path: str | Path) -> AxesMap:
        """Load AxesMap from safetensors format.

        Args:
            path: Path to a ``.safetensors`` file saved by ``save_axes_map()``.

        Returns:
            Reconstructed AxesMap (axis_names restored from TETRAD_AXES).
        """
        from safetensors.torch import load_file  # type: ignore[import-untyped]

        state = load_file(str(path), device="cpu")
        return AxesMap(
            axis_names=TETRAD_AXES,
            feature_indices=state["feature_indices"],
            correlations=state["correlations"],
            canonical_loadings=state["canonical_loadings"],
        )


def identify_tetrad_axes(
    sae_features: Tensor,
    labels: Tensor,
    top_k: int = 32,
) -> AxesMap:
    """Convenience wrapper: fit CCA and return AxesMap.

    Args:
        sae_features: SAE latent activations [B, n_features].
        labels: One-hot Tetrad label matrix [B, 4].
        top_k: Number of top features to report per axis.

    Returns:
        AxesMap with identified Tetrad axis features.
    """
    finder = CCAAxisFinder(top_k=top_k)
    return finder.fit(sae_features, labels)
