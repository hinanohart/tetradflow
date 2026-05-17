"""P1-2 ablation: orthogonality of Tetrad axes under different identification methods.

Compares pairwise cosine similarity of the 4 Tetrad axis direction vectors
extracted by three methods, on the same SAE feature matrix:

1. ``cca_residual_pca`` — 3 CCA components + residual PCA top-1
                          (production path, P0-2/P0-3 compliant).
2. ``svd_top4``         — SVD top-4 of raw CCA loadings
                          (the ``ode.py::svd_top4_basis`` path).
3. ``gram_schmidt``     — Sequential Gram-Schmidt of raw CCA loadings
                          (P0-3 FORBIDDEN — kept as a baseline to demonstrate
                          axis-order dependence, NOT used in production).

Outputs a JSON report suitable for a paper appendix or research note.

Usage:
    python scripts/ablate_orthogonality.py \\
        --mode synthetic \\
        --output reports/ablate_orthogonality.json

    python scripts/ablate_orthogonality.py \\
        --mode from-sae \\
        --sae checkpoints/sae.safetensors \\
        --labels eval/manual_labels.json \\
        --output reports/ablate_orthogonality.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Number of Tetrad axes (Enhance / Obsolesce / Retrieve / Reverse).
N_AXES = 4
PAIR_LABELS = ("E-O", "E-Re", "E-Rv", "O-Re", "O-Rv", "Re-Rv")


def _pairwise_abs_cosines(loadings: torch.Tensor) -> list[float]:
    """Return the 6 absolute pairwise cosines between columns of loadings [n_features, 4]."""
    L = loadings
    norms = L.norm(dim=0, keepdim=True).clamp(min=1e-8)
    L_norm = L / norms
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    return [(L_norm[:, i] * L_norm[:, j]).sum().abs().item() for i, j in pairs]


def _gram_schmidt(vectors: torch.Tensor) -> torch.Tensor:
    """Sequential Gram-Schmidt orthogonalisation of rows of ``vectors`` [K, D].

    P0-3 FORBIDDEN in production. Implemented here only for ablation comparison.
    """
    K, D = vectors.shape
    basis = torch.zeros_like(vectors)
    for k in range(K):
        v = vectors[k].clone()
        for j in range(k):
            v = v - (v @ basis[j]) * basis[j]
        n = v.norm()
        basis[k] = v / n.clamp(min=1e-8)
    return basis


def _build_synthetic(n_samples: int = 240, n_features: int = 96, seed: int = 0):
    """Build a small synthetic SAE / label dataset for CPU smoke testing."""
    g = torch.Generator().manual_seed(seed)
    sae = torch.randn(n_samples, n_features, generator=g).abs()
    labels = torch.zeros(n_samples, N_AXES)
    for i in range(n_samples):
        labels[i, i % N_AXES] = 1.0
    return sae, labels


def _load_from_files(sae_path: Path, labels_path: Path):
    """Load SAE features and labels from a real CCA-ready dataset."""
    from safetensors.torch import load_file  # type: ignore[import-untyped]

    sae_state = load_file(str(sae_path))
    # By convention scripts/identify_axes_cca.py writes the activation matrix
    # under "sae_features" (see cli.py:139-146).
    sae_features = sae_state.get("sae_features")
    if sae_features is None:
        raise KeyError(
            f"{sae_path} does not contain 'sae_features' tensor. "
            "Use the activation matrix produced by scripts/identify_axes_cca.py."
        )

    suffix = labels_path.suffix.lower()
    if suffix == ".safetensors":
        ls = load_file(str(labels_path))
        labels = ls["labels"]
    elif suffix == ".json":
        raw = json.loads(labels_path.read_text())
        labels = torch.tensor(raw["labels"], dtype=torch.float32)
    else:
        raise ValueError(f"--labels must be .safetensors or .json, got {suffix!r}")
    return sae_features, labels


def _fit_cca_residual_pca(sae_features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Run production-path 3-CCA + residual PCA. Returns [n_features, 4] loadings."""
    from tetradflow.cca import CCAAxisFinder

    finder = CCAAxisFinder(top_k=8)
    axes_map = finder.fit(sae_features, labels)
    return axes_map.canonical_loadings  # [n_features, 4]


def _svd_top4(loadings_cca_only: torch.Tensor) -> torch.Tensor:
    """Take 4-component loadings and re-extract via SVD top-4. Returns [n_features, 4]."""
    # The SVD top-4 basis lives in the row space, so transpose: [4, n_features].
    L_T = loadings_cca_only.T  # [4, n_features]
    _U, _S, Vh = torch.linalg.svd(L_T, full_matrices=False)
    return Vh.T  # [n_features, 4]


def _run_ablation(sae_features: torch.Tensor, labels: torch.Tensor) -> dict:
    """Run the 3-way ablation and return JSON-serialisable report."""
    L_cca = _fit_cca_residual_pca(sae_features, labels)  # [n_features, 4]

    results: dict[str, dict] = {}

    # 1. CCA + residual PCA (production)
    cosines_cca = _pairwise_abs_cosines(L_cca)
    results["cca_residual_pca"] = {
        "description": "3 CCA components + residual PCA top-1 (P0-2 production)",
        "pairwise_cosines": cosines_cca,
        "max_cosine": max(cosines_cca),
        "passes_p0_2_gate": max(cosines_cca) < 0.3,
    }

    # 2. SVD top-4 on the same loadings (rotation-only, no info loss)
    L_svd = _svd_top4(L_cca)
    cosines_svd = _pairwise_abs_cosines(L_svd)
    results["svd_top4"] = {
        "description": "SVD top-4 rotation of the CCA loadings (P0-3 production)",
        "pairwise_cosines": cosines_svd,
        "max_cosine": max(cosines_svd),
        "passes_p0_2_gate": max(cosines_svd) < 0.3,
    }

    # 3. Gram-Schmidt forced orthogonalisation (FORBIDDEN, comparison only)
    L_gs = _gram_schmidt(L_cca.T).T  # gram_schmidt operates on rows
    cosines_gs = _pairwise_abs_cosines(L_gs)
    results["gram_schmidt"] = {
        "description": (
            "Sequential Gram-Schmidt of the CCA loadings. P0-3 FORBIDDEN in "
            "production (axis order matters; downstream meaning rotates). "
            "Reported here purely as a comparison baseline."
        ),
        "pairwise_cosines": cosines_gs,
        "max_cosine": max(cosines_gs),
        "passes_p0_2_gate": max(cosines_gs) < 0.3,
        "warning": (
            "By construction Gram-Schmidt forces orthogonality to numerical zero, "
            "so the cosine gate trivially passes — but the resulting axes are NOT "
            "semantically aligned with Tetrad axes. Do not use in production."
        ),
    }

    return {
        "n_samples": int(sae_features.shape[0]),
        "n_features": int(sae_features.shape[1]),
        "pair_labels": list(PAIR_LABELS),
        "methods": results,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["synthetic", "from-sae"], default="synthetic")
    p.add_argument("--sae", type=Path, default=None, help="SAE activation .safetensors")
    p.add_argument("--labels", type=Path, default=None, help="Labels .json or .safetensors")
    p.add_argument("--output", "-o", type=Path, required=True, help="Output JSON path")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.mode == "synthetic":
        sae, labels = _build_synthetic(seed=args.seed)
    else:
        if args.sae is None or args.labels is None:
            logger.error("--mode from-sae requires --sae and --labels")
            return 1
        sae, labels = _load_from_files(args.sae, args.labels)

    report = _run_ablation(sae, labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True))
    logger.info("Wrote %s", args.output)

    # Print summary table to stdout
    print(f"\n{'method':<24} {'max cos':>9} {'p0_2_gate':>11}")
    print("-" * 48)
    for name, data in report["methods"].items():
        gate = "PASS" if data["passes_p0_2_gate"] else "FAIL"
        print(f"{name:<24} {data['max_cosine']:>9.4f} {gate:>11}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
