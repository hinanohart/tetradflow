"""TetradFlow: McLuhan Tetrad as inductive bias in Janus-Pro + Flux.

Research prototype — not a validated cognitive model.
See docs/disclaimer.md for scope limitations.
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"
__author__ = "TetradFlow contributors"
__license__ = "MIT"

# Public re-exports (lazy to avoid heavy model imports at package load time)
__all__ = [
    "__version__",
    "BatchTopKSAE",
    "CCAAxisFinder",
    "svd_top4_basis",
    "tetrad_step",
    "JanusGaugeFlip",
    "JanusActivationHook",
    "TetradFlowPipeline",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy import for heavy submodules (explicit per-name imports, no dynamic paths)."""
    if name == "BatchTopKSAE":
        from tetradflow.sae import BatchTopKSAE

        return BatchTopKSAE
    if name == "CCAAxisFinder":
        from tetradflow.cca import CCAAxisFinder

        return CCAAxisFinder
    if name == "svd_top4_basis":
        from tetradflow.ode import svd_top4_basis

        return svd_top4_basis
    if name == "tetrad_step":
        from tetradflow.ode import tetrad_step

        return tetrad_step
    if name == "JanusGaugeFlip":
        from tetradflow.gauge import JanusGaugeFlip

        return JanusGaugeFlip
    if name == "JanusActivationHook":
        from tetradflow.hooks import JanusActivationHook

        return JanusActivationHook
    if name == "TetradFlowPipeline":
        from tetradflow.pipeline import TetradFlowPipeline

        return TetradFlowPipeline
    raise AttributeError(f"module 'tetradflow' has no attribute {name!r}")
