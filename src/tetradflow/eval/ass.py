"""Axis Specificity Score (ASS) + CFG multi-direction baseline comparison (P1-3).

ASS measures how much the generated image activates the intended Tetrad axis
versus the other 3 axes, relative to a CFG multi-direction baseline.

P1-3 gate: TetradFlow ASS - CFG ASS bootstrap CI lower bound > 0.
If this condition is NOT met, human review is required (P0-4: no auto-degradation).
"""

from __future__ import annotations

import logging
from typing import NamedTuple

import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ASSResult(NamedTuple):
    """Result of Axis Specificity Score computation.

    Attributes:
        ass_mean: Mean ASS across samples.
        ass_std: Standard deviation of ASS across samples.
        ci_lower: Bootstrap 95% CI lower bound.
        ci_upper: Bootstrap 95% CI upper bound.
        gate_pass: True if CI lower > 0 (P1-3).
    """

    ass_mean: float
    ass_std: float
    ci_lower: float
    ci_upper: float
    gate_pass: bool


def axis_specificity_score(
    feature_activations: Tensor,
    target_axis_idx: int,
    axis_feature_indices: Tensor,
) -> Tensor:
    """Compute per-sample Axis Specificity Score.

    ASS_i = mean activation on target-axis features
            - mean activation on non-target-axis features

    Args:
        feature_activations: SAE latent activations [B, n_features].
        target_axis_idx: Which axis (0–3) is the intended generation axis.
        axis_feature_indices: Top-k feature indices per axis [4, top_k].

    Returns:
        Per-sample ASS tensor [B].
    """
    target_idx = axis_feature_indices[target_axis_idx]  # [top_k]
    other_idx = torch.cat(
        [axis_feature_indices[i] for i in range(4) if i != target_axis_idx]
    )  # [3 * top_k]

    target_acts = feature_activations[:, target_idx].mean(dim=1)  # [B]
    other_acts = feature_activations[:, other_idx].mean(dim=1)  # [B]

    return target_acts - other_acts  # [B]


def bootstrap_ci(
    samples: Tensor,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        samples: 1-D tensor of scalar observations.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (default 0.95 → 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple (ci_lower, ci_upper).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)

    n = samples.shape[0]
    boot_means = torch.zeros(n_bootstrap, dtype=torch.float32)
    for i in range(n_bootstrap):
        idx = torch.randint(0, n, (n,), generator=rng)
        boot_means[i] = samples[idx].mean()

    alpha = (1.0 - ci) / 2.0
    lower = float(boot_means.quantile(alpha).item())
    upper = float(boot_means.quantile(1.0 - alpha).item())
    return lower, upper


def compute_ass(
    tetrad_activations: Tensor,
    cfg_activations: Tensor,
    target_axis_idx: int,
    axis_feature_indices: Tensor,
    n_bootstrap: int = 1000,
) -> tuple[ASSResult, ASSResult]:
    """Compute ASS for TetradFlow and CFG baseline, with bootstrap CI for the difference.

    P1-3 gate: (TetradFlow ASS) - (CFG ASS) bootstrap CI lower > 0.

    Args:
        tetrad_activations: SAE features from TetradFlow generated images [B, n_features].
        cfg_activations: SAE features from CFG baseline images [B, n_features].
        target_axis_idx: Intended Tetrad axis index (0–3).
        axis_feature_indices: [4, top_k] feature indices per axis.
        n_bootstrap: Number of bootstrap resamples for CI.

    Returns:
        Tuple (tetrad_ass_result, cfg_ass_result). Both include CI.
        Call ``check_p1_3_gate()`` to evaluate the difference CI.
    """
    tetrad_scores = axis_specificity_score(
        tetrad_activations, target_axis_idx, axis_feature_indices
    )
    cfg_scores = axis_specificity_score(cfg_activations, target_axis_idx, axis_feature_indices)

    def _to_result(scores: Tensor) -> ASSResult:
        mean = float(scores.mean().item())
        std = float(scores.std().item())
        lo, hi = bootstrap_ci(scores, n_bootstrap=n_bootstrap)
        return ASSResult(
            ass_mean=mean,
            ass_std=std,
            ci_lower=lo,
            ci_upper=hi,
            gate_pass=lo > 0.0,
        )

    return _to_result(tetrad_scores), _to_result(cfg_scores)


def check_p1_3_gate(
    tetrad_activations: Tensor,
    cfg_activations: Tensor,
    target_axis_idx: int,
    axis_feature_indices: Tensor,
    n_bootstrap: int = 1000,
) -> dict[str, object]:
    """Check P1-3 gate: TetradFlow ASS - CFG ASS bootstrap CI lower > 0.

    Args:
        tetrad_activations: [B, n_features] from TetradFlow.
        cfg_activations: [B, n_features] from CFG baseline.
        target_axis_idx: Intended axis index.
        axis_feature_indices: [4, top_k].
        n_bootstrap: Bootstrap resamples.

    Returns:
        Dict with keys: tetrad_ass, cfg_ass, diff_ci_lower, diff_ci_upper, gate_pass.
    """
    tetrad_scores = axis_specificity_score(
        tetrad_activations, target_axis_idx, axis_feature_indices
    )
    cfg_scores = axis_specificity_score(cfg_activations, target_axis_idx, axis_feature_indices)

    # Bootstrap the difference of means
    diff_samples = tetrad_scores - cfg_scores  # [B] (assumes paired samples)
    diff_lo, diff_hi = bootstrap_ci(diff_samples, n_bootstrap=n_bootstrap)

    gate_pass = diff_lo > 0.0

    result: dict[str, object] = {
        "tetrad_ass_mean": float(tetrad_scores.mean().item()),
        "cfg_ass_mean": float(cfg_scores.mean().item()),
        "diff_mean": float(diff_samples.mean().item()),
        "diff_ci_lower": diff_lo,
        "diff_ci_upper": diff_hi,
        "gate_pass": gate_pass,
    }

    if not gate_pass:
        logger.warning(
            "P1-3 gate FAIL: diff CI lower = %.4f <= 0. "
            "TetradFlow ASS not distinguishable from CFG baseline. "
            "Human review required (P0-4).",
            diff_lo,
        )
    else:
        logger.info("P1-3 gate PASS: diff CI lower = %.4f > 0.", diff_lo)

    return result
