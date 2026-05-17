"""P0-2 direct orthogonality metrics for Tetrad axis SAE features.

Pytest tests that verify the 6 pairwise cosine similarities between the
4 Tetrad axis canonical loading vectors are all < 0.3.

Gate conditions (A1 metric):
  PASS: all 6 pairwise |cosine| < 0.3
  FAIL (trigger human review, P0-4):
    - median cosine > 0.5, OR
    - 2 or more pairs have |cosine| > 0.6

These tests run in CI (cpu-only). Heavy model loading is skipped;
a pre-computed AxesMap .safetensors must be provided via --axes-map.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch

# ---------------------------------------------------------------------------
# Pytest fixture / config hook
# ---------------------------------------------------------------------------


def pytest_addoption(parser: Any) -> None:
    """Add --axes-map CLI option for specifying the AxesMap file."""
    try:
        parser.addoption(
            "--axes-map",
            action="store",
            default=None,
            help="Path to AxesMap .safetensors for direct_metrics tests.",
        )
    except ValueError:
        # Option already added (happens when conftest also registers it)
        pass


@pytest.fixture(scope="module")
def axes_map(request: Any) -> Any:
    """Load AxesMap from --axes-map path, or skip if not provided."""
    axes_map_path = request.config.getoption("--axes-map", default=None)
    if axes_map_path is None:
        pytest.skip("No --axes-map provided; skipping direct_metrics tests.")

    path = Path(axes_map_path)
    if not path.exists():
        pytest.skip(f"AxesMap file not found: {path}")

    from tetradflow.cca import CCAAxisFinder

    return CCAAxisFinder.load_axes_map(path)


@pytest.fixture(scope="module")
def pairwise_cosines(axes_map: Any) -> torch.Tensor:
    """Compute the 6 pairwise cosine values from AxesMap."""
    from tetradflow.cca import CCAAxisFinder

    return CCAAxisFinder.pairwise_cosines(axes_map)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

PAIR_LABELS = [
    ("Enhance", "Obsolesce"),
    ("Enhance", "Retrieve"),
    ("Enhance", "Reverse"),
    ("Obsolesce", "Retrieve"),
    ("Obsolesce", "Reverse"),
    ("Retrieve", "Reverse"),
]

# P0-2 gate thresholds
_HARD_THRESHOLD = 0.3  # all 6 pairs must be below this
_MEDIAN_FAIL = 0.5  # median above this triggers human review
_PAIR_FAIL_COUNT = 2  # number of pairs above _PAIR_HIGH that triggers human review
_PAIR_HIGH = 0.6  # per-pair threshold for the count gate


@pytest.mark.parametrize(
    "pair_idx,label", enumerate(PAIR_LABELS), ids=[f"{a}-{b}" for a, b in PAIR_LABELS]
)
def test_pairwise_cosine_below_threshold(
    pairwise_cosines: torch.Tensor,
    pair_idx: int,
    label: tuple[str, str],
) -> None:
    """Each of the 6 axis-pair cosine similarities must be < 0.3 (P0-2 gate).

    Failing this test means the SAE features are not orthogonal enough.
    Human review is required before release (P0-4).
    """
    cosine = pairwise_cosines[pair_idx].item()
    assert cosine < _HARD_THRESHOLD, (
        f"Axis pair {label[0]}-{label[1]}: |cosine| = {cosine:.4f} >= {_HARD_THRESHOLD}. "
        "P0-2 gate FAIL. Human review required (P0-4)."
    )


def test_median_cosine_below_fail_threshold(pairwise_cosines: torch.Tensor) -> None:
    """Median pairwise cosine must be <= 0.5.

    If the median exceeds 0.5, the axis set is too correlated overall
    and the Tetrad alignment is not distinguishable from CFG multi-direction.
    """
    median = pairwise_cosines.median().item()
    assert median <= _MEDIAN_FAIL, (
        f"Median pairwise cosine = {median:.4f} > {_MEDIAN_FAIL}. "
        "Axis directions are globally too correlated. Human review required (P0-4)."
    )


def test_no_two_pairs_above_high_threshold(pairwise_cosines: torch.Tensor) -> None:
    """At most 1 pair is allowed to have |cosine| > 0.6.

    Having 2+ pairs above 0.6 indicates near-collapse of the axis space.
    """
    high_count = int((pairwise_cosines > _PAIR_HIGH).sum().item())
    assert high_count < _PAIR_FAIL_COUNT, (
        f"{high_count} pairs have |cosine| > {_PAIR_HIGH} (threshold: < {_PAIR_FAIL_COUNT}). "
        "Axis space near-collapse. Human review required (P0-4)."
    )


# ---------------------------------------------------------------------------
# Standalone cosine summary (importable, not pytest)
# ---------------------------------------------------------------------------


def cosine_summary(axes_map: Any) -> dict[str, float]:
    """Return a dict of pair-label -> cosine for reporting.

    Args:
        axes_map: AxesMap from CCAAxisFinder.

    Returns:
        Dict mapping ``"Enhance-Obsolesce"`` etc. to cosine float value.
    """
    from tetradflow.cca import CCAAxisFinder

    cosines = CCAAxisFinder.pairwise_cosines(axes_map)
    return {f"{a}-{b}": float(cosines[i].item()) for i, (a, b) in enumerate(PAIR_LABELS)}


def write_eval_json(axes_map: Any, output_path: str | Path) -> None:
    """Write cosine summary to JSON for CI artifact consumption.

    The JSON is consumed by scripts/human_gate_check.py.

    Args:
        axes_map: AxesMap from CCAAxisFinder.
        output_path: Destination JSON file path.
    """
    summary = cosine_summary(axes_map)
    cosines = list(summary.values())
    median = float(torch.tensor(cosines).median().item())
    n_high = int(sum(1 for c in cosines if c > _PAIR_HIGH))

    payload = {
        "pairwise_cosines": summary,
        "median_cosine": median,
        "n_pairs_above_0_6": n_high,
        "gate_pass": all(c < _HARD_THRESHOLD for c in cosines) and median <= _MEDIAN_FAIL,
    }

    Path(output_path).write_text(json.dumps(payload, indent=2))
