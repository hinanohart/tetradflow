"""Human evaluation utilities for TetradFlow (P2-1, P2-3).

Provides:
  - CSV template generator for n=30 forced-choice evaluation
  - Result loader and basic inter-rater agreement (Cohen's kappa) computation
  - Summary report generator

P2-1: McLuhan expert forced-choice (n=30, 4-axis perception rate)
P2-3: 4-divergent ODE output human eval suite (n=30)
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

TETRAD_AXES = ("Enhance", "Obsolesce", "Retrieve", "Reverse")

# Number of items per axis in the forced-choice template
ITEMS_PER_AXIS: int = 8  # 8 * 4 axes = 32 items (rounds to n=30 after exclusions)


def generate_forced_choice_csv(
    output_path: str | Path,
    n_per_axis: int = ITEMS_PER_AXIS,
    image_dir: str | Path | None = None,
) -> None:
    """Generate a blank forced-choice CSV template for human evaluation.

    The evaluator fills in the ``rater_choice`` column with one of:
    ``Enhance``, ``Obsolesce``, ``Retrieve``, ``Reverse``.

    Args:
        output_path: Destination ``.csv`` file path.
        n_per_axis: Number of items per Tetrad axis. Default 8 (→ 32 items).
        image_dir: Optional path to image directory. If provided, image
            filenames are pre-filled in ``image_path`` column.
    """
    output_path = Path(output_path)
    fieldnames = [
        "item_id",
        "axis_ground_truth",
        "prompt",
        "image_path",
        "rater_a_choice",
        "rater_b_choice",
        "notes",
    ]

    rows: list[dict[str, str]] = []
    item_num = 0
    for axis in TETRAD_AXES:
        for i in range(n_per_axis):
            item_num += 1
            item_id = f"{axis[:2].upper()}{i + 1:02d}"
            img_path = ""
            if image_dir is not None:
                img_path = str(Path(image_dir) / f"{item_id}.png")
            rows.append(
                {
                    "item_id": item_id,
                    "axis_ground_truth": axis,
                    "prompt": f"[FILL: {axis} example prompt {i + 1}]",
                    "image_path": img_path,
                    "rater_a_choice": "",
                    "rater_b_choice": "",
                    "notes": "",
                }
            )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Forced-choice CSV template written to %s (%d items)", output_path, len(rows))


def load_ratings(csv_path: str | Path) -> list[dict[str, str]]:
    """Load completed ratings from a forced-choice CSV.

    Args:
        csv_path: Path to a completed forced-choice CSV.

    Returns:
        List of row dicts with all columns.
    """
    rows: list[dict[str, str]] = []
    with Path(csv_path).open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def cohens_kappa(ratings: list[dict[str, str]]) -> dict[str, float]:
    """Compute Cohen's kappa between rater_a_choice and rater_b_choice.

    Filters out rows where either rater choice is blank.

    Args:
        ratings: List of row dicts from ``load_ratings()``.

    Returns:
        Dict with keys: ``kappa``, ``agreement``, ``n_rated``.

    Raises:
        ValueError: If fewer than 2 rated rows are found.
    """
    rated = [
        r
        for r in ratings
        if r.get("rater_a_choice", "").strip() and r.get("rater_b_choice", "").strip()
    ]
    n = len(rated)
    if n < 2:
        raise ValueError(f"Need at least 2 rated rows to compute kappa, got {n}.")

    categories = list(TETRAD_AXES)

    # Observed agreement
    agree = sum(1 for r in rated if r["rater_a_choice"] == r["rater_b_choice"])
    p_o = agree / n

    # Expected agreement
    a_counts = {c: 0 for c in categories}
    b_counts = {c: 0 for c in categories}
    for r in rated:
        ca = r["rater_a_choice"]
        cb = r["rater_b_choice"]
        if ca in a_counts:
            a_counts[ca] += 1
        if cb in b_counts:
            b_counts[cb] += 1

    p_e = sum((a_counts.get(c, 0) / n) * (b_counts.get(c, 0) / n) for c in categories)

    if abs(1.0 - p_e) < 1e-10:
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        "kappa": round(kappa, 4),
        "agreement": round(p_o, 4),
        "n_rated": n,
    }


def eval_summary(csv_path: str | Path) -> dict[str, Any]:
    """Load ratings and compute summary statistics.

    Args:
        csv_path: Completed forced-choice CSV path.

    Returns:
        Dict with kappa, agreement, per-axis accuracy, and P2-1 gate status.
    """
    ratings = load_ratings(csv_path)
    kappa_result = cohens_kappa(ratings)

    # Per-axis accuracy (rater A as proxy rater, ground truth as reference)
    axis_correct: dict[str, int] = {ax: 0 for ax in TETRAD_AXES}
    axis_total: dict[str, int] = {ax: 0 for ax in TETRAD_AXES}

    for r in ratings:
        gt = r.get("axis_ground_truth", "")
        ra = r.get("rater_a_choice", "")
        if gt in axis_total:
            axis_total[gt] += 1
            if ra == gt:
                axis_correct[gt] += 1

    per_axis_acc = {
        ax: round(axis_correct[ax] / axis_total[ax], 4) if axis_total[ax] > 0 else None
        for ax in TETRAD_AXES
    }

    gate_pass = kappa_result["kappa"] >= 0.7

    if not gate_pass:
        logger.warning(
            "P2-1 gate FAIL: Cohen's kappa = %.4f < 0.7. "
            "Tetrad axis operational definitions need revision. "
            "Human review required (P0-4).",
            kappa_result["kappa"],
        )
    else:
        logger.info("P2-1 gate PASS: kappa = %.4f >= 0.7.", kappa_result["kappa"])

    return {
        **kappa_result,
        "per_axis_accuracy": per_axis_acc,
        "p2_1_gate_pass": gate_pass,
    }


def write_summary_json(csv_path: str | Path, output_path: str | Path) -> None:
    """Write eval summary to JSON for CI artifact consumption.

    Args:
        csv_path: Completed forced-choice CSV.
        output_path: Destination JSON file path.
    """
    summary = eval_summary(csv_path)
    Path(output_path).write_text(json.dumps(summary, indent=2))
    logger.info("Human eval summary written to %s", output_path)
