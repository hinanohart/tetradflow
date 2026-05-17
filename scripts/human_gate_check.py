"""P0-4 Human Gate Check for TetradFlow release pipeline.

Reads a CI eval JSON artifact (written by eval/direct_metrics.py or
scripts/identify_axes_cca.py --eval-json) and exits 1 if the P0-2
cosine gate has failed.

Called by .github/workflows/release.yml BEFORE any publish step.
On failure: prints a GitHub Actions annotation, exits 1 to block release.
Human review and explicit re-trigger are required to proceed.

NO automatic degradation is performed (P0-4 constraint).

Usage:
  python scripts/human_gate_check.py --eval-json eval_output.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P0-4 human gate check. Exits 1 on gate failure to block release."
    )
    p.add_argument(
        "--eval-json",
        required=True,
        help="Path to eval JSON artifact from direct_metrics.write_eval_json()",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _github_actions_error(message: str) -> None:
    """Print GitHub Actions error annotation if running in CI."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error::{message}", flush=True)
    else:
        print(f"ERROR: {message}", flush=True)


def _github_actions_notice(message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::notice::{message}", flush=True)
    else:
        print(f"NOTICE: {message}", flush=True)


def check_gate(eval_json_path: str) -> bool:
    """Load eval JSON and check the gate conditions.

    Returns:
        True if gate passes, False if it fails.
    """
    path = Path(eval_json_path)
    if not path.exists():
        _github_actions_error(
            f"Eval JSON not found: {path}. "
            "Run identify_axes_cca.py --eval-json or direct_metrics.write_eval_json() first."
        )
        return False

    try:
        with path.open() as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        _github_actions_error(f"Failed to parse eval JSON {path}: {exc}")
        return False

    gate_pass: bool = data.get("gate_pass", False)
    median: float = data.get("median_cosine", 999.0)
    n_high: int = data.get("n_pairs_above_0_6", 0)
    pairwise: dict[str, float] = data.get("pairwise_cosines", {})

    print("\n=== TetradFlow P0-4 Human Gate Check ===")
    print(f"  Eval JSON: {path}")
    print(f"  Median pairwise cosine: {median:.4f}")
    print(f"  Pairs above 0.6: {n_high}")
    print()
    for pair, cosine in pairwise.items():
        status = "PASS" if cosine < 0.3 else "FAIL"
        print(f"  {pair}: {cosine:.4f}  [{status}]")

    if gate_pass:
        _github_actions_notice(
            "P0-4 gate: PASS. Cosine orthogonality verified. Release may proceed."
        )
        print("\nP0-4 gate: PASS")
        return True
    else:
        _github_actions_error(
            "P0-4 gate: FAIL. "
            "Tetrad axis cosine orthogonality check failed. "
            "AUTOMATIC DEGRADATION IS FORBIDDEN (P0-4). "
            "Human review required. "
            "Fix the SAE / CCA configuration, re-run eval, and manually re-trigger the release."
        )
        print(
            "\nP0-4 gate: FAIL\n"
            "Action required:\n"
            "  1. Review pairwise cosine values above.\n"
            "  2. Re-train SAE or adjust CCA top_k.\n"
            "  3. Re-run: python scripts/identify_axes_cca.py --eval-json <path>\n"
            "  4. Manually approve and re-trigger the release workflow.\n"
            "  DO NOT bypass this gate or auto-degrade (P0-4).",
            file=sys.stderr,
        )
        return False


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    passed = check_gate(args.eval_json)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
