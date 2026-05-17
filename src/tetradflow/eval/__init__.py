"""TetradFlow evaluation suite.

Sub-modules:
  direct_metrics  — P0-2 pairwise cosine orthogonality tests (pytest).
  ass             — Axis Specificity Score + CFG baseline comparison (P1-3).
  human_eval      — n=30 forced choice CSV template (P2-1, P2-3).
"""

from __future__ import annotations

__all__ = ["direct_metrics", "ass", "human_eval"]
