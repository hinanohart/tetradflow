"""Smoke test: scripts/ablate_orthogonality.py on synthetic data (CPU only)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_ablate_orthogonality_synthetic(tmp_path: Path) -> None:
    pytest.importorskip("sklearn")
    script = Path(__file__).parent.parent / "scripts" / "ablate_orthogonality.py"
    assert script.exists(), f"missing script: {script}"

    output = tmp_path / "report.json"
    result = subprocess.run(
        [sys.executable, str(script), "--mode", "synthetic", "--output", str(output)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"non-zero exit\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert output.exists()

    data = json.loads(output.read_text())
    assert "methods" in data
    assert set(data["methods"].keys()) == {
        "cca_residual_pca",
        "svd_top4",
        "gram_schmidt",
    }
    for name, method_data in data["methods"].items():
        cosines = method_data["pairwise_cosines"]
        assert len(cosines) == 6, f"{name}: 6 pairs expected, got {len(cosines)}"
        assert all(0.0 <= c <= 1.0 + 1e-5 for c in cosines), (
            f"{name}: cosines out of range: {cosines}"
        )
        assert "max_cosine" in method_data
        assert "passes_p0_2_gate" in method_data

    # Gram-Schmidt forces orthogonality numerically: its max cosine should be tiny.
    gs_max = data["methods"]["gram_schmidt"]["max_cosine"]
    assert gs_max < 1e-3, f"Gram-Schmidt should produce orthogonal axes, got {gs_max}"
