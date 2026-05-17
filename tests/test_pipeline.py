"""Smoke tests for TetradFlowPipeline (import + init only, no model load)."""

from __future__ import annotations

import pytest


def test_pipeline_import() -> None:
    from tetradflow.pipeline import TetradFlowPipeline

    assert TetradFlowPipeline is not None


def test_pipeline_init_defaults() -> None:
    from tetradflow.pipeline import TetradFlowPipeline

    p = TetradFlowPipeline()
    # Default is bf16 (FP8 is a P1 milestone, not yet implemented)
    assert p.vram_mode == "bf16"
    assert p.gamma == 2.0
    assert not p._loaded


def test_pipeline_init_custom() -> None:
    from tetradflow.pipeline import TetradFlowPipeline

    p = TetradFlowPipeline(
        janus_model_id="test/model",
        flux_model_id="test/flux",
        vram_mode="bf16",
        gamma=1.5,
        device="cpu",
    )
    assert p.janus_model_id == "test/model"
    assert p.vram_mode == "bf16"
    assert p.gamma == 1.5
    assert p.device == "cpu"


def test_pipeline_generate_without_load_raises() -> None:
    from tetradflow.pipeline import TetradFlowPipeline

    p = TetradFlowPipeline(device="cpu")
    with pytest.raises(RuntimeError, match="not loaded"):
        p.generate("test prompt")


def test_pipeline_unload_safe_without_load() -> None:
    """unload() on a never-loaded pipeline must not raise."""
    from tetradflow.pipeline import TetradFlowPipeline

    p = TetradFlowPipeline(device="cpu")
    p.unload()  # Should not raise


@pytest.mark.gpu
def test_pipeline_load_gpu() -> None:
    """Full pipeline load — requires GPU and model downloads. Marked @gpu."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # This test requires actual HF model access; skip in offline CI.
    pytest.skip("Skipping model download in CI; run manually with GPU + HF access.")
