"""TetradFlow Gradio demo for HuggingFace Spaces (ZeroGPU).

Provides a 4-axis dropdown + prompt textbox + 4-image gallery interface.
Model is loaded globally at cold start to amortize load time across requests.

VRAM: Janus-Pro 7B + Flux 12B in BF16 requires ~52 GB (A100 80GB recommended).
FP8 quantization via torchao is a P1 milestone (NOT yet implemented).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import gradio as gr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model state (loaded once at cold start)
# ---------------------------------------------------------------------------

_PIPELINE: Any = None
_LOAD_ERROR: str | None = None

TETRAD_AXES = ["Enhance", "Obsolesce", "Retrieve", "Reverse"]
AXIS_DESCRIPTIONS = {
    "Enhance": "What does the medium amplify or intensify?",
    "Obsolesce": "What does the medium push aside or make obsolete?",
    "Retrieve": "What does the medium resurrect from the past?",
    "Reverse": "What does the medium become when pushed to its limits?",
}

HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")  # e.g. "username/tetradflow-sae"


def _load_pipeline() -> None:
    global _PIPELINE, _LOAD_ERROR
    try:
        from tetradflow.pipeline import TetradFlowPipeline

        sae_path = None
        axes_map_path = None

        # Download SAE and AxesMap from HF Hub if configured
        if HF_MODEL_REPO:
            try:
                from huggingface_hub import hf_hub_download  # type: ignore[import-untyped]

                sae_path = hf_hub_download(HF_MODEL_REPO, "sae.safetensors")
                axes_map_path = hf_hub_download(HF_MODEL_REPO, "axes_map.safetensors")
                logger.info("Downloaded SAE artifacts from %s", HF_MODEL_REPO)
            except Exception as exc:
                logger.warning("Could not download SAE artifacts: %s. Running without SAE.", exc)

        _PIPELINE = TetradFlowPipeline(
            sae_path=sae_path,
            axes_map_path=axes_map_path,
            vram_mode="bf16",
        )
        _PIPELINE.load()
        logger.info("Pipeline loaded successfully.")
    except Exception as exc:
        _LOAD_ERROR = str(exc)
        logger.error("Pipeline load failed: %s", exc)


# C1 (2026-05-17): pipeline is lazy-loaded on first generate() call, not at
# module import. Synchronous cold-start load of Janus-Pro 7B (~14 GB) inside
# module import would crash the Spaces container before Gradio can render the
# error UI (the `_LOAD_ERROR` banner below cannot reach the user if the import
# itself raises). Lazy load keeps the UI alive so the user sees the failure.
def _ensure_pipeline_loaded() -> None:
    if _PIPELINE is None and _LOAD_ERROR is None:
        _load_pipeline()


# ---------------------------------------------------------------------------
# Generation function
# ---------------------------------------------------------------------------

try:
    import spaces  # type: ignore[import-untyped]

    # C2 (2026-05-17): duration=120s budgets 1 axis of Janus+Flux at 4 steps.
    # When the T4 4-divergent ODE wiring lands and "All (4 axes)" runs all 4
    # back-to-back inside one GPU lease, bump this to ~300s or split the loop
    # into 4 separate @spaces.GPU calls (one per axis) so each gets its own
    # 120s lease.
    GPU_DECORATOR = spaces.GPU(duration=120)
except ImportError:
    # Not running on HF Spaces — identity decorator
    def GPU_DECORATOR(fn):  # type: ignore[misc]
        return fn


@GPU_DECORATOR
def generate_tetrad(
    prompt: str,
    mode: str,
    axis: str,
    steps: int,
    guidance: float,
    seed: int,
) -> list[Any]:
    """Generate images for all 4 Tetrad axes (or CFG baseline).

    Returns a list of 4 PIL images.
    """
    _ensure_pipeline_loaded()  # C1: lazy first-call load
    if _LOAD_ERROR is not None:
        raise gr.Error(f"Pipeline failed to load: {_LOAD_ERROR}")
    if _PIPELINE is None:
        raise gr.Error("Pipeline not loaded. Please retry.")

    if not prompt.strip():
        raise gr.Error("Please enter a text prompt.")

    images = []
    if mode == "Tetrad (4-axis)":
        # axis=="All (4 axes)" → generate one image per Tetrad axis
        # axis==single axis name → generate only that axis (returns 1-element list)
        target_axes = TETRAD_AXES if axis == "All (4 axes)" else [axis]
        for ax in target_axes:
            # P0-4: no silent fallback — RuntimeError / NotImplementedError propagates to user
            img = _PIPELINE.generate(
                prompt=prompt,
                axis=ax,
                mode="tetrad",
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )
            images.append(img)
    else:
        # CFG baseline — 4 images with same prompt (one per axis slot for comparison)
        for _ in TETRAD_AXES:
            img = _PIPELINE.generate(
                prompt=prompt,
                mode="cfg_baseline",
                num_inference_steps=steps,
                guidance_scale=guidance,
                seed=seed,
            )
            images.append(img)

    return images


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="TetradFlow Demo",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown(
        "# TetradFlow Demo\n"
        "**Research prototype** — not a validated cognitive model. "
        "[Disclaimer](https://github.com/tetradflow-dev/tetradflow/blob/main/docs/disclaimer.md)\n\n"
        "Generate images guided by McLuhan's 4 Tetrad axes using Janus-Pro + Flux."
    )

    with gr.Row():
        with gr.Column(scale=1):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="A photograph of a city at night...",
                lines=3,
            )
            mode_radio = gr.Radio(
                choices=["Tetrad (4-axis)", "CFG Baseline"],
                value="Tetrad (4-axis)",
                label="Generation Mode",
                info="Tetrad uses 4-divergent ODE; CFG Baseline for comparison (P1-3).",
            )
            axis_dropdown = gr.Dropdown(
                choices=["All (4 axes)", "Enhance", "Obsolesce", "Retrieve", "Reverse"],
                value="All (4 axes)",
                label="Axis",
                info="Select a single Tetrad axis or generate all 4 in one click.",
            )
            axis_info = gr.Markdown(
                "\n".join(f"**{ax}**: {desc}" for ax, desc in AXIS_DESCRIPTIONS.items())
            )
            with gr.Accordion("Advanced", open=False):
                steps_slider = gr.Slider(
                    minimum=1,
                    maximum=20,
                    value=4,
                    step=1,
                    label="Inference Steps",
                )
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.5,
                    label="Guidance Scale",
                )
                seed_number = gr.Number(
                    value=42,
                    precision=0,
                    label="Seed",
                )
            generate_btn = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            gallery = gr.Gallery(
                label="Generated Images (Enhance / Obsolesce / Retrieve / Reverse)",
                columns=2,
                rows=2,
                height=600,
            )
            axis_labels = gr.Markdown(
                "**Top-left**: Enhance  |  **Top-right**: Obsolesce  |  "
                "**Bottom-left**: Retrieve  |  **Bottom-right**: Reverse"
            )

    generate_btn.click(
        fn=generate_tetrad,
        inputs=[
            prompt_input,
            mode_radio,
            axis_dropdown,
            steps_slider,
            guidance_slider,
            seed_number,
        ],
        outputs=[gallery],
    )

    gr.Markdown(
        "---\n"
        "**Citation**: TetradFlow (2026). "
        "[GitHub](https://github.com/tetradflow-dev/tetradflow) | "
        "[PyPI](https://pypi.org/project/tetradflow/) | MIT License"
    )

if __name__ == "__main__":
    demo.launch()
