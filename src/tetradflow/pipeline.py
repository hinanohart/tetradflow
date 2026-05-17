"""TetradFlowPipeline: Janus-Pro + Flux integration (v0.1 PoC scope).

Orchestrates the generation flow currently in scope:
1. Janus-Pro encodes image/text via SAE hook at layer 20
2. CCA-derived Tetrad axes guide 4-divergent ODE sampling (PoC T4 wiring)
3. Flux DiT executes the velocity field
4. JanusGaugeFlip selects Figure/Ground blend

NOT YET INTEGRATED (post-PoC milestones, see ROADMAP.md):
- Show-o bidirectional text grounding (v1.0 contingent on P0 pass)
- LanguageBind multimodal grounding (v1.0 contingent on P0 pass)

VRAM modes:
- BF16 (full): ~52 GB — requires A100/H100 multi-GPU
  (FP8 quantization via torchao = P1 milestone, NOT yet implemented)

P0-4: generate() never auto-degrades. If a component fails to load,
      it raises RuntimeError and requires user explicit fallback choice.

Security note: pipeline loads Janus-Pro with ``trust_remote_code=True``,
which executes arbitrary Python from the HF model repository. Pin a
commit SHA for production use. See README.md "Security" section.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import torch

from tetradflow.hooks import JANUS_HOOK_LAYER, JanusActivationHook

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
VramMode = Literal["bf16"]
GenerationMode = Literal["tetrad", "cfg_baseline"]


class TetradFlowPipeline:
    """Unified Tetrad-guided generation pipeline.

    Args:
        janus_model_id: HuggingFace model ID for Janus-Pro (default 7B).
        flux_model_id: HuggingFace model ID for Flux.1-schnell.
        sae_path: Path to trained SAE ``.safetensors`` file.
        axes_map_path: Path to CCA-derived AxesMap ``.safetensors`` file.
        vram_mode: ``'bf16'`` for BF16 full precision (~52 GB, A100/H100 recommended).
            FP8 quantization is a P1 milestone (torchao) and not yet implemented.
        device: Torch device string. Defaults to 'cuda' if available else 'cpu'.
        gamma: Tetrad injection strength for the ODE step. Default 2.0.
    """

    def __init__(
        self,
        janus_model_id: str = "deepseek-ai/Janus-Pro-7B",
        flux_model_id: str = "black-forest-labs/FLUX.1-schnell",
        sae_path: str | Path | None = None,
        axes_map_path: str | Path | None = None,
        vram_mode: VramMode = "bf16",
        device: str | None = None,
        gamma: float = 2.0,
    ) -> None:
        self.janus_model_id = janus_model_id
        self.flux_model_id = flux_model_id
        self.sae_path = Path(sae_path) if sae_path else None
        self.axes_map_path = Path(axes_map_path) if axes_map_path else None
        self.vram_mode = vram_mode
        self.gamma = gamma

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Components — None until load() is called
        self._janus: Any = None
        self._janus_processor: Any = None
        self._flux_pipe: Any = None
        self._sae: Any = None
        self._axes_map: Any = None
        self._gauge: Any = None
        self._janus_hook: JanusActivationHook | None = None

        self._loaded = False

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load all pipeline components.

        P0-4: Raises RuntimeError on any component failure instead of
        silently degrading.
        """
        logger.info(
            "Loading TetradFlowPipeline (vram_mode=%s, device=%s)", self.vram_mode, self.device
        )
        self._load_janus()
        self._load_flux()
        if self.sae_path is not None:
            self._load_sae()
        if self.axes_map_path is not None:
            self._load_axes_map()
        self._load_gauge()
        self._loaded = True
        logger.info("TetradFlowPipeline loaded successfully.")

    def _dtype(self) -> torch.dtype:
        return torch.bfloat16

    def _load_janus(self) -> None:
        """Load Janus-Pro model and processor, then attach activation hook."""
        try:
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForCausalLM,
                AutoProcessor,
            )

            logger.info("Loading Janus-Pro from %s", self.janus_model_id)
            self._janus_processor = AutoProcessor.from_pretrained(
                self.janus_model_id, trust_remote_code=True
            )
            self._janus = AutoModelForCausalLM.from_pretrained(
                self.janus_model_id,
                torch_dtype=self._dtype(),
                device_map=self.device,
                trust_remote_code=True,
            )
            self._janus.eval()
            # Attach activation hook at layer JANUS_HOOK_LAYER (20) for SAE input
            self._janus_hook = JanusActivationHook(self._janus, layer_idx=JANUS_HOOK_LAYER)
            logger.info("Janus-Pro loaded (hook attached at layer %d).", JANUS_HOOK_LAYER)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Janus-Pro ({self.janus_model_id}). "
                "P0-4: manual fallback decision required."
            ) from exc

    def _load_flux(self) -> None:
        """Load Flux.1-schnell pipeline."""
        try:
            from diffusers import FluxPipeline  # type: ignore[import-untyped]

            logger.info("Loading Flux from %s", self.flux_model_id)
            self._flux_pipe = FluxPipeline.from_pretrained(
                self.flux_model_id,
                torch_dtype=self._dtype(),
            )
            self._flux_pipe.to(self.device)
            logger.info("Flux loaded.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load Flux ({self.flux_model_id}). "
                "P0-4: manual fallback decision required."
            ) from exc

    def _load_sae(self) -> None:
        """Load trained BatchTopK SAE."""
        from tetradflow.sae import BatchTopKSAE

        assert self.sae_path is not None
        logger.info("Loading SAE from %s", self.sae_path)
        self._sae = BatchTopKSAE.load(str(self.sae_path))
        self._sae.to(self.device)
        self._sae.eval()

    def _load_axes_map(self) -> None:
        """Load CCA-derived AxesMap."""
        from tetradflow.cca import CCAAxisFinder

        assert self.axes_map_path is not None
        logger.info("Loading AxesMap from %s", self.axes_map_path)
        self._axes_map = CCAAxisFinder.load_axes_map(self.axes_map_path)

    def _load_gauge(self) -> None:
        """Instantiate JanusGaugeFlip."""
        from tetradflow.gauge import JanusGaugeFlip

        self._gauge = JanusGaugeFlip(sae_n_axes=4).to(self.device)
        self._gauge.eval()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        axis: Literal["Enhance", "Obsolesce", "Retrieve", "Reverse"] | None = None,
        mode: GenerationMode = "tetrad",
        num_inference_steps: int = 4,
        guidance_scale: float = 3.5,
        height: int = 512,
        width: int = 512,
        seed: int | None = None,
    ) -> Any:
        """Generate an image using Tetrad-guided or CFG-baseline sampling.

        Args:
            prompt: Text prompt for generation.
            axis: Which Tetrad axis to emphasise. Required when
                ``mode='tetrad'``.
            mode: ``'tetrad'`` applies 4-divergent ODE injection;
                ``'cfg_baseline'`` runs standard Flux CFG for comparison
                (P1-3 baseline).
            num_inference_steps: Number of denoising steps.
            guidance_scale: CFG guidance scale.
            height: Output image height in pixels.
            width: Output image width in pixels.
            seed: Optional random seed for reproducibility.

        Returns:
            PIL.Image.Image output from the pipeline.

        Raises:
            RuntimeError: If pipeline is not loaded or required components
                are missing for the requested mode.
        """
        if not self._loaded:
            raise RuntimeError("Pipeline not loaded. Call pipeline.load() first.")

        if mode == "tetrad" and self._sae is None:
            raise RuntimeError(
                "SAE not loaded. Provide sae_path= to use tetrad mode. "
                "P0-4: manual fallback decision required."
            )

        if mode == "tetrad" and self._axes_map is None:
            raise RuntimeError(
                "AxesMap not loaded. Provide axes_map_path= to use tetrad mode. "
                "P0-4: manual fallback decision required."
            )

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        if mode == "cfg_baseline":
            return self._generate_cfg_baseline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        # Tetrad mode
        return self._generate_tetrad(
            prompt=prompt,
            axis=axis,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

    def _generate_cfg_baseline(
        self,
        prompt: str,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        generator: Any,
    ) -> Any:
        """Standard Flux CFG generation (P1-3 baseline)."""
        assert self._flux_pipe is not None
        result = self._flux_pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )
        return result.images[0]

    def _generate_tetrad(
        self,
        prompt: str,
        axis: str | None,
        num_inference_steps: int,
        guidance_scale: float,
        height: int,
        width: int,
        generator: Any,
    ) -> Any:
        """Tetrad-guided generation with 4-divergent ODE injection.

        Integration chain (PoC milestone T4):
        1. Run Janus-Pro forward to populate JanusActivationHook cache.
        2. get() activation → SAE forward → CCA-derived axis features.
        3. Build SVD top-4 basis from axis decoder directions.
        4. Inject into Flux velocity field via tetrad_step (Flux callback_on_step_end).

        Full Flux callback wiring requires diffusers FluxPipeline.callback_on_step_end
        which provides (pipeline, step_index, timestep, callback_kwargs).
        This is PoC milestone T4 — the hook chain below is the integration skeleton.

        Raises:
            NotImplementedError: Flux callback_on_step_end wiring is PoC milestone T4.
                Use mode='cfg_baseline' for runnable generation until T4 is complete.
        """
        assert self._flux_pipe is not None
        assert self._axes_map is not None
        assert self._sae is not None
        assert self._janus_hook is not None

        logger.info(
            "Tetrad generation: prompt=%r axis=%s steps=%d",
            prompt[:60],
            axis,
            num_inference_steps,
        )

        # --- Integration skeleton (Flux callback wiring, PoC milestone T4) ---
        #
        # Step 1: Janus forward populates hook cache
        #   self._janus(**janus_inputs)
        #   act = self._janus_hook.get()  # [batch, hidden_dim]
        #
        # Step 2: SAE encode → sparse features → CCA axis activations
        #   sae_out = self._sae(act)
        #   axis_feature_idx = self._axes_map.feature_indices  # [4, top_k]
        #   axis_acts = sae_out.z_topk[:, axis_feature_idx[:, 0]]  # [batch, 4]
        #
        # Step 3: Build SVD top-4 basis from axis decoder directions
        #   W_dec = self._sae._effective_W_dec()  # [n_features, input_dim]
        #   tetrad_dirs = W_dec[axis_feature_idx[:, 0]]  # [4, input_dim]
        #   basis = svd_top4_basis(tetrad_dirs)  # [4, D]
        #
        # Step 4: Inject via Flux callback
        #   def _tetrad_callback(pipe, step, t, kwargs):
        #       latents = kwargs["latents"]
        #       v_cfg = kwargs.get("noisy_residual", latents)
        #       x_next = tetrad_step(latents, t, dt, v_cfg, v_cfg, v_cfg, basis, self.gamma)
        #       return {"latents": x_next}
        #
        #   result = self._flux_pipe(
        #       prompt=prompt, ...,
        #       callback_on_step_end=_tetrad_callback,
        #       callback_on_step_end_tensor_inputs=["latents"],
        #   )
        #
        # The above pattern compiles but Flux callback API shape varies by diffusers version.
        # Activating this code path requires T4 integration testing on a GPU node.
        # ------------------------------------------------------------------

        raise NotImplementedError(
            "Tetrad mode requires Flux callback_on_step_end wiring (PoC milestone T4). "
            "The JanusActivationHook is attached and the SAE/CCA/ODE chain is ready; "
            "see _generate_tetrad() docstring for the full integration skeleton. "
            "Use mode='cfg_baseline' for runnable generation until T4 is complete."
        )

    # ------------------------------------------------------------------
    # Context manager support
    # ------------------------------------------------------------------

    def __enter__(self) -> TetradFlowPipeline:
        self.load()
        return self

    def __exit__(self, *args: object) -> None:
        self.unload()

    def unload(self) -> None:
        """Release model references and free GPU memory."""
        if self._janus_hook is not None:
            self._janus_hook.remove()
            self._janus_hook = None
        self._janus = None
        self._janus_processor = None
        self._flux_pipe = None
        self._sae = None
        self._axes_map = None
        self._gauge = None
        self._loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("TetradFlowPipeline unloaded.")
