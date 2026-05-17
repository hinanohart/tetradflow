"""Janus-Pro activation hooks for residual stream extraction.

Hooks into layer 20 (mid-point of 30-layer LM stack) at the per-sample last
real text-token position to capture the representation used by the SAE.

F3 fix (2026-05-17): the previous implementation used ``hidden[:, -1, :]``
which silently picked the sequence-end position for every batch element,
mishandling left/right padding (and producing meaningless SAE input for
shorter samples). The fix routes an ``attention_mask`` through
``set_attention_mask()`` so the hook can gather each sample's true last
real token via per-row index.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import Tensor

logger = logging.getLogger(__name__)

# Target layer index (0-indexed). Janus-Pro has 30 LM layers; layer 20 is mid.
JANUS_HOOK_LAYER: int = 20


class JanusActivationHook:
    """Forward hook that caches residual-stream activations from Janus-Pro.

    Attaches to the output of a single transformer layer and stores each
    sample's last-real-text-token hidden state so the SAE can consume it.

    Usage::

        hook = JanusActivationHook(model, layer_idx=20)
        hook.set_attention_mask(inputs["attention_mask"])  # [B, seq_len]
        with torch.no_grad():
            model(**inputs)
        activation = hook.get()  # Tensor[batch, hidden_dim]
        hook.remove()

    If ``set_attention_mask()`` is not called, the hook falls back to
    ``hidden[:, -1, :]`` (correct only when batch size is 1 or every sample
    has identical length with right padding stripped).

    Args:
        model: A Janus-Pro model with ``model.language_model.model.layers`` list.
        layer_idx: Which layer to hook (default: ``JANUS_HOOK_LAYER`` = 20).
    """

    def __init__(
        self,
        model: Any,
        layer_idx: int = JANUS_HOOK_LAYER,
    ) -> None:
        self._cache: dict[str, Tensor] = {}
        self._attention_mask: Tensor | None = None

        # Janus-Pro wraps a causal LM; access the transformer layers.
        try:
            target_layer = model.language_model.model.layers[layer_idx]
        except (AttributeError, IndexError) as exc:
            raise ValueError(
                f"Cannot locate layer {layer_idx} in model. "
                "Expected model.language_model.model.layers to exist."
            ) from exc

        self._handle = target_layer.register_forward_hook(self._hook_fn)
        logger.debug("JanusActivationHook attached to layer %d", layer_idx)

    # ------------------------------------------------------------------
    # Public API: attention mask routing
    # ------------------------------------------------------------------

    def set_attention_mask(self, mask: Tensor) -> None:
        """Provide the input attention mask for per-sample last-token gather.

        One-shot: the mask is consumed by ``_hook_fn`` on the **next** forward
        pass and then cleared. Callers MUST re-call this before each forward;
        otherwise the hook falls back to ``hidden[:, -1, :]`` (correct only for
        batch=1 / uniform-length sequences). This protects multi-batch training
        loops from silently reusing a stale mask after the batch dims change.

        Args:
            mask: Tensor of shape ``[batch, seq_len]`` with 1 for real tokens
                and 0 for padding (the standard HuggingFace convention).
        """
        if mask.ndim != 2:
            raise ValueError(f"attention_mask must be [B, seq_len], got {mask.shape}")
        self._attention_mask = mask

    def clear_attention_mask(self) -> None:
        """Drop any cached attention mask (revert to fallback last-token mode)."""
        self._attention_mask = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _hook_fn(
        self,
        module: Any,  # noqa: ANN401
        inputs: tuple[Any, ...],
        output: Any,  # noqa: ANN401
    ) -> None:
        """Registered forward hook callback."""
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: [batch, seq_len, hidden_dim]

        if self._attention_mask is not None:
            mask = self._attention_mask
            B = hidden.shape[0]
            if mask.shape[0] != B:
                raise ValueError(
                    f"attention_mask batch dim {mask.shape[0]} != hidden batch dim {B}. "
                    "Call set_attention_mask() with the same batch as the forward pass."
                )
            # Per-sample index of the last real token = sum(mask) - 1.
            # Works for both left- and right-padding (mask is 1 on real tokens regardless).
            last_idx = mask.to(hidden.device).sum(dim=1).long() - 1  # [B]
            # Guard against all-padding rows (would give -1).
            last_idx = last_idx.clamp(min=0)
            batch_idx = torch.arange(B, device=hidden.device)
            gathered = hidden[batch_idx, last_idx]  # [B, hidden_dim]
            self._cache["_latest"] = gathered.detach().clone()
            # One-shot consume: clear so the next forward without a re-set falls
            # back instead of silently reusing the now-stale mask (e.g. when the
            # next batch has a different batch dim or different padding layout).
            self._attention_mask = None
        else:
            # Fallback: no mask supplied — single-sample or pre-stripped batches only.
            self._cache["_latest"] = hidden[:, -1, :].detach().clone()

    # ------------------------------------------------------------------
    # Public API: activation retrieval
    # ------------------------------------------------------------------

    def get(self) -> Tensor:
        """Return the most recently cached activation.

        Returns:
            Tensor of shape ``[batch, hidden_dim]``.

        Raises:
            RuntimeError: If no forward pass has been run yet.
        """
        if "_latest" not in self._cache:
            raise RuntimeError(
                "No activation cached yet. Run a forward pass through the model first."
            )
        return self._cache["_latest"]

    def clear(self) -> None:
        """Clear the cached activation (does not touch the attention mask)."""
        self._cache.clear()

    def remove(self) -> None:
        """Remove the hook from the model layer.

        Must be called to avoid memory leaks when the hook is no longer needed.
        """
        self._handle.remove()
        logger.debug("JanusActivationHook removed")
