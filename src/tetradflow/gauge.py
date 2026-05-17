"""Figure/Ground gauge flip module (A2, F4 mitigation).

JanusGaugeFlip implements the SigLIP/VQ blending gate described in the
TetradFlow architecture:

    alpha = sigmoid(linear(sae_4axis))   [automatic mode]
    output = alpha * siglip_path + (1 - alpha) * vq_path

F4 (circular reasoning) mitigation: a human-verifiable trigger token can
override alpha in 'manual' mode, bypassing the SAE sigmoid entirely.
This ensures gauge behaviour can be audited without relying on the SAE
to explain itself.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Sentinel value used when alpha is forced via manual mode
_MANUAL_ALPHA_SENTINEL: float = -1.0


class JanusGaugeFlip(nn.Module):
    """Figure/Ground gauge flip for Janus-Pro encoder paths.

    Blends the SigLIP visual path (figure) with the VQ discrete path
    (ground) using a learned alpha gating parameter derived from the
    SAE 4-axis representation.

    F4 mitigation
    -------------
    In 'manual' mode the caller supplies ``alpha`` directly (e.g. from a
    human-verifiable trigger token), completely bypassing the SAE-derived
    sigmoid. This breaks the circular dependency where the gauge relies on
    the SAE to interpret itself.

    Args:
        sae_n_axes: Number of SAE axis inputs (should be 4 for Tetrad).
        hidden_dim: Hidden dimension of the alpha projection MLP.
        dropout: Dropout rate for alpha MLP during training.
    """

    def __init__(
        self,
        sae_n_axes: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.sae_n_axes = sae_n_axes

        # Small MLP: sae_axes -> hidden -> scalar alpha (before sigmoid)
        self.alpha_proj = nn.Sequential(
            nn.Linear(sae_n_axes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights so alpha starts near 0.5 (balanced blend)."""
        for module in self.alpha_proj.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        siglip_features: Tensor,
        vq_features: Tensor,
        sae_axis_activations: Tensor | None = None,
        mode: Literal["auto", "manual"] = "auto",
        alpha: float | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Blend SigLIP and VQ encoder paths.

        Args:
            siglip_features: SigLIP path output [B, D] or [B, T, D].
            vq_features: VQ discrete path output, same shape as siglip_features.
            sae_axis_activations: SAE 4-axis activations [B, 4]. Required when
                ``mode='auto'``. Ignored when ``mode='manual'``.
            mode: ``'auto'`` uses SAE sigmoid; ``'manual'`` uses supplied alpha.
            alpha: Scalar in [0, 1]. Required when ``mode='manual'``.
                Ignored when ``mode='auto'``.

        Returns:
            Tuple of (blended_features, alpha_tensor):
                - blended_features: alpha * siglip + (1 - alpha) * vq, same shape.
                - alpha_tensor: The alpha value used [B, 1] or scalar.

        Raises:
            ValueError: On invalid mode or missing required arguments.
        """
        if siglip_features.shape != vq_features.shape:
            raise ValueError(
                f"siglip_features and vq_features must have the same shape, "
                f"got {siglip_features.shape} vs {vq_features.shape}"
            )

        if mode == "manual":
            if alpha is None:
                raise ValueError("alpha must be provided when mode='manual'.")
            if not (0.0 <= alpha <= 1.0):
                raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
            alpha_t = torch.tensor(
                alpha,
                dtype=siglip_features.dtype,
                device=siglip_features.device,
            )
            logger.debug("GaugeFlip: manual mode, alpha=%.4f", alpha)

        elif mode == "auto":
            if sae_axis_activations is None:
                raise ValueError("sae_axis_activations must be provided when mode='auto'.")
            if sae_axis_activations.shape[-1] != self.sae_n_axes:
                raise ValueError(
                    f"sae_axis_activations last dim must be {self.sae_n_axes}, "
                    f"got {sae_axis_activations.shape[-1]}"
                )
            # [B, 1] sigmoid alpha
            alpha_logit = self.alpha_proj(sae_axis_activations.to(siglip_features.dtype))  # [B, 1]
            alpha_t = torch.sigmoid(alpha_logit)  # [B, 1]
            logger.debug(
                "GaugeFlip: auto mode, alpha mean=%.4f",
                alpha_t.mean().item(),
            )

        else:
            raise ValueError(f"mode must be 'auto' or 'manual', got {mode!r}.")

        # Blend: alpha * siglip + (1 - alpha) * vq
        # alpha_t broadcasts over feature dims
        if alpha_t.dim() > 0 and siglip_features.dim() > alpha_t.dim():
            # Add extra dims for broadcasting over [B, T, D] etc.
            for _ in range(siglip_features.dim() - alpha_t.dim()):
                alpha_t = alpha_t.unsqueeze(-1)

        blended = alpha_t * siglip_features + (1.0 - alpha_t) * vq_features

        return blended, alpha_t

    # ------------------------------------------------------------------
    # Convenience: axis-level alpha inspection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def axis_alpha(self, sae_axis_activations: Tensor) -> Tensor:
        """Return alpha values for a batch without blending.

        Useful for monitoring gauge behaviour during evaluation.

        Args:
            sae_axis_activations: [B, 4] SAE axis activations.

        Returns:
            Alpha values [B, 1].
        """
        alpha_logit = self.alpha_proj(sae_axis_activations.float())
        return torch.sigmoid(alpha_logit)
