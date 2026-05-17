"""BatchTopK Sparse Autoencoder (SAE) for TetradFlow.

Implements the BatchTopK SAE from arXiv:2412.06410.
- 16 384 features, k=64, expansion factor 4x
- Unsupervised learning only (P0-2: no anchor index fixing)
- After training, use CCAAxisFinder (cca.py) to identify the 4 Tetrad axes post-hoc
"""

from __future__ import annotations

import logging
import math
from typing import NamedTuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


class SAEOutput(NamedTuple):
    """Output of a forward pass through BatchTopKSAE."""

    x_hat: Tensor  # Reconstructed input  [batch, input_dim]
    z_topk: Tensor  # Sparse latent (sparse, values at top-k pos) [batch, n_features]
    indices: Tensor  # Top-k feature indices per sample  [batch, k]


class BatchTopKSAE(nn.Module):
    """BatchTopK Sparse Autoencoder (arXiv:2412.06410).

    Architecture
    ------------
    - Encoder: linear (input_dim → n_features) + ReLU, then BatchTopK sparsification
    - Decoder: linear (n_features → input_dim) with unit-norm columns
    - Bias: pre-encoder bias (input centering), encoder bias, NO decoder bias

    P0-2 constraint
    ---------------
    This module is fully unsupervised. There is no ``axis_feature_idx`` argument
    or any fixed-axis regularization. After training, call
    ``CCAAxisFinder.fit(sae_features, tetrad_labels)`` in ``cca.py`` to identify
    which feature indices correspond to the 4 Tetrad axes.

    Args:
        input_dim: Dimensionality of the input activations (e.g. 4096 for Janus-Pro).
        n_features: Total number of SAE features. Default 16 384 (= 4 × input_dim
            when input_dim=4096).
        k: Number of active features per sample in BatchTopK. Default 64.
        tied_weights: If True, decoder weight = encoder weight transposed (optional).
    """

    def __init__(
        self,
        input_dim: int = 4096,
        n_features: int = 16_384,
        k: int = 64,
        tied_weights: bool = False,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.n_features = n_features
        self.k = k
        self.tied_weights = tied_weights

        # Pre-encoder bias (learnable input centering)
        self.pre_bias = nn.Parameter(torch.zeros(input_dim))

        # Encoder: W_enc [input_dim, n_features], b_enc [n_features]
        self.W_enc = nn.Parameter(torch.empty(input_dim, n_features))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: W_dec [n_features, input_dim] — columns are normalised after each step
        if not tied_weights:
            self.W_dec = nn.Parameter(torch.empty(n_features, input_dim))
        else:
            self.register_parameter("W_dec", None)

        self._init_weights()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        """Kaiming uniform for encoder; decoder columns initialised to unit norm."""
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.pre_bias)

        if not self.tied_weights:
            assert self.W_dec is not None
            nn.init.kaiming_uniform_(self.W_dec, a=math.sqrt(5))
            # Normalise each decoder column (feature direction) to unit norm
            self._normalise_decoder_()

    @torch.no_grad()
    def _normalise_decoder_(self) -> None:
        """Project decoder columns onto the unit sphere (in-place)."""
        if self.tied_weights:
            return
        assert self.W_dec is not None
        # W_dec: [n_features, input_dim] — norm over input_dim axis
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(norms)

    # ------------------------------------------------------------------
    # Encode / decode helpers
    # ------------------------------------------------------------------

    def _effective_W_dec(self) -> Tensor:
        """Return W_dec, using W_enc.T when tied."""
        if self.tied_weights:
            return self.W_enc.T  # [n_features, input_dim]
        assert self.W_dec is not None
        return self.W_dec

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to sparse latent representation using true BatchTopK.

        Implements the BatchTopK algorithm from arXiv:2412.06410:
        activations are pooled globally across all (batch, feature) positions,
        the top k*B values are selected, and the sparsity mask is redistributed
        per sample. This differs from per-sample TopK in that a single sample
        can contribute more than k active features if its activations are
        disproportionately large, preserving the batch-level sparsity budget.

        Args:
            x: Input activations [batch, input_dim].

        Returns:
            z_topk: Sparse latent [batch, n_features] (zeros outside batch-top-k).
            indices: Top-k indices per sample [batch, k] (from per-sample view of mask).
        """
        x_centered = x - self.pre_bias  # [batch, input_dim]
        z_pre = torch.relu(x_centered @ self.W_enc + self.b_enc)  # [batch, n_features]

        B, n_features = z_pre.shape

        # True BatchTopK (arXiv:2412.06410):
        # 1. Flatten all activations to [B * n_features]
        # 2. Select global top-(k * B) values
        # 3. Build binary mask, reshape to [B, n_features]
        flat = z_pre.reshape(-1)  # [B * n_features]
        global_k = self.k * B
        # clamp in case B*k > total elements (edge case for tiny batches)
        global_k = min(global_k, flat.shape[0])
        topk_flat = flat.topk(global_k, sorted=False)
        mask_flat = torch.zeros_like(flat)
        mask_flat.scatter_(0, topk_flat.indices, 1.0)
        mask = mask_flat.reshape(B, n_features)  # [B, n_features]

        z_topk = z_pre * mask  # [B, n_features], sparse

        # Derive per-sample top-k indices (for API compatibility with AxesMap consumers)
        # Use the mask to find active positions per row; pad with 0 if fewer than k active.
        _, indices = z_topk.topk(self.k, dim=1)

        return z_topk, indices

    def decode(self, z: Tensor) -> Tensor:
        """Decode sparse latent to reconstructed input.

        Args:
            z: Latent [batch, n_features].

        Returns:
            x_hat: Reconstruction [batch, input_dim].
        """
        W_dec = self._effective_W_dec()  # [n_features, input_dim]
        return z @ W_dec + self.pre_bias  # [batch, input_dim]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> SAEOutput:
        """Full forward pass: encode then decode.

        Args:
            x: Input activations [batch, input_dim].

        Returns:
            SAEOutput namedtuple with (x_hat, z_topk, indices).
        """
        z_topk, indices = self.encode(x)
        x_hat = self.decode(z_topk)
        return SAEOutput(x_hat=x_hat, z_topk=z_topk, indices=indices)

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def loss(self, x: Tensor, output: SAEOutput) -> Tensor:
        """MSE reconstruction loss (no auxiliary losses by default).

        Args:
            x: Original input [batch, input_dim].
            output: Result from forward().

        Returns:
            Scalar loss tensor.
        """
        return nn.functional.mse_loss(output.x_hat, x)

    # ------------------------------------------------------------------
    # Training utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def post_step(self) -> None:
        """Call after each optimizer step to re-normalise decoder columns."""
        self._normalise_decoder_()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model state dict using safetensors format (pickle-free).

        Requires the ``safetensors`` package (``pip install safetensors``).

        Args:
            path: Destination file path (should end in ``.safetensors``).
        """
        from safetensors.torch import save_file  # type: ignore[import-untyped]

        # safetensors requires plain str keys and contiguous tensors
        state = {k: v.contiguous() for k, v in self.state_dict().items()}
        save_file(state, path)
        logger.info("SAE saved to %s", path)

    @classmethod
    def load(cls, path: str, **kwargs: object) -> BatchTopKSAE:
        """Load a previously saved SAE from safetensors format.

        Args:
            path: Path to the ``.safetensors`` file.
            **kwargs: Passed to ``__init__`` (must match saved architecture).

        Returns:
            Loaded BatchTopKSAE instance.
        """
        from safetensors.torch import load_file  # type: ignore[import-untyped]

        model = cls(**kwargs)  # type: ignore[arg-type]
        state = load_file(path, device="cpu")
        model.load_state_dict(state)
        logger.info("SAE loaded from %s", path)
        return model
