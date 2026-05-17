# TetradFlow Architecture

**Research prototype** — not a validated cognitive model. See [disclaimer.md](disclaimer.md).

## Overview

TetradFlow embeds McLuhan's Tetrad (Enhance / Obsolesce / Retrieve / Reverse) as an inductive
bias in a multi-model generation pipeline. The SAE feature space provides the alignment layer;
the 4-divergent ODE injects Tetrad direction into Flux velocity field during sampling.

## ASCII Architecture Diagram

```
                        ┌──────────────────────────────────────────┐
                        │         TetradFlowPipeline               │
                        │                                          │
  text prompt ──────────┤                                          │
                        │  ┌──────────────┐   hook (layer 20)     │
                        │  │  Janus-Pro   │──────────────┐         │
                        │  │  (7B / 1B)   │              │         │
                        │  └──────────────┘              ▼         │
                        │       ▲              ┌─────────────────┐ │
                        │       │ gauge flip   │  BatchTopK SAE  │ │
                        │  ┌──────────────┐   │  (16K features) │ │
                        │  │JanusGauge    │   └────────┬────────┘ │
                        │  │Flip (α gate) │            │          │
                        │  │SigLIP + VQ   │   CCA post-hoc       │
                        │  └──────────────┘            │          │
                        │                    ┌──────────▼───────┐ │
                        │                    │  CCAAxisFinder   │ │
                        │                    │  4 Tetrad axes   │ │
                        │                    │  (top-k indices) │ │
                        │                    └──────────┬───────┘ │
                        │                               │          │
                        │                    ┌──────────▼───────┐ │
                        │                    │  svd_top4_basis  │ │
                        │                    │  [4, D] ortho    │ │
                        │                    └──────────┬───────┘ │
                        │                               │          │
                        │  ┌──────────────┐             │          │
                        │  │  Flux.1      │◄────────────┘          │
                        │  │  (schnell)   │  tetrad_step ODE       │
                        │  │  DiT + CFG   │  v_k = v_θ + γ·⟨δ,eₖ⟩·eₖ │
                        │  └──────┬───────┘                        │
                        │         │                                │
                        │  ┌──────▼───────┐                        │
                        │  │  Show-o      │  (optional text-image  │
                        │  │  (NUS)       │   bidirectional ground)│
                        │  └──────┬───────┘                        │
                        │         │                                │
                        └─────────┼────────────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │   4 generated images       │
                    │   (one per Tetrad axis)    │
                    └────────────────────────────┘
```

## Component Descriptions

### 1. Janus-Pro (backbone)
- Model: `deepseek-ai/Janus-Pro-7B` (MIT license)
- Role: Unified image understanding + generation via SigLIP (vision) + VQ (discrete)
- Hook point: Layer 20 residual stream, last text token (see `hooks.py`)

### 2. BatchTopK SAE (`sae.py`)
- Architecture: BatchTopK SAE (arXiv:2412.06410), 16 384 features, k=64
- Training: Fully unsupervised (P0-2: no anchor index fixing)
- Input: Janus-Pro layer-20 activations [batch, 4096]
- Output: Sparse latent z_topk [batch, 16384]

### 3. CCAAxisFinder (`cca.py`, P0-2)
- After unsupervised SAE training, CCA maps 4 Tetrad axes to SAE feature indices
- Input: z_topk activations + manual one-hot labels [B, 4]
- Output: AxesMap (feature_indices [4, top_k], canonical_loadings [n_features, 4])
- Gate: 6 pairwise cosines < 0.3 (tested in `eval/direct_metrics.py`)

### 4. SVD Top-4 Basis (`ode.py`, P0-3)
- Replaces Gram-Schmidt (F3 fix)
- Input: 4 Tetrad direction vectors [4, D]
- Output: Orthonormal basis [4, D] via `torch.linalg.svd`

### 5. 4-Divergent ODE Step (`ode.py`)
- Injects Tetrad axis projections into Flux velocity field:
  `v_k = v_θ(x,t,c_text) + γ · ⟨δ, eₖ⟩ · eₖ`  for k ∈ {0,1,2,3}
- γ = 2.0 (default, configurable)
- Evaluated against CFG multi-direction baseline via ASS (P1-3)

### 6. JanusGaugeFlip (`gauge.py`, F4 mitigation)
- Blends SigLIP path (figure) with VQ path (ground) via learned α gate
- α = sigmoid(MLP(sae_4axis)) in auto mode
- Human-verifiable trigger token overrides α in manual mode (F4 fix)

### 7. TetradFlowPipeline (`pipeline.py`)
- Orchestrates all components
- VRAM: BF16 (~52 GB full, A100/H100 recommended); FP8 = P1 milestone (torchao, not yet implemented)
- P0-4: no auto-degradation on component failure — raises RuntimeError

## Data Flow

```
prompt → Janus-Pro (layer 20 hook) → SAE encode → CCA axes
                                                       │
                                              SVD top-4 basis [4, D]
                                                       │
Flux DiT velocity field ←──── tetrad_step ODE injection
        │
  4 latent samples (one per axis)
        │
  4 decoded images → HF Spaces gallery
```

## VRAM Budget

| Mode | Models loaded | VRAM |
|---|---|---|
| BF16 (default) | Janus-Pro 7B + Flux + SAE | ~52 GB (A100/H100 recommended) |
| SAE-only (Plan C) | SAE library standalone | < 2 GB |

> **Note**: FP8 quantization via torchao is a P1 milestone (NOT yet implemented).
> Running the full BF16 stack requires ~52 GB VRAM (Janus-Pro 7B + Flux 12B in BF16).

## License

All components use MIT or Apache 2.0 licensed base models. See [README.md](../README.md).
