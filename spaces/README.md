---
title: TetradFlow Demo
emoji: 🌊
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.0"
app_file: app.py
python_version: "3.11"
hardware: zero-a10g
pinned: false
license: mit
---

# TetradFlow Demo

**Research prototype** — not a validated cognitive model.
See [disclaimer](https://github.com/tetradflow-dev/tetradflow/blob/main/docs/disclaimer.md).

Generate images guided by McLuhan's 4 Tetrad axes:
- **Enhance** — What does the medium amplify?
- **Obsolesce** — What does the medium push aside?
- **Retrieve** — What does the medium resurrect?
- **Reverse** — What does the medium become at saturation?

## Usage

1. Enter a text prompt.
2. Select a Tetrad axis (or use CFG baseline for comparison).
3. Click Generate to see 4 images (one per axis in Tetrad mode).

## Configuration

The Space reads one optional environment variable at first-call (lazy load):

| Variable | Purpose | Default |
|---|---|---|
| `HF_MODEL_REPO` | HuggingFace repo id (`<owner>/<name>`) hosting `sae.safetensors` and `axes_map.safetensors`. When set, both are downloaded via `hf_hub_download` and wired into the pipeline so the demo runs with the trained SAE + Tetrad axes map. When unset, the pipeline runs **without** SAE artifacts (CFG baseline only; `mode="tetrad"` will raise `NotImplementedError` per P0-4). | `""` (unset) |

Set this in **Space settings → Variables and secrets** (no secret needed; the repo
contents are public). The download is read-only — Spaces never needs an HF token
unless `HF_MODEL_REPO` is private (in which case add `HF_TOKEN` as a secret).

## Citation

```bibtex
@software{tetradflow2026,
  title  = {TetradFlow: McLuhan Tetrad as Inductive Bias in Janus-Pro + Flux},
  year   = {2026},
  url    = {https://github.com/tetradflow-dev/tetradflow}
}
```
