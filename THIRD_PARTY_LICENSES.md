# Third-Party Notices

TetradFlow itself is licensed under MIT (see [LICENSE](LICENSE)). The third-party
dependencies loaded at runtime carry the licenses listed below. This file is
maintained to satisfy Apache License 2.0 §4(d) attribution requirements when
TetradFlow is redistributed alongside (or against) these dependencies.

For a machine-generated, fully accurate snapshot in any specific environment,
run:

```bash
pip install pip-licenses
pip-licenses --format=markdown --with-urls --with-license-file \
  --output-file=THIRD_PARTY_LICENSES.full.md
```

The list below is the curated, human-readable summary intended for source-form
redistributions and citation.

## Apache License 2.0 dependencies (NOTICE inclusion required)

Per Apache 2.0 §4(d), redistributions of these dependencies must include
their original LICENSE and (when present) NOTICE files. The references below
identify the upstream sources; downstream redistributors must consult and
include those files verbatim.

| Project | Upstream | Copyright |
|---|---|---|
| transformers | https://github.com/huggingface/transformers | © 2018- HuggingFace, Inc. |
| diffusers | https://github.com/huggingface/diffusers | © 2022- HuggingFace, Inc. |
| huggingface-hub | https://github.com/huggingface/huggingface_hub | © 2020- HuggingFace, Inc. |
| safetensors | https://github.com/huggingface/safetensors | © 2022- HuggingFace, Inc. |
| spaces (`[spaces]` extra) | https://github.com/huggingface/spaces | © HuggingFace, Inc. |
| gradio (`[spaces]` extra) | https://github.com/gradio-app/gradio | © 2019- HuggingFace, Inc. |
| Flux.1-schnell (model weights) | https://github.com/black-forest-labs/flux | © 2024 Black Forest Labs |

## MIT-licensed dependencies (no NOTICE requirement, listed for transparency)

| Project | Upstream | Copyright |
|---|---|---|
| sae-lens (`[research]` extra) | https://github.com/jbloomAus/SAELens | © 2024- Joseph Bloom et al. |
| typer | https://github.com/tiangolo/typer | © 2019- Sebastián Ramírez |
| rich | https://github.com/Textualize/rich | © 2020- Will McGugan |
| einops | https://github.com/arogozhnikov/einops | © 2018- Alex Rogozhnikov |
| Janus-Pro (model weights) | https://github.com/deepseek-ai/Janus | © 2024 DeepSeek AI |

## BSD-3-Clause dependencies

| Project | Upstream | Copyright |
|---|---|---|
| torch | https://github.com/pytorch/pytorch | © Meta Platforms / The Linux Foundation |
| numpy | https://github.com/numpy/numpy | © NumPy Developers |
| scikit-learn | https://github.com/scikit-learn/scikit-learn | © The scikit-learn developers |

## Model weights — separate from package code

The Janus-Pro 7B and Flux.1-schnell weights are not packaged with `pip install
tetradflow`. They are fetched at runtime via HuggingFace Hub. Each model has
its own license file in its respective HF repo (Janus: MIT, Flux: Apache 2.0)
that downstream users accept upon download.

Trained SAE / CCA artifacts that TetradFlow produces are not derivative works
of Janus-Pro / Flux per common interpretation (they are independently
optimised parameters on internal activations), but downstream redistributors
should consult counsel for their jurisdiction.

## Reporting issues

If you spot a missing attribution or an incorrect license here, please open
an issue: https://github.com/tetradflow-dev/tetradflow/issues
