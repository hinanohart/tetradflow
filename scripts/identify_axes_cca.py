"""Identify 4 Tetrad axes via CCA on trained SAE features (P0-2).

Usage:
  python scripts/identify_axes_cca.py \\
    --sae checkpoints/sae.safetensors \\
    --labels eval/rater_a_labels.json \\
    --output checkpoints/axes_map.safetensors

Input labels JSON schema (same as eval/manual_labels_template.json):
  {"items": [{"id": str, "prompt": str, "axis": "Enhance|Obsolesce|Retrieve|Reverse"}]}

Output: AxesMap .safetensors consumed by TetradFlowPipeline and direct_metrics tests.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

AXIS_TO_IDX = {"Enhance": 0, "Obsolesce": 1, "Retrieve": 2, "Reverse": 3}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Identify Tetrad axes via CCA after unsupervised SAE training (P0-2)."
    )
    p.add_argument("--sae", required=True, help="Path to trained SAE .safetensors")
    p.add_argument(
        "--labels",
        required=True,
        help="Path to labelled prompts JSON (manual_labels_template.json format)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="checkpoints/axes_map.safetensors",
        help="Output AxesMap .safetensors path",
    )
    p.add_argument("--top-k", type=int, default=32, help="Top-k features to report per axis")
    p.add_argument("--janus-model-id", default="deepseek-ai/Janus-Pro-7B")
    p.add_argument("--layer-idx", type=int, default=20)
    p.add_argument(
        "--eval-json", default=None, help="If set, write cosine summary JSON for CI gate check"
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def collect_sae_features(
    sae_path: str,
    labels_path: str,
    janus_model_id: str,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect SAE features and one-hot labels for labelled prompts.

    Returns:
        sae_features: [N, n_features] float32
        labels: [N, 4] float32 one-hot
    """
    with open(labels_path) as f:
        data = json.load(f)

    items = [item for item in data["items"] if item.get("axis") in AXIS_TO_IDX]
    if len(items) == 0:
        logger.error("No labelled items found in %s", labels_path)
        sys.exit(1)

    logger.info("Found %d labelled items", len(items))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load SAE
    from tetradflow.sae import BatchTopKSAE

    logger.info("Loading SAE from %s", sae_path)
    # Load state to infer architecture
    from safetensors.torch import load_file  # type: ignore[import-untyped]

    state = load_file(sae_path, device="cpu")
    input_dim = state["pre_bias"].shape[0]
    n_features = state["b_enc"].shape[0]
    has_w_dec = "W_dec" in state

    sae = BatchTopKSAE(input_dim=input_dim, n_features=n_features, tied_weights=not has_w_dec)
    sae.load_state_dict(state)
    sae = sae.to(device).eval()

    # Load Janus-Pro and hook
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]

    from tetradflow.hooks import JanusActivationHook

    logger.info("Loading Janus-Pro: %s", janus_model_id)
    processor = AutoProcessor.from_pretrained(janus_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        janus_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    hook = JanusActivationHook(model, layer_idx=layer_idx)

    feature_list: list[torch.Tensor] = []
    label_list: list[torch.Tensor] = []

    for item in items:
        prompt = item["prompt"]
        axis_idx = AXIS_TO_IDX[item["axis"]]

        inputs = processor(text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)
            act = hook.get()  # [1, hidden_dim]

        with torch.no_grad():
            z_topk, _ = sae.encode(act.float())

        feature_list.append(z_topk.cpu())
        one_hot = torch.zeros(1, 4)
        one_hot[0, axis_idx] = 1.0
        label_list.append(one_hot)

    hook.remove()

    sae_features = torch.cat(feature_list, dim=0)  # [N, n_features]
    labels = torch.cat(label_list, dim=0)  # [N, 4]
    return sae_features, labels


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    sae_features, labels = collect_sae_features(
        sae_path=args.sae,
        labels_path=args.labels,
        janus_model_id=args.janus_model_id,
        layer_idx=args.layer_idx,
    )

    from tetradflow.cca import TETRAD_AXES, CCAAxisFinder

    finder = CCAAxisFinder(top_k=args.top_k)
    logger.info(
        "Running CCA (%d samples, %d features)...", sae_features.shape[0], sae_features.shape[1]
    )
    axes_map = finder.fit(sae_features, labels)

    # Print cosine summary
    cosines = CCAAxisFinder.pairwise_cosines(axes_map)
    pairs = [("E", "O"), ("E", "Re"), ("E", "Rv"), ("O", "Re"), ("O", "Rv"), ("Re", "Rv")]
    print("\n=== Pairwise axis cosines (P0-2 gate: all < 0.3) ===")
    all_pass = True
    for (a, b), c in zip(pairs, cosines.tolist()):
        status = "PASS" if c < 0.3 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  {a}-{b}: {c:.4f}  [{status}]")

    if not all_pass:
        print("\nWARNING: P0-2 gate FAIL. Human review required before release (P0-4).")
    else:
        print("\nP0-2 gate: PASS")

    for i, ax in enumerate(TETRAD_AXES):
        top5 = axes_map.feature_indices[i, :5].tolist()
        print(f"  {ax}: top-5 features = {top5}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    finder.save_axes_map(axes_map, output_path)
    logger.info("AxesMap saved to %s", output_path)

    if args.eval_json:
        from tetradflow.eval.direct_metrics import write_eval_json

        write_eval_json(axes_map, args.eval_json)
        logger.info("Eval JSON written to %s", args.eval_json)


if __name__ == "__main__":
    main()
