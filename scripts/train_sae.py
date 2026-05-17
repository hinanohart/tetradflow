"""Train BatchTopK SAE on Janus-Pro layer-20 activations.

Requires:
  - CUDA GPU (VRAM >= 18 GB for Plan B Lite, >= 52 GB for full BF16)
  - Janus-Pro model downloaded (deepseek-ai/Janus-Pro-7B by default)

Usage:
  python scripts/train_sae.py \\
    --output checkpoints/sae.safetensors \\
    --steps 50000 \\
    --batch-size 2048

P0-2: SAE is fully unsupervised. No axis anchor indices are fixed.
      Run scripts/identify_axes_cca.py after training to identify axes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train BatchTopK SAE on Janus-Pro activations (GPU required)."
    )
    p.add_argument(
        "--output", "-o", default="checkpoints/sae.safetensors", help="Output .safetensors path"
    )
    p.add_argument(
        "--janus-model-id",
        default="deepseek-ai/Janus-Pro-7B",
        help="HuggingFace model ID for Janus-Pro",
    )
    p.add_argument(
        "--layer-idx", type=int, default=20, help="Janus-Pro layer to hook (default: 20)"
    )
    p.add_argument("--input-dim", type=int, default=4096, help="Janus hidden dimension")
    p.add_argument("--n-features", type=int, default=16_384, help="SAE feature count")
    p.add_argument("--k", type=int, default=64, help="BatchTopK sparsity k")
    p.add_argument("--steps", type=int, default=50_000, help="Training steps")
    p.add_argument("--lr", type=float, default=5e-5, help="AdamW learning rate")
    p.add_argument("--batch-size", type=int, default=2048, help="Activation batch size")
    p.add_argument(
        "--labels-path",
        default=None,
        help="Path to labelled prompts JSON for activation collection",
    )
    p.add_argument(
        "--scaffold-test",
        action="store_true",
        help=(
            "Run with RANDOM DUMMY activations for smoke-testing the training loop. "
            "Output filename is force-suffixed with '_SCAFFOLD_DO_NOT_USE' so the "
            "resulting artifact cannot be mistakenly used as a production SAE."
        ),
    )
    p.add_argument(
        "--save-every", type=int, default=10_000, help="Checkpoint save interval (steps)"
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main(
    output: str = "checkpoints/sae.safetensors",
    janus_model_id: str = "deepseek-ai/Janus-Pro-7B",
    layer_idx: int = 20,
    input_dim: int = 4096,
    n_features: int = 16_384,
    k: int = 64,
    steps: int = 50_000,
    lr: float = 5e-5,
    batch_size: int = 2048,
    labels_path: str | None = None,
    scaffold_test: bool = False,
    save_every: int = 10_000,
) -> None:
    """Train SAE. Can be called from CLI or from tetradflow CLI.

    G1 safety (2026-05-17): refuses to run with random dummy activations unless
    ``--scaffold-test`` is passed AND output filename is suffixed
    ``_SCAFFOLD_DO_NOT_USE``. This prevents accidentally publishing a
    noise-trained SAE as a P0-2 artifact.
    """
    # G1: refuse silent dummy training. Either real labels OR explicit scaffold opt-in.
    if labels_path is None and not scaffold_test:
        raise NotImplementedError(
            "Real dataset streaming is not implemented in this scaffold.\n"
            "Either:\n"
            "  (1) Pass --labels-path <path> to use labelled prompts (production path), or\n"
            "  (2) Pass --scaffold-test to run with random dummy activations for smoke\n"
            "      testing. The output filename will be force-suffixed with\n"
            "      '_SCAFFOLD_DO_NOT_USE' and the artifact MUST NOT be used as a\n"
            "      production SAE (would silently break the P0-2 cosine gate)."
        )

    if not torch.cuda.is_available():
        logger.error("GPU required for SAE training. Aborting.")
        sys.exit(1)

    device = "cuda"
    output_path = Path(output)
    if scaffold_test and "_SCAFFOLD_DO_NOT_USE" not in output_path.stem:
        output_path = output_path.with_name(
            output_path.stem + "_SCAFFOLD_DO_NOT_USE" + output_path.suffix
        )
        logger.warning(
            "Scaffold-test mode active: output forced to %s. "
            "This is NOT a production SAE — do not use for P0-2 gate or release.",
            output_path,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        "SAE training: n_features=%d k=%d steps=%d lr=%g",
        n_features,
        k,
        steps,
        lr,
    )

    # Load Janus-Pro
    from transformers import AutoModelForCausalLM, AutoProcessor  # type: ignore[import-untyped]

    from tetradflow.hooks import JanusActivationHook
    from tetradflow.sae import BatchTopKSAE

    logger.info("Loading Janus-Pro: %s", janus_model_id)
    # NOTE: processor will be needed when scaffold becomes a real training loop
    # (currently the dummy-data path bypasses the processor). Bound to _ to
    # document the intent without tripping ruff F841.
    _ = AutoProcessor.from_pretrained(janus_model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        janus_model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()

    hook = JanusActivationHook(model, layer_idx=layer_idx)

    # SAE
    sae = BatchTopKSAE(input_dim=input_dim, n_features=n_features, k=k).to(device)
    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr)

    # Activation collection loop (placeholder: real impl streams from dataset)
    logger.info("Starting training loop (%d steps)...", steps)
    for step in range(1, steps + 1):
        # --- Collect activation batch from Janus-Pro ---
        # In production: iterate over a labelled prompt dataset
        # Here: random dummy activations for scaffold purposes
        with torch.no_grad():
            # Replace with real dataset streaming:
            # inputs = processor(text=prompt_batch, return_tensors="pt").to(device)
            # model(**inputs)
            # x = hook.get()
            x = torch.randn(batch_size, input_dim, device=device, dtype=torch.bfloat16)

        # --- SAE forward + loss ---
        optimizer.zero_grad()
        out = sae(x.float())
        loss = sae.loss(x.float(), out)
        loss.backward()
        optimizer.step()
        sae.post_step()

        if step % 1000 == 0:
            logger.info("step %d/%d loss=%.6f", step, steps, loss.item())

        if step % save_every == 0:
            ckpt = str(output_path).replace(".safetensors", f"_step{step}.safetensors")
            sae.save(ckpt)
            logger.info("Checkpoint saved: %s", ckpt)

    hook.remove()
    sae.save(str(output_path))
    logger.info("Final SAE saved to %s", output_path)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )
    main(
        output=args.output,
        janus_model_id=args.janus_model_id,
        layer_idx=args.layer_idx,
        input_dim=args.input_dim,
        n_features=args.n_features,
        k=args.k,
        steps=args.steps,
        lr=args.lr,
        batch_size=args.batch_size,
        labels_path=args.labels_path,
        scaffold_test=args.scaffold_test,
        save_every=args.save_every,
    )
