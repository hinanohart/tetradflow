"""TetradFlow CLI entry point.

Sub-commands:
  train-sae        Train BatchTopK SAE on Janus-Pro activations (GPU required).
  identify-axes    Run CCA to identify 4 Tetrad axes from trained SAE.
  generate         Generate images with Tetrad-guided or CFG-baseline sampling.
  eval             Run eval/direct_metrics orthogonality checks.
  gate-check       Run P0-4 human gate check (wraps scripts/human_gate_check.py).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Annotated

import typer

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="tetradflow",
    help="TetradFlow: McLuhan Tetrad as inductive bias in Janus-Pro + Flux.",
    add_completion=False,
)


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
    )


# ---------------------------------------------------------------------------
# train-sae
# ---------------------------------------------------------------------------


@app.command("train-sae")
def train_sae(
    output: Annotated[Path, typer.Option("--output", "-o", help="Output .safetensors path")] = Path(
        "sae.safetensors"
    ),
    input_dim: Annotated[int, typer.Option(help="Janus hidden dim")] = 4096,
    n_features: Annotated[int, typer.Option(help="SAE feature count")] = 16_384,
    k: Annotated[int, typer.Option(help="BatchTopK sparsity k")] = 64,
    steps: Annotated[int, typer.Option(help="Training steps")] = 50_000,
    lr: Annotated[float, typer.Option(help="AdamW learning rate")] = 5e-5,
    batch_size: Annotated[int, typer.Option(help="Batch size")] = 2048,
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
) -> None:
    """Train the BatchTopK SAE on Janus-Pro layer-20 activations.

    Requires a CUDA GPU. Delegates to scripts/train_sae.py logic.
    """
    _configure_logging(verbose)
    if not _cuda_available():
        typer.echo("ERROR: GPU required for SAE training. Aborting.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Training SAE: n_features={n_features} k={k} steps={steps}")

    # M9 fix (2026-05-17): resolve scripts/ via TETRADFLOW_SCRIPTS_DIR env var
    # (consistent with gate-check below), not via `import scripts.*` which
    # silently breaks on PyPI installs where the scripts/ dir is not packaged.
    import os
    import subprocess

    scripts_dir_env = os.environ.get("TETRADFLOW_SCRIPTS_DIR")
    if scripts_dir_env:
        scripts_dir = Path(scripts_dir_env)
    else:
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"

    train_script = scripts_dir / "train_sae.py"
    if not train_script.exists():
        typer.echo(
            f"ERROR: train_sae.py not found at {train_script}\n"
            "Hint: set TETRADFLOW_SCRIPTS_DIR=/path/to/repo/scripts if installed from PyPI,\n"
            "or run `python scripts/train_sae.py` directly from a cloned repo.",
            err=True,
        )
        raise typer.Exit(1)

    cmd = [
        sys.executable,
        str(train_script),
        "--output",
        str(output),
        "--input-dim",
        str(input_dim),
        "--n-features",
        str(n_features),
        "--k",
        str(k),
        "--steps",
        str(steps),
        "--lr",
        str(lr),
        "--batch-size",
        str(batch_size),
    ]
    if verbose:
        cmd.append("--verbose")

    result = subprocess.run(cmd, capture_output=False)
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# identify-axes
# ---------------------------------------------------------------------------


@app.command("identify-axes")
def identify_axes(
    sae: Annotated[Path, typer.Option("--sae", help="Path to trained SAE .safetensors")],
    labels: Annotated[
        Path,
        typer.Option(
            "--labels",
            help=(
                "Path to labels file. Accepts .safetensors (keys: sae_features, labels) "
                "or .json (keys: sae_features list[list[float]], labels list[list[float]])."
            ),
        ),
    ],
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Output AxesMap .safetensors")
    ] = Path("axes_map.safetensors"),
    top_k: Annotated[int, typer.Option(help="Top-k features per axis")] = 32,
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
) -> None:
    """Identify 4 Tetrad axes via CCA on trained SAE features (P0-2).

    Loads the SAE and label file, runs CCA, prints pairwise cosine
    orthogonality check, and saves AxesMap.

    Labels file format:
      .safetensors — tensors "sae_features" [B, n_features] and "labels" [B, 4]
      .json        — {"sae_features": [[...], ...], "labels": [[...], ...]}
    """
    _configure_logging(verbose)

    if not sae.exists():
        typer.echo(f"ERROR: SAE file not found: {sae}", err=True)
        raise typer.Exit(1)
    if not labels.exists():
        typer.echo(f"ERROR: Labels file not found: {labels}", err=True)
        raise typer.Exit(1)

    import json

    import torch

    from tetradflow.cca import TETRAD_AXES, CCAAxisFinder

    typer.echo(f"Loading labels from {labels}")
    suffix = labels.suffix.lower()
    if suffix == ".safetensors":
        from safetensors.torch import load_file  # type: ignore[import-untyped]

        label_state = load_file(str(labels))
        sae_features_t: torch.Tensor = label_state["sae_features"]
        label_t: torch.Tensor = label_state["labels"]
    elif suffix == ".json":
        with labels.open() as fh:
            raw = json.load(fh)
        sae_features_t = torch.tensor(raw["sae_features"], dtype=torch.float32)
        label_t = torch.tensor(raw["labels"], dtype=torch.float32)
    else:
        typer.echo(f"ERROR: --labels must be .safetensors or .json, got {suffix!r}", err=True)
        raise typer.Exit(1)

    finder = CCAAxisFinder(top_k=top_k)
    typer.echo("Running CCA ...")
    axes_map = finder.fit(sae_features_t, label_t)

    cosines = CCAAxisFinder.pairwise_cosines(axes_map)
    typer.echo("\n=== Pairwise axis cosines (6 pairs) ===")
    pairs = [("E", "O"), ("E", "Re"), ("E", "Rv"), ("O", "Re"), ("O", "Rv"), ("Re", "Rv")]
    all_pass = True
    for (a, b), c in zip(pairs, cosines.tolist()):
        status = "PASS" if c < 0.3 else "FAIL"
        if status == "FAIL":
            all_pass = False
        typer.echo(f"  {a}-{b}: {c:.4f}  [{status}]")

    if not all_pass:
        typer.echo(
            "\nWARNING: P0-2 gate FAIL — cosine ≥ 0.3 detected. "
            "Human review required before proceeding (P0-4).",
            err=True,
        )

    finder.save_axes_map(axes_map, output)
    typer.echo(f"\nAxesMap saved to {output}")

    for i, ax in enumerate(TETRAD_AXES):
        top3 = axes_map.feature_indices[i, :3].tolist()
        typer.echo(f"  {ax}: top-3 features = {top3}")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@app.command("generate")
def generate(
    prompt: Annotated[str, typer.Argument(help="Text prompt")],
    axis: Annotated[
        str | None,
        typer.Option(
            "--axis",
            help="Tetrad axis: Enhance|Obsolesce|Retrieve|Reverse",
            show_default=False,
        ),
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="tetrad|cfg_baseline")] = "tetrad",
    sae: Annotated[Path | None, typer.Option("--sae", help="SAE .safetensors")] = None,
    axes_map: Annotated[
        Path | None, typer.Option("--axes-map", help="AxesMap .safetensors")
    ] = None,
    steps: Annotated[int, typer.Option(help="Inference steps")] = 4,
    guidance: Annotated[float, typer.Option(help="CFG guidance scale")] = 3.5,
    output: Annotated[Path, typer.Option("-o", help="Output image path")] = Path("output.png"),
    seed: Annotated[int | None, typer.Option(help="Random seed")] = None,
    vram: Annotated[
        str, typer.Option(help="bf16 only (fp8 = P1 milestone, NOT yet implemented)")
    ] = "bf16",
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
) -> None:
    """Generate an image with Tetrad-guided or CFG-baseline sampling."""
    _configure_logging(verbose)

    if mode not in ("tetrad", "cfg_baseline"):
        typer.echo(f"ERROR: --mode must be tetrad or cfg_baseline, got {mode!r}", err=True)
        raise typer.Exit(1)

    from tetradflow.pipeline import TetradFlowPipeline

    pipeline = TetradFlowPipeline(
        sae_path=sae,
        axes_map_path=axes_map,
        vram_mode=vram,  # type: ignore[arg-type]
    )
    typer.echo("Loading pipeline (may take a few minutes on first run) ...")
    try:
        pipeline.load()
    except RuntimeError as exc:
        typer.echo(f"ERROR: {exc}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Generating: mode={mode} axis={axis} steps={steps}")
    image = pipeline.generate(
        prompt=prompt,
        axis=axis,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        num_inference_steps=steps,
        guidance_scale=guidance,
        seed=seed,
    )
    image.save(str(output))
    typer.echo(f"Saved to {output}")
    pipeline.unload()


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


@app.command("eval")
def run_eval(
    axes_map: Annotated[Path, typer.Option("--axes-map", help="AxesMap .safetensors")],
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
) -> None:
    """Run P0-2 orthogonality eval (6 pairwise cosine checks)."""
    _configure_logging(verbose)

    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "src/tetradflow/eval/direct_metrics.py",
            "-v",
            f"--axes-map={axes_map}",
        ],
        capture_output=False,
    )
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# gate-check
# ---------------------------------------------------------------------------


@app.command("gate-check")
def gate_check(
    eval_json: Annotated[Path, typer.Option("--eval-json", help="CI eval JSON artifact")],
    verbose: Annotated[bool, typer.Option("--verbose/--no-verbose")] = False,
) -> None:
    """Run P0-4 human gate check on CI eval artifact.

    Delegates to scripts/human_gate_check.py.
    Exits 1 if gate fails (CI will block release).
    """
    _configure_logging(verbose)

    # Resolve scripts/ directory: support both editable-install (repo layout) and
    # --scripts-dir env var for PyPI-installed usage.
    import os
    import subprocess

    scripts_dir_env = os.environ.get("TETRADFLOW_SCRIPTS_DIR")
    if scripts_dir_env:
        scripts_dir = Path(scripts_dir_env)
    else:
        # Editable install: src/tetradflow/cli.py → repo root is 3 levels up
        scripts_dir = Path(__file__).parent.parent.parent / "scripts"

    gate_script = scripts_dir / "human_gate_check.py"

    if not gate_script.exists():
        typer.echo(
            f"ERROR: gate check script not found at {gate_script}\n"
            "Hint: set TETRADFLOW_SCRIPTS_DIR=/path/to/repo/scripts if installed from PyPI.",
            err=True,
        )
        raise typer.Exit(1)

    result = subprocess.run(
        [sys.executable, str(gate_script), "--eval-json", str(eval_json)],
        capture_output=False,
    )
    raise typer.Exit(result.returncode)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def main() -> None:
    """Entry point for ``tetradflow`` console script."""
    app()


if __name__ == "__main__":
    main()
