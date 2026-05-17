"""Push TetradFlow model artifacts to HuggingFace Hub.

IMPORTANT (R11): HF_TOKEN must be set as an environment variable.
Never hard-code tokens in this script.

Usage:
  HF_TOKEN=hf_... python scripts/push_to_hf.py \\
    --repo-id your-username/tetradflow-sae \\
    --sae checkpoints/sae.safetensors \\
    --axes-map checkpoints/axes_map.safetensors

WARNING (R13): This script must NOT be executed automatically in CI.
PyPI publish uses Trusted Publishing (OIDC). HF push is manual user step.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Push TetradFlow SAE artifacts to HuggingFace Hub (manual step)."
    )
    p.add_argument("--repo-id", required=True, help="HF repo id, e.g. 'username/tetradflow-sae'")
    p.add_argument("--sae", default=None, help="Path to SAE .safetensors to upload")
    p.add_argument("--axes-map", default=None, help="Path to AxesMap .safetensors to upload")
    p.add_argument(
        "--repo-type",
        default="model",
        choices=["model", "dataset", "space"],
        help="HuggingFace repo type",
    )
    p.add_argument(
        "--commit-message",
        default="Upload TetradFlow SAE artifacts",
        help="Commit message for HF Hub",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    # R11: token from environment only, never hardcoded
    token = os.environ.get("HF_TOKEN")
    if not token:
        logger.error(
            "HF_TOKEN environment variable not set. "
            "Set it before running: HF_TOKEN=hf_... python scripts/push_to_hf.py"
        )
        sys.exit(1)

    try:
        from huggingface_hub import HfApi  # type: ignore[import-untyped]
    except ImportError:
        logger.error("huggingface_hub not installed. Run: pip install huggingface-hub")
        sys.exit(1)

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            exist_ok=True,
        )
        logger.info("Repo ready: %s (type=%s)", args.repo_id, args.repo_type)
    except Exception as exc:
        logger.error("Failed to create/access repo %s: %s", args.repo_id, exc)
        sys.exit(1)

    files_to_upload: list[tuple[str, str]] = []

    if args.sae:
        sae_path = Path(args.sae)
        if not sae_path.exists():
            logger.error("SAE file not found: %s", sae_path)
            sys.exit(1)
        files_to_upload.append((str(sae_path), sae_path.name))

    if args.axes_map:
        axes_path = Path(args.axes_map)
        if not axes_path.exists():
            logger.error("AxesMap file not found: %s", axes_path)
            sys.exit(1)
        files_to_upload.append((str(axes_path), axes_path.name))

    if not files_to_upload:
        logger.warning("No files specified to upload. Use --sae and/or --axes-map.")
        sys.exit(0)

    for local_path, path_in_repo in files_to_upload:
        logger.info("Uploading %s -> %s/%s", local_path, args.repo_id, path_in_repo)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=path_in_repo,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=args.commit_message,
        )
        logger.info("Uploaded: %s", path_in_repo)

    logger.info("Upload complete. View at: https://huggingface.co/%s", args.repo_id)


if __name__ == "__main__":
    main()
