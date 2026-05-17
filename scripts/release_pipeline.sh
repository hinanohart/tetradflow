#!/usr/bin/env bash
# scripts/release_pipeline.sh — TetradFlow release orchestration.
#
# R11/R13 BOUNDARY: this script is intended for USER MANUAL EXECUTION.
# It needs HF_TOKEN and a logged-in `gh` CLI. Claude (the assistant) must
# not run --execute on the user's behalf; --dry-run is safe to call from any
# context because it makes no external writes.
#
# Usage:
#   bash scripts/release_pipeline.sh --dry-run        # safe, prints commands
#   bash scripts/release_pipeline.sh --execute        # writes for real, with
#                                                     # interactive confirmation
#                                                     # gates on irreversible steps
#
# Required environment in --execute mode:
#   HF_TOKEN          HuggingFace API token (export, never committed)
#   TF_REPO_OWNER     e.g. tetradflow-dev
#   TF_REPO_NAME      e.g. tetradflow
#   TF_HF_NAMESPACE   e.g. tetradflow-dev
#   TF_VERSION        e.g. v0.1.0
#   EVAL_JSON_SHA     sha256sum of checkpoints/eval_output.json
#                     (the same SHA must be in the git tag annotation;
#                      release.yml verifies this in the gate-check job)
#
# What this script does NOT do (one-time manual setup, on pypi.org / GitHub UI):
#   - PyPI Trusted Publishing registration
#     https://pypi.org/manage/account/publishing/
#     repo: ${TF_REPO_OWNER}/${TF_REPO_NAME}, workflow: release.yml,
#     environment: pypi-release
#   - GitHub Environments > pypi-release > required reviewers
#     (controls who can approve PyPI publish jobs)
#
# What this script delegates to release.yml (after S7):
#   - `pip install build && python -m build`
#   - `pypa/gh-action-pypi-publish` (OIDC, no token)
#   - `python scripts/push_to_hf.py` (model artifacts)
#   - Spaces deploy (spaces-deploy.yml)

set -euo pipefail

MODE="${1:-}"
case "$MODE" in
    --dry-run|--execute) ;;
    *)
        echo "Usage: $0 --dry-run | --execute"
        echo ""
        echo "  --dry-run   Print commands without running. Safe for any context."
        echo "  --execute   Run for real. Requires env vars (see script header)."
        exit 1
        ;;
esac

# --------------------------------------------------------------------------
# Required env in --execute mode (fail fast; never read .env to avoid R11
# leakage into shell history).
# --------------------------------------------------------------------------
if [[ "$MODE" == "--execute" ]]; then
    : "${HF_TOKEN:?HF_TOKEN required (export HF_TOKEN=hf_xxx). Do NOT commit.}"
    : "${TF_REPO_OWNER:?TF_REPO_OWNER required (e.g. tetradflow-dev)}"
    : "${TF_REPO_NAME:?TF_REPO_NAME required (e.g. tetradflow)}"
    : "${TF_HF_NAMESPACE:?TF_HF_NAMESPACE required (e.g. tetradflow-dev)}"
    : "${TF_VERSION:?TF_VERSION required (e.g. v0.1.0)}"
    : "${EVAL_JSON_SHA:?EVAL_JSON_SHA required (sha256sum checkpoints/eval_output.json)}"
fi

# --dry-run defaults so the script can be inspected end-to-end without env.
TF_REPO_OWNER="${TF_REPO_OWNER:-DRY-RUN-OWNER}"
TF_REPO_NAME="${TF_REPO_NAME:-DRY-RUN-REPO}"
TF_HF_NAMESPACE="${TF_HF_NAMESPACE:-DRY-RUN-NS}"
TF_VERSION="${TF_VERSION:-v0.0.0-dry}"
EVAL_JSON_SHA="${EVAL_JSON_SHA:-DRY_RUN_SHA}"

# Helper: in dry-run print only; in execute run and print.
run() {
    if [[ "$MODE" == "--dry-run" ]]; then
        echo "+ $*"
    else
        echo "+ $*" >&2
        "$@"
    fi
}

# Helper: irreversible step → require interactive y/N in --execute.
confirm() {
    if [[ "$MODE" == "--dry-run" ]]; then
        echo "  [dry-run, skipping confirmation]"
        return 0
    fi
    read -r -p "  Proceed? [y/N]: " ans
    if [[ "$ans" != "y" && "$ans" != "Y" ]]; then
        echo "Aborted by user."
        exit 1
    fi
}

# ==========================================================================
# S1 — gh auth status check
# ==========================================================================
echo "=== S1: gh auth status ==="
if [[ "$MODE" == "--execute" ]]; then
    if ! gh auth status >/dev/null 2>&1; then
        echo "ERROR: 'gh auth login' first."
        exit 1
    fi
fi
echo "OK"

# ==========================================================================
# S2 — GitHub repo create (idempotent; NO push yet — push happens at S6.5
# after secrets/vars/env/protection are configured so the first CI run has
# everything it needs).
# ==========================================================================
echo ""
echo "=== S2: Create GitHub repo ${TF_REPO_OWNER}/${TF_REPO_NAME} (no push) ==="
if [[ "$MODE" == "--execute" ]]; then
    if gh repo view "${TF_REPO_OWNER}/${TF_REPO_NAME}" >/dev/null 2>&1; then
        echo "  already exists, skipping create"
    else
        echo "  This will create a public repo (no initial push)."
        confirm
        run gh repo create "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
            --public --source=. \
            --description "McLuhan Tetrad as inductive bias in Janus-Pro+Flux"
    fi
else
    echo "+ gh repo view ${TF_REPO_OWNER}/${TF_REPO_NAME} || gh repo create ... --public --source=. (no push)"
fi

# ==========================================================================
# S2.5 — Repository settings hardening (M2: wiki/merge/auto-delete)
# Sets sane OSS defaults: squash merge only, auto-delete head branches,
# disable wiki spam target, allow auto-merge for dependabot batches.
# ==========================================================================
echo ""
echo "=== S2.5: Repository settings PATCH ==="
run gh api -X PATCH "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}" \
    -F "has_wiki=false" \
    -F "has_projects=false" \
    -F "allow_merge_commit=false" \
    -F "allow_rebase_merge=false" \
    -F "allow_squash_merge=true" \
    -F "delete_branch_on_merge=true" \
    -F "allow_auto_merge=true"

# ==========================================================================
# S2.6 — Vulnerability alerts + Dependabot security fixes (M3)
# These are 204 No Content endpoints — they cannot be "checked"; calling them
# twice is harmless.
# ==========================================================================
echo ""
echo "=== S2.6: Enable vulnerability alerts + automated security fixes ==="
run gh api -X PUT "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/vulnerability-alerts"
run gh api -X PUT "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/automated-security-fixes"

# ==========================================================================
# S2.7 — Default workflow permissions = read-only (M1, defence in depth)
# ==========================================================================
echo ""
echo "=== S2.7: Default workflow permissions read-only ==="
run gh api -X PUT "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/actions/permissions/workflow" \
    -F "default_workflow_permissions=read" \
    -F "can_approve_pull_request_reviews=false"

# ==========================================================================
# S3 — Set HF_TOKEN GitHub secret (token piped via stdin; never echoed)
# ==========================================================================
echo ""
echo "=== S3: Set HF_TOKEN repo secret ==="
if [[ "$MODE" == "--execute" ]]; then
    confirm
    # IMPORTANT: --body - reads from stdin so the token never lands in argv
    # (which would leak via ps / shell history).
    printf '%s' "$HF_TOKEN" | gh secret set HF_TOKEN \
        --repo "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
        --body -
fi
echo "OK (token never echoed)"

# ==========================================================================
# S4 — Set HF_REPO_ID + HF_SPACES_REPO_ID repo variables (non-secret)
# ==========================================================================
echo ""
echo "=== S4: Set HF_REPO_ID + HF_SPACES_REPO_ID variables ==="
run gh variable set HF_REPO_ID \
    --repo "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
    --body "${TF_HF_NAMESPACE}/janus-pro-sae"
run gh variable set HF_SPACES_REPO_ID \
    --repo "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
    --body "${TF_HF_NAMESPACE}/tetradflow"

# ==========================================================================
# S4.5 — GitHub Environment "pypi-release" (C5 fix 2026-05-18:
# environment MUST be created even without reviewer, otherwise release.yml's
# `environment: pypi-release` pins the publish job into permanent pending).
# Reviewers added on top if TF_REVIEWER_USER_ID is exported.
# Get a numeric id with: `gh api /user | jq .id`
# ==========================================================================
echo ""
echo "=== S4.5: Force-create pypi-release environment (+ optional reviewers) ==="
run gh api -X PUT \
    "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/environments/pypi-release"
if [[ "$MODE" == "--execute" ]] && [[ -n "${TF_REVIEWER_USER_ID:-}" ]]; then
    run gh api -X PUT \
        "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/environments/pypi-release" \
        -F "reviewers[][type]=User" \
        -F "reviewers[][id]=${TF_REVIEWER_USER_ID}"
else
    echo "  TF_REVIEWER_USER_ID unset — environment created without reviewer gate."
    echo "  To add: export TF_REVIEWER_USER_ID=\$(gh api /user | jq .id) && rerun."
fi

# ==========================================================================
# S5 — HF model repo (exist_ok=True idempotent)
# ==========================================================================
echo ""
echo "=== S5: Create HF model repo ${TF_HF_NAMESPACE}/janus-pro-sae ==="
if [[ "$MODE" == "--execute" ]]; then
    confirm
    python3 - <<PY
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).create_repo(
    repo_id="${TF_HF_NAMESPACE}/janus-pro-sae",
    repo_type="model",
    exist_ok=True,
    private=False,
)
print("OK")
PY
else
    echo "+ HfApi.create_repo(${TF_HF_NAMESPACE}/janus-pro-sae, model, exist_ok=True)"
fi

# ==========================================================================
# S6 — HF Spaces (zero-a10g)
# ==========================================================================
echo ""
echo "=== S6: Create HF Spaces ${TF_HF_NAMESPACE}/tetradflow (gradio) ==="
if [[ "$MODE" == "--execute" ]]; then
    confirm
    python3 - <<PY
import os
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).create_repo(
    repo_id="${TF_HF_NAMESPACE}/tetradflow",
    repo_type="space",
    space_sdk="gradio",
    exist_ok=True,
    private=False,
)
print("OK (set hardware=zero-a10g on the Spaces settings page)")
PY
else
    echo "+ HfApi.create_repo(${TF_HF_NAMESPACE}/tetradflow, space, gradio, exist_ok=True)"
fi

# ==========================================================================
# S6.5 — First main push (now that secrets/vars/env are configured)
# ==========================================================================
echo ""
echo "=== S6.5: Push initial main branch ==="
if [[ "$MODE" == "--execute" ]]; then
    if git ls-remote --exit-code origin main >/dev/null 2>&1; then
        echo "  origin/main already exists, skipping initial push"
    else
        confirm
        run git push -u origin main
    fi
else
    echo "+ git ls-remote origin main || git push -u origin main"
fi

# ==========================================================================
# S6.6 — Branch protection on main (C2 fix 2026-05-18)
# Must run AFTER initial push so the status-check contexts have been
# registered by CI. Brief wait + retry handles the registration race.
# ==========================================================================
echo ""
echo "=== S6.6: Branch protection on main ==="
if [[ "$MODE" == "--execute" ]]; then
    confirm
    echo "  Waiting 30s for CI to register status-check contexts..."
    sleep 30
    run gh api -X PUT \
        "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/branches/main/protection" \
        -F "required_status_checks[strict]=true" \
        -F "required_status_checks[contexts][]=Test (Python 3.11)" \
        -F "enforce_admins=false" \
        -F "required_pull_request_reviews[required_approving_review_count]=1" \
        -F "required_pull_request_reviews[dismiss_stale_reviews]=true" \
        -F "restrictions=" \
        -F "allow_force_pushes=false" \
        -F "allow_deletions=false"
    # Note: only `Test (Python 3.11)` is required. 3.10/3.12 are advisory
    # (matrix visibility) and `Security scan` is non-blocking by design
    # (pip-audit advisory at ci.yml). Add more contexts later if desired.
else
    echo "+ sleep 30 && gh api -X PUT branches/main/protection (require 'Test (Python 3.11)')"
fi

# ==========================================================================
# S7 — Tag and push (HUMAN GATE — triggers release.yml)
# ==========================================================================
echo ""
echo "=== S7: Tag ${TF_VERSION} with eval_sha256=${EVAL_JSON_SHA} and push ==="
echo "    THIS TRIGGERS release.yml: PyPI publish + HF Hub + Spaces deploy."
echo "    The release.yml gate-check job will verify the tag annotation"
echo "    contains eval_sha256=${EVAL_JSON_SHA} — change it and the release fails."
if [[ "$MODE" == "--execute" ]]; then
    confirm
    run git tag -a "${TF_VERSION}" -m "eval_sha256=${EVAL_JSON_SHA}"
    echo ""
    echo "  Tag created locally. Push to origin? (this is the point of no return)"
    confirm
    run git push origin "${TF_VERSION}"
else
    echo "+ git tag -a ${TF_VERSION} -m \"eval_sha256=${EVAL_JSON_SHA}\""
    echo "+ git push origin ${TF_VERSION}"
fi

# ==========================================================================
# S8 — Post-publish verification (N3, 2026-05-18)
# Surfaces failing CI runs, open PRs with ✗, branch protection state, and
# repo settings drift so user can spot trouble within ~60s of going live.
# ==========================================================================
echo ""
echo "=== S8: Post-publish verification ==="
if [[ "$MODE" == "--execute" ]]; then
    echo "Waiting 60s for CI workflow runs to register..."
    sleep 60
    echo ""
    echo "--- recent workflow runs (last 5) ---"
    run gh run list \
        --repo "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
        --limit 5
    echo ""
    echo "--- open PRs ---"
    run gh pr list \
        --repo "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
        --state open
    echo ""
    echo "--- branch protection on main ---"
    run gh api "/repos/${TF_REPO_OWNER}/${TF_REPO_NAME}/branches/main/protection" \
        --jq '{required_status_checks: .required_status_checks.contexts, require_pr_review: .required_pull_request_reviews.required_approving_review_count, force_push: .allow_force_pushes.enabled, deletions: .allow_deletions.enabled}'
    echo ""
    echo "--- repo settings ---"
    run gh repo view "${TF_REPO_OWNER}/${TF_REPO_NAME}" \
        --json visibility,hasWikiEnabled,deleteBranchOnMerge,mergeCommitAllowed,squashMergeAllowed,rebaseMergeAllowed
else
    echo "+ gh run list / gh pr list / gh api branches/main/protection / gh repo view (after publish)"
fi

# ==========================================================================
# Done
# ==========================================================================
echo ""
echo "============================================================"
if [[ "$MODE" == "--dry-run" ]]; then
    echo "Dry run complete. No external state changed."
    echo ""
    echo "To execute for real, set env vars and run with --execute:"
    echo "  export HF_TOKEN=hf_xxx          # never commit"
    echo "  export TF_REPO_OWNER=...        # e.g. tetradflow-dev"
    echo "  export TF_REPO_NAME=tetradflow"
    echo "  export TF_HF_NAMESPACE=...      # e.g. tetradflow-dev"
    echo "  export TF_VERSION=v0.1.0"
    echo "  export EVAL_JSON_SHA=\$(sha256sum checkpoints/eval_output.json | cut -d' ' -f1)"
    echo "  bash scripts/release_pipeline.sh --execute"
else
    echo "Release pipeline triggered. Monitor:"
    echo "  https://github.com/${TF_REPO_OWNER}/${TF_REPO_NAME}/actions"
fi
