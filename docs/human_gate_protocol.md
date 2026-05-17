# Human Gate Protocol (P0-4)

This document defines the mandatory human review gate for TetradFlow releases.

**Core principle (P0-4)**: When the PoC gate fails (CI red), the system MUST NOT
automatically degrade to a fallback configuration. All degradation decisions
require explicit human approval and manual re-trigger.

---

## Gate Trigger Conditions

The gate fires (CI exits 1, release blocked) when ANY of the following is true:

| Condition | Threshold | Location |
|---|---|---|
| Any pairwise cosine ≥ 0.3 | All 6 pairs must be < 0.3 | `checkpoints/eval_output.json` |
| Median pairwise cosine > 0.5 | ≤ 0.5 | same |
| 2+ pairs with cosine > 0.6 | < 2 pairs | same |
| `gate_pass` field is false | must be true | same |

---

## scripts/human_gate_check.py Usage

```bash
# Basic usage (run from repo root)
python scripts/human_gate_check.py --eval-json checkpoints/eval_output.json

# Verbose output
python scripts/human_gate_check.py --eval-json checkpoints/eval_output.json --verbose

# Exit codes:
#   0 = gate passes, release may proceed
#   1 = gate fails, release BLOCKED, human review required
```

The script prints GitHub Actions `::error::` annotations when running in CI.

---

## CI Release Pipeline (`.github/workflows/release.yml`)

```
tag push "v*"
    │
    ▼
gate-check job (human_gate_check.py)
    │ ── FAIL → CI exits 1, release BLOCKED ──────────────────┐
    │                                                          │
    │ ── PASS ─────────────────────────────────────────────┐  │
    ▼                                                       │  │
pypi job (Trusted Publishing, OIDC)                        │  │
    │                                                       │  │
    ▼                                                       │  │
hf-hub job (HF_TOKEN secret)                               │  │
    │                                                       │  │
    ▼                                                       │  │
spaces-deploy job                                           │  │
                                                            │  │
                ← release complete ─────────────────────────┘  │
                ← BLOCKED, awaiting human ──────────────────────┘
```

---

## When Gate Fails: Human Review Procedure

1. **Read the CI log** — `human_gate_check.py` prints all pairwise cosine values
   and identifies which pairs violated the threshold.

2. **Do NOT auto-degrade** — Do not manually trigger a release with a lower
   threshold or a fallback model. This constitutes a governance violation (P0-4).

3. **Diagnose** — Common failure causes:
   - Insufficient training data (< 60 labelled examples with κ ≥ 0.7)
   - SAE dead features (monitor with P1-1 dead-feature rate histogram)
   - CCA `top_k` too large (try reducing to 16 or 8)
   - Neutral/adversarial examples too few (should be 25% of training set)

4. **Fix** — Re-run one or more of:
   ```bash
   # Option A: Retrain SAE with different hyperparameters
   python scripts/train_sae.py --output checkpoints/sae_v2.safetensors --k 32

   # Option B: Re-run CCA with different top_k
   python scripts/identify_axes_cca.py \
     --sae checkpoints/sae_v2.safetensors \
     --labels eval/rater_a_labels.json \
     --output checkpoints/axes_map_v2.safetensors \
     --top-k 16 \
     --eval-json checkpoints/eval_output.json
   ```

5. **Verify locally**:
   ```bash
   python scripts/human_gate_check.py --eval-json checkpoints/eval_output.json
   # Must exit 0 before proceeding
   ```

6. **Commit the new eval JSON** to the repo:
   ```bash
   git add checkpoints/eval_output.json
   git commit -m "Update eval output after SAE retrain (gate fix)"
   ```

7. **Manually re-tag** to trigger release:
   ```bash
   git tag v0.1.1
   # git push origin v0.1.1  (user executes, R13)
   ```

---

## Degradation Plan Approval (If Needed)

If after 3 re-train attempts the P0-2 cosine gate cannot be met, the user
must explicitly choose one of:

| Plan | Description | Trigger |
|---|---|---|
| Plan A continue | Keep retrying SAE training (extend PoC) | User decision |
| Plan B Lite | Drop Flux, keep Janus-Pro + Show-o + SAE | User explicit `--fallback=plan-b` flag |
| Plan C | TetradSAE standalone library (no generation) | User explicit `--fallback=plan-c` flag |

**The system will NEVER automatically choose Plan B or Plan C** (P0-4).
The `--fallback` flag must be explicitly passed by the user.

---

## Audit Trail

All gate check results are recorded in `checkpoints/eval_output.json`.
This file should be committed to the repo for audit purposes.
Do not delete or overwrite without re-running the full P0-2 pipeline.
