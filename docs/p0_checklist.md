# P0 Checklist (착수前 48-72h、絶対条件)

All 4 items must be resolved before the 3-month PoC begins.
Failing any item blocks the release pipeline (P0-4: human gate, no auto-degradation).

---

## P0-1: Seed-60 Manual Labels — Cohen's κ ≥ 0.7

**What**: Two human raters independently label 60 examples (15 per Tetrad axis)
using the template in `eval/seed_60_template.md`.

**Verification**:
```bash
# After both raters complete eval/rater_a_labels.json and eval/rater_b_labels.json:
python - <<'EOF'
import json
from sklearn.metrics import cohen_kappa_score

with open("eval/rater_a_labels.json") as f:
    a = json.load(f)
with open("eval/rater_b_labels.json") as f:
    b = json.load(f)

labels_a = [item["axis"] for item in a["items"] if item["axis"]]
labels_b = [item["axis"] for item in b["items"] if item["axis"]]
kappa = cohen_kappa_score(labels_a, labels_b)
print(f"Cohen's kappa: {kappa:.4f}")
assert kappa >= 0.7, f"P0-1 FAIL: kappa={kappa:.4f} < 0.7"
print("P0-1 PASS")
EOF
```

**Fail response**: Rewrite Tetrad axis operational definitions in `eval/seed_60_template.md`
and re-label from scratch. Do NOT proceed to P0-2 until kappa ≥ 0.7.

**Owner**: User (requires 2 human raters).

---

## P0-2: SAE Anchor Fix Removed — CCA Post-hoc Axis Identification

**What**: The BatchTopK SAE is trained fully unsupervised. After training,
CCAAxisFinder maps 4 Tetrad axes to SAE feature indices.
All 6 pairwise cosine similarities between axis canonical loadings must be < 0.3.

**Verification**:
```bash
# Train SAE (GPU required)
python scripts/train_sae.py --output checkpoints/sae.safetensors

# Identify axes via CCA (GPU required, uses labelled prompts from P0-1)
python scripts/identify_axes_cca.py \
  --sae checkpoints/sae.safetensors \
  --labels eval/rater_a_labels.json \
  --output checkpoints/axes_map.safetensors \
  --eval-json checkpoints/eval_output.json

# Check gate
python scripts/human_gate_check.py --eval-json checkpoints/eval_output.json

# Run pytest orthogonality suite
pytest src/tetradflow/eval/direct_metrics.py -v \
  --axes-map checkpoints/axes_map.safetensors
```

**Pass condition**: All 6 pairwise |cosine| < 0.3 AND median ≤ 0.5.

**Fail response**: Re-examine SAE hyperparameters (k, n_features, lr).
Consider increasing dataset size or adding dead-feature monitoring (P1-1).
Do NOT auto-degrade to CFG baseline without explicit user approval (P0-4).

**Owner**: Claude (automated) + User (approval gate).

---

## P0-3: Gram-Schmidt Removed — SVD Top-4 Basis

**What**: `ode.py::svd_top4_basis()` uses `torch.linalg.svd` exclusively.
No Gram-Schmidt code exists anywhere in the codebase.

**Verification**:
```bash
# Grep to confirm no Gram-Schmidt code
grep -r "gram.schmidt\|gram_schmidt\|GramSchmidt" src/ tests/ scripts/
# Expected: no output

# Run ODE tests
pytest tests/test_ode.py -v

# Confirm orthonormality of basis
python - <<'EOF'
import torch
from tetradflow.ode import svd_top4_basis

dirs = torch.randn(4, 4096)
basis = svd_top4_basis(dirs)
gram = basis @ basis.T
print("Gram matrix (should be I_4):")
print(gram.round(decimals=5))
assert torch.allclose(gram, torch.eye(4), atol=1e-4), "Not orthonormal!"
print("P0-3 PASS")
EOF
```

**Fail response**: Any Gram-Schmidt usage must be removed. This is a hard
architectural constraint (F3 fix). Do not reintroduce Gram-Schmidt.

**Owner**: Claude (already implemented). Verify with grep.

---

## P0-4: Automatic Degradation Removed — Human Gate

**What**: The release pipeline NEVER auto-degrades when the PoC gate fails.
`release.yml` runs `human_gate_check.py` before any publish step.
If the gate fails, CI exits 1, blocks release, and requires user approval + re-trigger.

**Verification**:
```bash
# Confirm release.yml has gate-check job as a prerequisite
grep -A5 "needs: gate-check" .github/workflows/release.yml

# Test gate check manually with a failing JSON
python - <<'EOF'
import json, subprocess, sys, tempfile, os
payload = {"gate_pass": False, "median_cosine": 0.6,
           "n_pairs_above_0_6": 3, "pairwise_cosines": {"Enhance-Obsolesce": 0.7}}
with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
    json.dump(payload, f)
    path = f.name
result = subprocess.run([sys.executable, "scripts/human_gate_check.py",
                        "--eval-json", path], capture_output=True)
os.unlink(path)
assert result.returncode == 1, "Gate should exit 1 on failure!"
print("P0-4 PASS: gate correctly blocks on fail")
EOF
```

**Pass condition**: `human_gate_check.py` exits 1 on gate failure and prints
a clear human-review-required message. No code path auto-selects Plan B/C.

**Fail response**: If any code path silently degrades without human approval,
that path must be replaced with a RuntimeError + human instruction message.
This is a non-negotiable constraint (F5 fix).

**Owner**: Claude (already implemented). User verifies no bypass exists.

---

## Summary Gate Table

| # | Item | Owner | Gate command |
|---|---|---|---|
| P0-1 | κ ≥ 0.7 | User (2 raters) | `python -c "... cohen_kappa_score ..."` |
| P0-2 | Cosine < 0.3 | Claude + User | `pytest eval/direct_metrics.py --axes-map ...` |
| P0-3 | No Gram-Schmidt | Claude | `grep -r gram_schmidt src/` → empty |
| P0-4 | Human gate enforced | Claude + User | `python scripts/human_gate_check.py --eval-json ...` exits 1 on fail |

All 4 must PASS before the 3-month PoC begins.
