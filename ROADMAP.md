# TetradFlow Roadmap

> **Status**: CONDITIONAL GO (3-month PoC, P0 conditions pending).
> See [docs/p0_checklist.md](docs/p0_checklist.md) for blocking prerequisites.

---

## P0: Prerequisites (~1 week, blocking)

All items block PoC start. Failure stops the pipeline with a human review gate.

| Item | Acceptance Criteria | Owner | Status |
|---|---|---|---|
| P0-1: Seed-60 label κ ≥ 0.7 | `cohen_kappa_score(rater_a, rater_b) >= 0.7` | **User + 1 rater** | TODO |
| P0-2: SAE cosine orthogonality | All 6 pairwise `|cosine| < 0.3` in `direct_metrics.py` | **Claude** (automated) + User (approval) | In repo (needs GPU run) |
| P0-3: No Gram-Schmidt | `grep -r gram_schmidt src/` → empty | **Claude** | DONE |
| P0-4: Human gate enforced | `human_gate_check.py` exits 1 on fail; no auto-degrade code path | **Claude** | DONE |

---

## P1: PoC Month 1-2

| Item | Acceptance Criteria | Owner | Status |
|---|---|---|---|
| P1-1: Dead-feature rate monitoring | Activation histogram + dead-feature rate logged per SAE checkpoint | **Claude** | TODO |
| P1-2: Linear map orthogonality ablation | Janus 4096-dim → Flux velocity space linear map: orthogonality preservation measured | **Claude** | TODO |
| P1-3: ASS vs CFG baseline | Bootstrap CI lower bound of (TetradFlow ASS - CFG ASS) > 0 | **Claude** + User (GPU run) | In repo (`eval/ass.py`); needs GPU eval |
| P1-4: Gauge α validation | FID + human eval comparing α=learned vs α=constant baseline | **User** (GPU) + Claude (eval script) | TODO |

---

## P2: Release Prerequisites (before v0.1.0 tag)

| Item | Acceptance Criteria | Owner | Status |
|---|---|---|---|
| P2-1: McLuhan expert review | n=30 forced choice, 4-axis perception rate documented | **User** (recruit expert) | TODO |
| P2-2: SAE adapter (LoRA-style) | SAE fine-tune cost < full retrain when base model updates | **Claude** | TODO |
| P2-3: 4-divergent ODE human eval suite | n=30 forced choice comparing TetradFlow vs CFG baseline | **User** + Claude | Template in `eval/human_eval.py` |
| P2-4: Disclaimer in README + Spaces | "research prototype, not validated cognitive model" visible | **Claude** | DONE |

---

## v1.0 (Post-PoC, Conditional)

Items below are contingent on PoC success (P0-2 cosine gate passing at 3 months).

| Item | Description | Owner |
|---|---|---|
| Show-o bidirectional text | Full integration of Show-o for grounded text-image generation | Claude + User |
| LanguageBind multi-modal | Video/audio axis alignment via LanguageBind (PKU, MIT license) | Claude + User |
| SAE adapter (LoRA-style) | Model-version-stable SAE via adapter pattern (P2-2) | Claude |
| Multi-agent self-evolution | Darwin Gödel Machine / Project Sid style self-modification | Future |

---

## Degradation Plan (if P0-2 gate fails at 3 months)

| Plan | Trigger | Description |
|---|---|---|
| Plan A continue | User explicit choice | Extend PoC, retry SAE training |
| Plan B Lite | User `--fallback=plan-b` | Drop Flux, keep Janus-Pro + Show-o + SAE |
| Plan C | User `--fallback=plan-c` | TetradSAE standalone library (no generation) |

**No automatic fallback** — all degradation requires explicit user command (P0-4).

---

## Known Risks

| Risk | Probability | Mitigation |
|---|---|---|
| Tetrad axis cosine > 0.3 (P0-2 fail) | Medium | Increase labelled dataset, tune SAE k |
| SAE dead-feature explosion | Medium | Monitor P1-1 histogram, adjust sparsity |
| McLuhan expert review fails (P2-1) | Unknown | Clarify axis operational definitions |
| Flux API breaking changes | Low | Pin diffusers version, track changelog |
| HF Spaces ZeroGPU quota exceeded | Medium | Reduce inference steps default to 4 |

---

## Open Questions (Research)

1. Is the Tetrad really 4-dimensional in SAE space, or do axes collapse?
2. Does `svd_top4_basis` give semantically meaningful directions, or arbitrary rotation?
3. What is the right `gamma` value for the ODE injection (currently 2.0)?
4. Can LanguageBind replace SigLIP path for richer multi-modal Figure/Ground?
5. Is the CCA post-hoc identification stable across SAE random seeds?

---

## Claude-unreachable boundary (operational, 2026-05-17)

The items below are not automatable from inside the AI coding session. They
require either a GPU host that the assistant cannot reach, multiple humans
performing independent judgement, or an interactive third-party Web UI.
Listed for traceability; the `scripts/release_pipeline.sh --execute` flow
expects the first batch to be complete before being run.

| Boundary | Why unreachable | Who | Replacement Claude does instead |
|---|---|---|---|
| κ ≥ 0.7 manual labelling (P0-1) | 2 independent human raters by design | User + 1 rater | Provides `eval/seed_60_template.md`, `eval/manual_labels_template.json`, and (P1) `scripts/compute_kappa.py` once written |
| GPU SAE training (P0-2 production run) | ~52 GB VRAM BF16, no GPU in session | User (A100/H100 / Modal / RunPod) | Provides `scripts/train_sae.py` with `--scaffold-test` smoke path; CPU CI runs the loop on dummy data |
| CCA on real SAE (P0-2 numerical) | Depends on above | User | Provides `scripts/identify_axes_cca.py` |
| PyPI Trusted Publishing registration | pypi.org has Web UI only, no API | User | `scripts/release_pipeline.sh` prints the URL and field values |
| McLuhan expert external review (P2-1) | Domain-expert human panel | User | Provides forced-choice CSV harness in `eval/human_eval.py` |
| 4-divergent ODE human eval (P2-3) | ≥ 2 independent raters required | User | Provides `eval/human_eval_4divergent.md` protocol |

**Correction (2026-05-17, AN1)**: "GitHub Environments required reviewers"
was previously listed here as UI-only. This was wrong — the REST API
`PUT /repos/{owner}/{repo}/environments/{env}` accepts a `reviewers` body
since 2023. `scripts/release_pipeline.sh:S4.5` now sets it programmatically
when `TF_REVIEWER_USER_ID` is exported. The boundary count drops from 7 → 6.

## Improvement DAG (T-N mapping, post-skeleton, 2026-05-17)

The original architecture memo references a T-1 … T13 DAG. Where each task
lives in the current repo:

| Tag | Item | Location |
|---|---|---|
| T-1 | P0 conditions (κ, anchor-free SAE, SVD, human gate) | rows above + P0-1 to P0-4 |
| T0  | Skeleton (pyproject / LICENSE / CI) | `pyproject.toml`, `LICENSE`, `.github/workflows/ci.yml` |
| T1  | CCA + SVD axes | `src/tetradflow/cca.py`, `src/tetradflow/ode.py` |
| T2  | Janus backend / hooks | `src/tetradflow/hooks.py`, `pipeline.py::_load_janus` |
| T3  | Flux + Show-o (deferred) + LanguageBind (deferred) | `pipeline.py::_load_flux` + ROADMAP v1.0 |
| T4  | SAE training | `scripts/train_sae.py` (G1 dummy trap added) |
| T5  | P0-2 cosine metric | `src/tetradflow/eval/direct_metrics.py` |
| T6  | Pipeline + CLI | `src/tetradflow/pipeline.py`, `cli.py` |
| T7  | Eval (ASS / human / direct) | `src/tetradflow/eval/*.py`, `eval/*.md` |
| T8  | Tests | `tests/*.py` |
| T9  | Docs | `docs/*.md`, `README.md`, `THIRD_PARTY_LICENSES.md` |
| T10 | Spaces app | `spaces/app.py` |
| T11 | GitHub Actions | `.github/workflows/*.yml` (+ security G3 added) |
| T12 | PoC gate | `scripts/human_gate_check.py` + `release.yml::gate-check` |
| T13 | v0.1.0 release | `scripts/release_pipeline.sh` (S1-S7) + `release.yml` |
