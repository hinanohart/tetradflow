# Human Eval Suite — 4-Divergent ODE Output (P2-3 harness)

This document is the protocol + scoring procedure for the P2-3 release gate
(see [docs/p0_checklist.md](../docs/p0_checklist.md) and
[ROADMAP.md](../ROADMAP.md)).

> **What this suite gates**: whether the 4-divergent ODE produces *perceptually
> distinct* outputs along the four Tetrad axes (Enhance / Obsolesce / Retrieve
> / Reverse) — not whether the axes match McLuhan theory (that is P2-1, a
> separate gate run by domain experts).

## When to run

Run before the v0.1.0 release tag is pushed, and any time
`src/tetradflow/ode.py::tetrad_step` or the CCA axis identifier in
`src/tetradflow/cca.py` is meaningfully changed.

The matching aggregation script:
[`scripts/aggregate_human_eval.py`](../scripts/aggregate_human_eval.py)
*(P2-3 milestone — to be implemented; this document specifies the interface
the script must consume).*

## Suite spec

| Parameter | Value |
|---|---|
| Prompts | 30 hand-picked prompts (see "Prompt set" below) |
| Outputs per prompt | 4 images (one per axis: Enhance, Obsolesce, Retrieve, Reverse) |
| Rater count | ≥ 2 independent raters (R11 / R13: raters must not see each other's labels) |
| Rater task | For each output, pick the **single most fitting axis label** from the 4-axis set |
| Inter-rater κ floor | Cohen's κ ≥ 0.4 (moderate agreement) — below this the test is inconclusive |
| Per-axis accuracy floor | Each axis ≥ 0.50 (chance is 0.25) |
| Mean accuracy floor | Overall ≥ 0.55 with bootstrap 95% CI lower bound > 0.30 |

> A 30-prompt × 4-axis suite = 120 outputs per rater. Plan ~2 hours per rater
> to label end-to-end with breaks.

## Prompt set

The prompt set lives in [`eval/seed_60_template.md`](seed_60_template.md);
use prompts 1–30 (skipping the 30 NT* neutral prompts which are for κ
calibration, not for ODE eval).

If the prompt set is revised, append a row to the "Revisions" table at the
end of this document with the date and SHA256 of the new prompt list.

## Rating CSV schema

Each rater fills one CSV file:

| column | type | meaning |
|---|---|---|
| `prompt_id` | str | matches `eval/seed_60_template.md` ID (e.g. `P01`) |
| `prompt_text` | str | the actual prompt string (sanity check, identical across raters) |
| `output_position` | str | one of `top_left`, `top_right`, `bottom_left`, `bottom_right` |
| `intended_axis` | str | hidden from rater; populated after blind labelling for scoring |
| `rater_label` | str | one of `Enhance`, `Obsolesce`, `Retrieve`, `Reverse`, `unsure` |
| `confidence` | int | 1–5 Likert (1 = guess, 5 = certain) |
| `notes` | str | free text; optional |

The rater **must not see** `intended_axis` while labelling. The eval driver
shuffles output positions per prompt so the per-cell axis is uncorrelated
across prompts.

CSV file naming: `eval/human_eval_4divergent_raterA_<date>.csv`,
`...raterB_<date>.csv`, … one per rater.

## Aggregation gate

`scripts/aggregate_human_eval.py` (P2-3, to be implemented) must report:

- **Per-rater accuracy** by axis (4 values per rater).
- **Cross-rater Cohen's κ** computed pairwise; report median.
- **Overall accuracy bootstrap 95% CI** (n_bootstrap = 10_000, paired by prompt).
- **Gate decision**: PASS / FAIL based on the three floors above.

The script must exit 1 on FAIL so it can chain into the release.yml gate.

## Failure modes (do NOT bypass)

- **κ < 0.4** → axes are not perceptually distinguishable. *Action*: revisit
  the CCA top-k feature selection or the seed prompt set; do NOT release.
- **One axis accuracy ≥ 0.50, others < 0.30** → mode collapse on one axis.
  *Action*: review SAE training dead-feature rate (P1-1).
- **Cosines pass (P0-2) but humans pass (P2-3) fails** → axes are mathematically
  orthogonal but semantically unsignaled. *Action*: this is the F4 (gauge α
  circularity) failure mode — escalate to a McLuhan expert review (P2-1) before
  doing anything else.

## R8 / R11 / R13 notes for raters

- Raters **must not** share their CSVs with each other until aggregation is
  complete. Inter-rater leakage invalidates κ.
- Raters **must not** be prompted by the original author of the ODE code
  ("here's what each axis 'should' look like"). The whole point is independent
  judgement.
- The eval driver does **not** transmit prompts or outputs to any third-party
  service. Outputs are produced locally and CSVs stay on the rater's machine
  until they hand back the file.

## Revisions

| Date | Prompt SHA256 | Notes |
|---|---|---|
| 2026-05-17 | *initial* | Harness spec only; aggregation script is P2-3 deliverable. |
