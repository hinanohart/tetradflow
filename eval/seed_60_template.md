# Seed-60 Manual Label Template (P0-1)

60 examples for Tetrad axis manual annotation (15 per axis + neutral items).
Two raters must label independently. Cohen's kappa >= 0.7 required before CCA training.

## Axis Definitions

| Axis | McLuhan Definition | Operational Test |
|---|---|---|
| **Enhance** | What does the medium amplify or intensify? | The image/prompt foregrounds a capability being pushed to its limit or made more powerful |
| **Obsolesce** | What does the medium push aside or make obsolete? | The image/prompt depicts replacement, redundancy, or decay of a prior medium/practice |
| **Retrieve** | What does the medium resurrect from the past? | The image/prompt evokes archaic or dormant forms brought back in new context |
| **Reverse** | What does the medium become when pushed to its limits? | The image/prompt shows a medium flipping into its opposite at saturation |

**Neutral**: Items that do not clearly belong to any single axis (adversarial examples, 25% of training set).

---

## Rating Instructions

1. Read each prompt carefully.
2. Choose **one** axis label: `Enhance`, `Obsolesce`, `Retrieve`, `Reverse`, or `Neutral`.
3. Record your choice in the `axis` field of `eval/manual_labels_template.json`.
4. Do not discuss with the other rater until both have finished.
5. After both raters complete, run the kappa script (see below).

---

## Cohen's Kappa Calculation

```python
# scripts/compute_kappa.py (example usage)
import json
from sklearn.metrics import cohen_kappa_score

with open("eval/rater_a_labels.json") as f:
    a = json.load(f)
with open("eval/rater_b_labels.json") as f:
    b = json.load(f)

labels_a = [item["axis"] for item in a["items"]]
labels_b = [item["axis"] for item in b["items"]]

kappa = cohen_kappa_score(labels_a, labels_b)
print(f"Cohen's kappa: {kappa:.4f}")
if kappa >= 0.7:
    print("P0-1 gate: PASS")
else:
    print("P0-1 gate: FAIL — rewrite axis operational definitions and re-label")
```

Target: **kappa >= 0.7**. If fail, revise the Axis Definitions table above and repeat.

---

## Item List (60 items = 15 per axis + 15 neutral)

### Enhance (EN01–EN15)

Fill in 15 prompts that clearly exemplify a medium enhancing its core capability.
Example theme ideas: high-fidelity audio, satellite surveillance, instant translation, algorithmic trading.

| ID | Prompt (to fill) | Rater A | Rater B |
|---|---|---|---|
| EN01 | | | |
| EN02 | | | |
| EN03 | | | |
| EN04 | | | |
| EN05 | | | |
| EN06 | | | |
| EN07 | | | |
| EN08 | | | |
| EN09 | | | |
| EN10 | | | |
| EN11 | | | |
| EN12 | | | |
| EN13 | | | |
| EN14 | | | |
| EN15 | | | |

### Obsolesce (OB01–OB15)

Fill in 15 prompts that clearly exemplify a medium making a prior form obsolete.
Example theme ideas: streaming vs physical media, GPS vs paper maps, e-mail vs postal mail, digital print vs letterpress.

| ID | Prompt (to fill) | Rater A | Rater B |
|---|---|---|---|
| OB01 | | | |
| OB02 | | | |
| OB03 | | | |
| OB04 | | | |
| OB05 | | | |
| OB06 | | | |
| OB07 | | | |
| OB08 | | | |
| OB09 | | | |
| OB10 | | | |
| OB11 | | | |
| OB12 | | | |
| OB13 | | | |
| OB14 | | | |
| OB15 | | | |

### Retrieve (RE01–RE15)

Fill in 15 prompts that clearly exemplify a medium retrieving a dormant or archaic form.
Example theme ideas: vinyl revival, film photography renaissance, oral storytelling via podcast, handwritten letters in digital art.

| ID | Prompt (to fill) | Rater A | Rater B |
|---|---|---|---|
| RE01 | | | |
| RE02 | | | |
| RE03 | | | |
| RE04 | | | |
| RE05 | | | |
| RE06 | | | |
| RE07 | | | |
| RE08 | | | |
| RE09 | | | |
| RE10 | | | |
| RE11 | | | |
| RE12 | | | |
| RE13 | | | |
| RE14 | | | |
| RE15 | | | |

### Reverse (RV01–RV15)

Fill in 15 prompts that clearly exemplify a medium flipping into its opposite at saturation.
Example theme ideas: 24/7 connectivity → isolation, information overload → ignorance, personalization → filter bubble, speed → paralysis.

| ID | Prompt (to fill) | Rater A | Rater B |
|---|---|---|---|
| RV01 | | | |
| RV02 | | | |
| RV03 | | | |
| RV04 | | | |
| RV05 | | | |
| RV06 | | | |
| RV07 | | | |
| RV08 | | | |
| RV09 | | | |
| RV10 | | | |
| RV11 | | | |
| RV12 | | | |
| RV13 | | | |
| RV14 | | | |
| RV15 | | | |

### Neutral / Adversarial (NT01–NT15)

Fill in 15 prompts that deliberately do NOT fit any single axis (adversarial examples).
These test the model's ability to avoid spurious axis assignment.

| ID | Prompt (to fill) | Rater A | Rater B |
|---|---|---|---|
| NT01 | | | |
| NT02 | | | |
| NT03 | | | |
| NT04 | | | |
| NT05 | | | |
| NT06 | | | |
| NT07 | | | |
| NT08 | | | |
| NT09 | | | |
| NT10 | | | |
| NT11 | | | |
| NT12 | | | |
| NT13 | | | |
| NT14 | | | |
| NT15 | | | |

---

## After Labelling

1. Export each rater's labels to `eval/rater_a_labels.json` and `eval/rater_b_labels.json`
   (same schema as `eval/manual_labels_template.json`).
2. Run `python scripts/compute_kappa.py` to compute Cohen's kappa.
3. If kappa >= 0.7: proceed to `python scripts/train_sae.py` (P0-2).
4. If kappa < 0.7: revise axis definitions, re-label, repeat.
