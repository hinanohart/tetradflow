# Disclaimer

TetradFlow is a **research prototype**. It is not a validated cognitive model
of McLuhan's media theory, and it does not claim to mechanistically implement
or verify any philosophical proposition from McLuhan's work.

## Scope Limitations

1. **Empirical validation pending**: The mapping between SAE feature clusters
   and McLuhan's Tetrad axes is operationally defined by a small manual label
   set (60 examples, P0-1). This has not been peer-reviewed or validated by
   McLuhan scholars.

2. **Correlation, not causation**: CCA identifies statistical associations
   between SAE activations and axis labels. This does not establish that the
   model "understands" Tetrad concepts in any meaningful sense.

3. **Research stage**: TetradFlow is at PoC (proof-of-concept) stage.
   The 3-month PoC condition (cosine < 0.3 orthogonality, P0-2) has not yet
   been experimentally verified at time of initial release.

4. **No clinical or safety-critical use**: This software is not intended for
   use in clinical, legal, safety-critical, or high-stakes decision-making
   contexts.

5. **SAE instability**: Sparse autoencoder feature spaces are known to change
   across model versions. Features identified in one Janus-Pro version may
   not transfer to future versions without re-training.

6. **Eval suite limitations**: The eval suite (direct_metrics, ASS, human_eval)
   provides necessary but not sufficient evidence for Tetrad alignment. External
   McLuhan expert review (P2-1) is required before making strong claims.

## Intended Use

- Interpretability research on multimodal language models
- Experimental art and media studies exploration
- Academic study of SAE-based concept alignment

## Not Intended For

- Production image generation without expert oversight
- Claims about the computational validity of McLuhan's media theory
- Automated content moderation or decision systems

---

*This disclaimer is displayed in the HF Spaces interface and referenced in
the README. It fulfils P2-4 from the TetradFlow architecture spec.*
