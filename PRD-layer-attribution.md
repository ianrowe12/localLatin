# PRD: Layer Selection and Attribution Evidence

## Purpose

This work prepares the next evidence bundle for the Latin retrieval paper. The professor's latest feedback is not only asking for more attribution numbers; it is asking for a stronger paper logic:

1. Why do we inspect intermediate transformer layers instead of only the final representation?
2. If layer choice matters, can we give readers a defensible way to choose a useful layer?
3. Are the attribution results stable across reasonable metric hyperparameters?
4. Can we include mT5 alongside LaTa and PhilTa in the attribution story without reverse-engineering the layer choice?

The final output of this work should make the later paper rewrite straightforward. Agents should produce evidence, tables, figures, memos, and patch-ready prose. They should not do a broad final rewrite of `overleaf_drafts/acl_latex.tex` unless their specific task says so.

## Background

The paper studies semantic retrieval for Medieval Latin manuscript fragments. The core retrieval story is that Latin-adapted T5 encoders show a mid-layer collapse under raw mean pooling, where cosine similarity loses semantic separability. ABTT, a train-free geometric correction, removes dominant directions and restores useful separation.

The current attribution work grew out of that story. Earlier runs used 20 curated examples, then 200 random pairs. The 200-random run was superseded because it mixed positive and negative pairs in metrics that are meaningful mainly for positive pairs, and because stale ABTT PC artifacts caused cosine inflation. The current trusted attribution baseline is the 200-positive-pair run for LaTa and PhilTa after refitting PCs through the same mean-pooling path used by the metrics code.

The new requirement is to extend and harden the story:

- Include mT5 in the main attribution experiment.
- Decide attribution layers using a defensible layer-selection rule before rerunning attribution.
- Sweep metric hyperparameters instead of relying only on `Suff@25%` and `Cmpct@0.8`.
- Treat Integrated Gradients and retrieval-adapted MaRC as parallel attribution views.

## Product Goals

### Goal 1: Defensible Layer-Selection Guidance

Produce a layer-selection analysis that explains why layerwise analysis matters and gives readers practical guidance for choosing layers.

Preferred claim:

- Unsupervised geometry diagnostics can flag unstable or collapsed layers and identify candidate layers where ABTT repairs the representation.

Acceptable fallback:

- If unsupervised diagnostics do not cleanly predict the best layer, present a two-stage practical rule: use geometry diagnostics to narrow candidates, then use a small labeled validation set, if available, to choose among them by validation cosine gap or AUROC.

Do not force a deterministic intrinsic selector if the evidence does not support one.

### Goal 2: Three-Model Attribution Scope

Run or prepare attribution evidence for the three main-paper models:

- LaTa
- PhilTa
- mT5-base

Attribution should use the layer rule decided by the layer-selection work. Do not pick mT5's attribution layer opportunistically from the final attribution results.

### Goal 3: Hyperparameter Sweep With One Main-Text Choice

Evaluate attribution metrics across reasonable thresholds:

- Sufficiency: 10%, 25%, 50%
- Comprehensiveness: 10%, 25%, 50%
- Compactness: 0.7, 0.8, 0.9, 0.95
- `rho_LOO`: unchanged primary ranking-faithfulness metric

The main paper should use one global threshold choice across models and methods. Do not choose model-specific or method-specific thresholds. The appendix should expose the full sweep.

### Goal 4: Balanced IG and MaRC Framing

Treat IG and retrieval-adapted MaRC as equal attribution views:

- IG is the standard gradient-based method applied to a cosine scalar.
- MaRC is the learned-mask method adapted from classification to retrieval by optimizing against pairwise cosine with a fixed partner embedding.

The MaRC adaptation is a methodological contribution, but the empirical ABTT attribution claim should not depend only on MaRC. Use both methods to show whether the story survives across attribution families.

### Goal 5: Paper-Ready Evidence

Produce artifacts that make a later paper rewrite efficient:

- Layer-selection memo
- Machine-readable diagnostic tables
- Attribution sweep summaries
- Main-text candidate table/figure
- Appendix sweep table/figure
- Metric and method provenance memo
- Interpretation/caveat memo
- Final rewrite brief and reproducibility audit

## Non-Goals

- Do not launch a broad final rewrite of `acl_latex.tex` during this orchestra.
- Do not expand the main attribution experiment beyond LaTa, PhilTa, and mT5 unless the task explicitly asks for cheap appendix-only context.
- Do not optimize attribution thresholds separately per model or method.
- Do not claim "ABTT improves explanation quality" without metric-specific qualification.
- Do not present the older 200-random or 20-curated attribution bundles as current headline evidence.
- Do not hide Sufficiency or Compactness regressions if they appear.

## Current Decisions

- Main objective: paper defensibility, not simply stronger-looking metrics.
- mT5 must be included in attribution.
- mT5 attribution layer should be decided after the layer-selection analysis.
- Attribution success does not require every metric to improve.
- `rho_LOO` is the primary attribution faithfulness metric because it evaluates the full attribution ranking against actual leave-one-out cosine changes.
- ERASER-style metrics are complementary and should remain visible.
- Main-paper attribution thresholds should be global.
- Appendix should show the full threshold sweep.
- Final paper rewrite happens later, after this evidence bundle is complete.

## Success Criteria

The work is successful if:

1. A reader can understand why intermediate layers are studied and why final-layer-only analysis is insufficient.
2. The paper has a defensible layer-selection rule or an honest two-stage fallback.
3. LaTa, PhilTa, and mT5 attribution results are available at pre-declared selected layers.
4. `rho_LOO` improves consistently enough across IG and MaRC to support a rank-faithfulness claim, or the report clearly explains where it does not.
5. At least one ERASER-style metric, ideally Comprehensiveness, supports the same direction for most cells, or disagreements are clearly documented.
6. The hyperparameter sweep shows whether the main-text threshold choice is representative.
7. Every headline number traces to a source CSV/JSON.
8. The final rewrite brief clearly separates main-text claims from appendix support.

## Key Risks

- mT5 may not follow the same attribution pattern as LaTa and PhilTa.
- Unsupervised layer diagnostics may not cleanly predict retrieval-best layers.
- MaRC's sparsity objective can make Compactness comparisons hard to interpret.
- Cross-variant attribution comparisons conflate two effects: ABTT changes attribution scores and also changes the cosine function being explained.
- Existing artifacts may mix canon/canon_labelled paths, stale PCs, or old layer choices.
- The current checkpoint preserved an empty `src/filter_embeddings_cli.py`; agents should not assume that file is usable unless they inspect it.

## Required Interpretive Guardrails

Use language like:

- "ABTT improves rank-correlation faithfulness under `rho_LOO`, while ERASER-style metrics reveal threshold-dependent tradeoffs."
- "We adapt MaRC to retrieval by replacing a classification logit target with pairwise cosine against a fixed partner representation."
- "Layer selection is guided by representation geometry and validated against retrieval outcomes."

Avoid language like:

- "ABTT uniformly improves attribution quality."
- "The best attribution layer was selected by test attribution performance."
- "Compactness proves MaRC is better than IG" without noting MaRC's built-in sparsity pressure.
- "The 200-random results show..." unless explicitly labeled as superseded.

## Main Outputs by Orchestra Run

### Run 1

- Layer-selection diagnostic package.
- Metric and method provenance memo.
- mT5 attribution pipeline readiness.

### Run 2

- Hard decision memo naming attribution layers for LaTa, PhilTa, and mT5.

### Run 3

- Three-model attribution artifacts.
- Metric sweep implementation and summary.

### Run 4

- Main-text candidate attribution table/figure.
- Appendix sweep package.
- Interpretation and caveat memo.

### Run 5

- Final paper rewrite brief.
- Reproducibility and consistency audit.

## Agent Operating Notes

- Read `CLAUDE.md` first for repo conventions.
- Read this PRD before starting task-specific work.
- Read `ORCHESTRA-layer-attribution.md` for run order and task prompt.
- Preserve existing user work. Do not revert unrelated changes.
- Use worktrees exactly as instructed in the orchestra prompt.
- Commit your branch when done; do not merge your own branch unless you are the merge agent.
- Prefer small reproducibility scripts and machine-readable summaries over one-off manual calculations.
