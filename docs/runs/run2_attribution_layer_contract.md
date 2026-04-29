# Run 2 Attribution Layer Contract

Generated: 2026-04-29

## Decision

Run 3 attribution must explain the retrieval-selected operational layer:

| Model | Run 3 attribution layer | Contrast layer for collapse/recovery diagnostics | Rule source |
|---|---:|---:|---|
| LaTa | 7 | 8 | Earliest layer within 0.5 pp of best train `dir_acc_at_1__abtt_optimal` |
| PhilTa | 1 | 6 | Earliest layer within 0.5 pp of best train `dir_acc_at_1__abtt_optimal` |
| mT5-base | 1 | 5 | Earliest layer within 0.5 pp of best train `dir_acc_at_1__abtt_optimal` |

This supersedes the professor-share meeting bundle's temporary LaTa L4 / PhilTa
L6 attribution framing. Those layers were useful for the meeting because they
made the collapse-recovery story concrete, but Run 3 needs a stable operational
rule before computing attribution metrics.

## Paper Reporting Plan

Use the retrieval-selected layers for the main attribution table. This table
should carry the operational claim: after a predeclared train-only layer
selection rule, do the attribution methods faithfully explain the same layer the
retrieval pipeline uses?

After the main table, include a mechanism figure using the recovered-collapse
diagnostic layers: LaTa L8, PhilTa L6, and mT5-base L5. This figure should not
replace the main table or reselect the headline attribution layers. Its job is
to make the geometric story visible: the baseline layer is collapsed under
PC1/effective-rank diagnostics, ABTT-D10 restores rank/cosine spread, and the
token-level attribution view becomes more interpretable in that repaired space.

Recommended framing:

- Main table: "operational attribution at retrieval-selected layers."
- Mechanism figure: "why ABTT matters at recovered-collapse layers."
- Appendix: full layer diagnostics and, if Run 3 has budget, attribution checks
  for the recovered-collapse layers as non-selector diagnostic evidence.

## Two Defensible Choices

**Choice A: explain the retrieval-selected layer.** This is the selected rule.
The layer is chosen before Run 3 attribution metrics by a predeclared retrieval
criterion on training data: choose the earliest layer within 0.5 percentage
points of the best `train_dir_acc_at_1__abtt_optimal`. It keeps the attribution
claim aligned with the retrieval system we would actually deploy and avoids
presenting intrinsic geometry as a best-layer oracle.

**Choice B: explain the recovered-collapse diagnostic layer.** This is
defensible for a mechanism appendix. The diagnostic layers are the strongest
middle-layer collapse points under PC1 dominance / effective-rank collapse:
LaTa L8, PhilTa L6, and mT5-base L5. They are exactly where ABTT-D10 visibly
recovers rank and cosine spread. The weakness is that they do not consistently
match the train-selected retrieval optimum, so making them the main attribution
layers would shift the claim from "this explains retrieval behavior" to "this
explains an illustrative failure mode."

## Why This Is Not Reverse-Engineered

The rule was fixed from Run 1 layer diagnostics and retrieval artifacts before
the Run 3 attribution experiment. It does not use attribution faithfulness,
sufficiency, compactness, MaRC, IG, or any test attribution metric as a selector.
The current 200-positive attribution summary was read only as context for why
the earlier L4/L6 meeting framing existed; it is not part of the layer choice.

The inputs for this decision are:

- `docs/analyses/layer_geometry_diagnostics.md`
- `runs/active/resubmit/layer_diagnostics/layer_rule_candidates.csv`
- `professor_share/2026-04-27/READY.md`
- `professor_share/2026-04-26/QA_PREP_2026_04_27.md`
- `runs/active/ig_examples_200pos/attribution_metrics/summary.csv`
- `scripts/ig/attribution_model_config.py`

## Fallback Guidance

If unsupervised diagnostics are imperfect, do not reselect layers from Run 3
attribution outcomes. Keep the train-selected operational layer for the main
Run 3 table, and report recovered-collapse layers as diagnostic appendix checks
when needed. If a future artifact lacks the train-selected retrieval sweep, use
the documented held-out fallback only with an explicit label that it is a
held-out retrieval selection, not an intrinsic diagnostic selection.

The machine-readable contract is `scripts/ig/attribution_model_config.py`:
LaTa L7, PhilTa L1, mT5-base L1, D=10.
