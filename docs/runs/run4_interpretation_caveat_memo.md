# Run 4 Interpretation and Caveat Memo

Generated: 2026-04-30

## Source of Truth

This memo is for the later paper rewrite. It interprets the active Run 3 attribution bundle, not the older 20-pair, 200-random, or two-model professor-share artifacts.

Primary inputs:

- `docs/analyses/layer_geometry_diagnostics.md`
- `docs/runs/run2_attribution_layer_contract.md`
- `docs/research/run1_metric_provenance.md`
- `runs/active/ig_examples_200pos_run3_operational/manifest.json`
- `runs/active/ig_examples_200pos_run3_operational/artifact_inventory.csv`
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv`

The active attribution bundle contains 200 positive query-candidate pairs for each of LaTa, PhilTa, and mT5-base. The manifest reports complete IG and retrieval-adapted MaRC artifacts for baseline and ABTT variants, with no warnings or artifact-method errors.

## Layer-Selection Interpretation

The layer-selection analysis supports a diagnostic, not oracle-like, role for intrinsic geometry. Intermediate layers are worth inspecting because they expose the failure mode that motivates ABTT: strong top-PC dominance, low effective rank, and cosine concentration in collapsed spaces. At the diagnostic collapse layers, ABTT-D10 sharply reduces PC1 dominance and restores effective rank. For the main models, the diagnostic collapse layers were LaTa L8, PhilTa L6, and mT5-base L5.

The same analysis does not show that unsupervised geometry alone reliably selects the best retrieval layer. The strongest collapse layer and the train-selected retrieval layer differ for all three main models except where the train rule happens to choose the earliest layer. The rewrite should therefore avoid claiming that a label-free diagnostic directly chooses the operational layer.

The attribution layer rule used for Run 3 was fixed before attribution metrics were computed:

| Model | Main attribution layer | Diagnostic collapse layer | Rule |
|---|---:|---:|---|
| LaTa | 7 | 8 | Earliest layer within 0.5 percentage points of the best train `dir_acc_at_1__abtt_optimal` |
| PhilTa | 1 | 6 | Same rule |
| mT5-base | 1 | 5 | Same rule |

Main-paper attribution should explain these retrieval-selected operational layers. The diagnostic collapse layers can support a mechanism figure or appendix discussion, but they should not replace the main attribution layers.

## Attribution Result Interpretation

Treat IG and retrieval-adapted MaRC as equal attribution views. IG is the standard gradient-based view applied to the retrieval cosine scalar. MaRC is the learned-mask view adapted from classification to bi-encoder retrieval by optimizing a soft query-side mask to preserve pairwise cosine against a fixed candidate representation; candidate-side attribution is obtained by swapping sides. That retrieval adaptation is a methodological contribution, but the empirical ABTT claim must survive both attribution families rather than depending on MaRC alone.

The cleanest empirical attribution claim is about rank faithfulness. `rho_LOO` improves under ABTT for all six main model-view cells:

| Model | View | `rho_LOO` baseline -> ABTT | Delta |
|---|---|---:|---:|
| LaTa | IG | -0.001 -> 0.396 | +0.397 |
| LaTa | MaRC | 0.042 -> 0.401 | +0.359 |
| PhilTa | IG | 0.182 -> 0.607 | +0.425 |
| PhilTa | MaRC | 0.190 -> 0.331 | +0.141 |
| mT5-base | IG | 0.164 -> 0.597 | +0.433 |
| mT5-base | MaRC | 0.250 -> 0.392 | +0.142 |

This supports wording like "ABTT improves leave-one-out rank faithfulness across both IG and retrieval-adapted MaRC." It does not support the broader claim that ABTT uniformly improves every attribution-quality metric.

## ERASER-Style Metrics

At the main thresholds, the ERASER-style metrics are mixed:

| Model | View | Suff@25 | Comp@25 | MinFrac@0.80 | Main-threshold wins including `rho_LOO` |
|---|---|---:|---:|---:|---:|
| LaTa | IG | 0.976 -> 0.681 | 0.719 -> 0.581 | 0.041 -> 0.379 | 1/4 |
| LaTa | MaRC | 0.534 -> 0.645 | 0.361 -> 0.539 | 0.383 -> 0.459 | 3/4 |
| PhilTa | IG | 0.959 -> 0.963 | 0.969 -> 0.747 | 0.079 -> 0.141 | 2/4 |
| PhilTa | MaRC | 0.911 -> 0.754 | 0.963 -> 0.521 | 0.104 -> 0.358 | 1/4 |
| mT5-base | IG | 0.892 -> 1.027 | 0.110 -> 0.717 | 0.116 -> 0.214 | 3/4 |
| mT5-base | MaRC | 0.913 -> 0.827 | 0.042 -> 0.592 | 0.066 -> 0.346 | 2/4 |

Higher is better for Sufficiency and Comprehensiveness; lower is better for MinFrac.

Across the threshold sweep stored in the current summary, Sufficiency improves in 9 of 18 cells across 10%, 25%, and 50%. Comprehensiveness improves in 10 of 18 cells. MinFrac@0.80 worsens in all six IG/MaRC cells, meaning ABTT generally needs a larger retained token fraction to recover 80% of the full cosine. This disagreement must stay visible in the main prose or table caption.

The strongest agreement with `rho_LOO` comes from mT5: Comprehensiveness improves under ABTT for both IG and MaRC at all three tested fractions, and IG Sufficiency improves at 25% and 50%. LaTa-MaRC also aligns well: Sufficiency and Comprehensiveness improve at all three fractions, while MinFrac worsens. PhilTa is the clearest caveat: `rho_LOO` improves for both views, but Comprehensiveness drops for both views and MaRC Sufficiency also drops.

## Recommended Paper Wording

Use this as the main interpretation:

```latex
Layerwise diagnostics show that the relevant failure mode is geometric rather
than simply a property of the final encoder layer: several intermediate layers
collapse into a low-rank cone, and ABTT-D10 restores effective rank and cosine
spread. Because these intrinsic diagnostics identify collapse regimes but do
not reliably select the best retrieval layer, we choose attribution layers with
a predeclared train-only retrieval rule and reserve the strongest collapse
layers for mechanism checks.
```

```latex
We evaluate two complementary attribution views for the same retrieval score.
Integrated gradients provides a standard gradient-based attribution of the
pairwise cosine, while our retrieval-adapted MaRC view optimizes a soft input
mask to preserve the same cosine against a fixed partner embedding. Across
LaTa, PhilTa, and mT5-base, ABTT improves leave-one-out rank faithfulness
(`rho_LOO`) for both views. The ERASER-style metrics are less uniform:
Comprehensiveness and Sufficiency improve for some model-view pairs, especially
mT5 and LaTa-MaRC, while the minimum retained fraction at 80% recovery generally
worsens. We therefore interpret attribution as supporting the ABTT repair story
through rank faithfulness, with threshold-dependent tradeoffs rather than a
uniform improvement in every rationale metric.
```

For the MaRC contribution:

```latex
The methodological adaptation is to replace MaRC's classification objective
with a bi-encoder retrieval objective. For each query-candidate pair, the mask
is optimized on one side while the partner embedding is held fixed, and the
score being preserved is the same baseline or ABTT cosine used by retrieval.
This makes MaRC applicable to pairwise retrieval without retraining the encoder.
```

For the table caption or limitations:

```latex
Cross-variant attribution comparisons are descriptive because ABTT changes both
the embedding geometry and the scalar cosine function being explained. Lower
MinFrac values indicate more compact rationales, but learned-mask methods have
their own sparsity pressure, so MinFrac should not be used alone to compare IG
and MaRC.
```

## Caveats That Must Stay in Main Prose

- Intrinsic geometry diagnoses collapse and ABTT recovery, but it is not a standalone best-layer selector.
- The main attribution layers were selected by a train-only retrieval rule before Run 3 attribution metrics, not by test attribution performance.
- The active attribution evidence is the three-model 200-positive bundle; older 20-pair, 200-random, LaTa-L4, or PhilTa-L6 headline results are superseded unless explicitly labeled historical or diagnostic.
- `rho_LOO` is the only metric that improves consistently across all IG and MaRC cells.
- ERASER-style Sufficiency and Comprehensiveness are mixed, and MinFrac@0.80 worsens in all six IG/MaRC cells.
- ABTT changes the decision function being explained: baseline attribution explains raw cosine, while ABTT attribution explains ABTT-corrected cosine.
- Ratio metrics depend on the full-cosine denominator policy; captions should keep the cosine floor visible.
- MaRC's retrieval adaptation is our contribution, but MaRC has built-in mask sparsity pressure, so compactness-style comparisons against IG need careful wording.
- These are local query-candidate explanations, not global explanations of model behavior or human scholarly decisions.

## Claims To Avoid

- "ABTT uniformly improves attribution quality."
- "Geometry diagnostics choose the best layer."
- "The attribution layer was selected from the test attribution results."
- "MaRC proves the empirical ABTT attribution claim by itself."
- "Compactness shows MaRC is better than IG."
- "The 200-random or 20-pair attribution results are the current headline evidence."
- "ABTT makes rationales more compact."
- "The diagnostic collapse layers are the operational retrieval layers."
