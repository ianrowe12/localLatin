# Run 5 Paper Rewrite Brief

Generated: 2026-05-01

Purpose: prepare the next paper-rewrite chat with patch-ready prose blocks and
section-level instructions. This is not a direct rewrite of
`overleaf_drafts/acl_latex.tex`; it maps the current evidence to main-text and
appendix edits so the rewrite can patch the LaTeX deliberately.

## Source Of Truth

Use the active Run 3 and Run 4 artifacts, not older meeting or visualization
bundles:

| Evidence role | Current source |
|---|---|
| Layer geometry and layer-selection analysis | `docs/analyses/layer_geometry_diagnostics.md`; `runs/active/resubmit/layer_diagnostics/` |
| Locked attribution layers | `docs/runs/run2_attribution_layer_contract.md`; `scripts/ig/attribution_model_config.py` |
| Three-model attribution bundle | `docs/runs/run3_agent1_three_model_artifacts.md`; `runs/active/ig_examples_200pos_run3_operational/` |
| Main attribution table and figure | `docs/runs/run4_main_attribution_artifacts.md`; `overleaf_drafts/tables/attribution_metrics_main.tex`; `overleaf_drafts/figures/fig_attribution_rho_loo_main.tex` |
| Appendix attribution sweep package | `docs/runs/run4_agent2_appendix_sweeps.md`; `overleaf_drafts/tables/attribution_metrics_sweep_main_methods.tex`; `overleaf_drafts/tables/attribution_metrics_sweep_supplemental_methods.tex` |
| Attribution method and metric provenance | `docs/research/run1_metric_provenance.md` |
| Interpretation and caveats | `docs/runs/run4_interpretation_caveat_memo.md` |

Do not promote these superseded artifacts as current headline evidence:

- `runs/active/ig_examples/` 20-pair or 80-pair visualization metrics.
- `runs/active/ig_examples_200pos/` two-model meeting bundle.
- Professor-share LaTa L4 / PhilTa L6 attribution framing except as historical
  or diagnostic context.
- Any 200-random-pair attribution result.

## Rewrite Thesis

Main text should make one coherent argument:

1. Intermediate layers matter because they reveal a geometric collapse that
   final-layer-only reporting hides.
2. ABTT repairs that collapse by removing train-fit top principal components
   after pooling.
3. Geometry diagnostics motivate and explain ABTT, but they are not a
   standalone best-layer selector.
4. Operational retrieval and attribution layers are selected by a predeclared
   train-only retrieval rule.
5. Attribution evaluates the same retrieval score under two views: integrated
   gradients and retrieval-adapted MaRC.
6. The main attribution result is narrow: ABTT improves leave-one-out rank
   faithfulness across all LaTa, PhilTa, and mT5 model-view cells, while
   ERASER-style metrics are mixed.

## Main Text Versus Appendix

Put in main text:

- Why layerwise analysis is necessary.
- The two-stage layer guidance: diagnostic geometry first, train-only retrieval
  layer selection second.
- The ABTT repair story for the three headline models.
- The locked operational attribution layers: LaTa L7, PhilTa L1, mT5-base L1.
- The attribution setup, with IG and retrieval-adapted MaRC treated as parallel
  views of the same retrieval cosine.
- The metric provenance in compressed form, especially which metrics are
  adapted from classification rationale evaluation.
- The main attribution table or figure showing `rho_LOO` improvement in 6/6
  model-method cells, with explicit caveats about mixed ERASER metrics.
- Caveats that prevent overclaiming.

Put in appendix:

- Full per-layer diagnostics, including main and non-headline models.
- The recovered-collapse diagnostic layers: LaTa L8, PhilTa L6, mT5-base L5.
- Correlation summaries linking PC1 dominance/effective-rank collapse to ABTT
  gains.
- Full attribution metric sweep: Sufficiency/Comprehensiveness at 10%, 25%,
  and 50%; MinFrac threshold sweep when complete.
- Supplemental attribution methods and baselines beyond IG and retrieval-adapted
  MaRC.
- Historical or diagnostic attribution examples from old-layer meeting bundles,
  only if clearly labelled as not the headline experiment.

## Motivation For Layerwise Analysis

Target location: Introduction and start of Results.

Instruction: replace any wording that suggests "we inspect layers because
layer choice matters" with the stronger geometric motivation. The key evidence
is that intermediate layers expose representation collapse: top-PC dominance,
low effective rank, and cosine concentration. This motivates ABTT as a repair,
not merely as another post-processing baseline.

Patch-ready prose:

```latex
We inspect intermediate layers because the relevant failure mode is geometric
rather than monotonic with depth. In the headline T5-family models, several
middle layers collapse into a low-rank cone: cosine scores concentrate,
effective rank drops, and a single principal component can dominate the
embedding variance. A final-layer-only evaluation would hide this failure mode.
Layerwise analysis therefore lets us distinguish a model that lacks retrieval
signal from a representation whose signal is present but distorted by
anisotropic geometry.
```

Patch-ready contribution wording:

```latex
We provide a layerwise geometric analysis showing that retrieval failures in
Latin T5 representations are driven in part by intermediate-layer anisotropy,
and we show that a simple train-free post-processing step can repair much of
that collapse without updating model parameters.
```

Evidence to cite in prose or caption:

- `docs/analyses/layer_geometry_diagnostics.md`
- `runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv`
- `runs/active/resubmit/layer_diagnostics/geometry_retrieval_join_main.csv`

Appendix instruction: expose the full layer diagnostic table rather than only
the three headline layers, so readers can see that collapse/recovery is a depth
profile and not a hand-picked snapshot.

## Layer-Selection Guidance

Target location: Methods after representation extraction, and Results before
the attribution subsection.

Instruction: state the two-stage rule. Geometry is label-free diagnostic
evidence; operational retrieval and attribution layers are selected by a
predeclared train-only retrieval criterion. Do not claim that intrinsic
geometry selects the best layer.

Patch-ready prose:

```latex
We use layer geometry diagnostically rather than as a best-layer oracle. First,
we inspect label-free geometry statistics--top-PC dominance, effective rank,
and cosine concentration--to identify collapsed layers and to test whether
ABTT repairs them. Second, for operational retrieval and attribution, we choose
the earliest layer within 0.5 percentage points of the best training-set
directory accuracy at rank 1 under ABTT. This rule is fixed before computing
the attribution metrics and uses no test attribution outcome.
```

Patch-ready table note:

```latex
Attribution layers are the retrieval-selected operational layers: LaTa layer 7,
PhilTa layer 1, and mT5-base layer 1. Diagnostic collapse layers are reported
separately as mechanism checks: LaTa layer 8, PhilTa layer 6, and mT5-base
layer 5.
```

Evidence:

- `docs/runs/run2_attribution_layer_contract.md`
- `runs/active/resubmit/layer_diagnostics/layer_rule_candidates.csv`

Main text should include only the headline operational layers and one sentence
about the diagnostic collapse layers. Appendix should carry the layer-rule
candidate table and the recovered-collapse layers for all available models.

## ABTT Repair Story

Target location: Methods ABTT subsection and main Results layerwise section.

Instruction: keep ABTT provenance clear. ABTT is borrowed from prior work; the
paper's contribution is applying and diagnosing it in this retrieval pipeline.
Use train-split fitting language whenever ABTT is described.

Patch-ready methods prose:

```latex
We use All-but-the-Top (ABTT) post-processing
\citep{mu2018allbutthetop} as a train-free geometric correction after pooling.
For each model and layer, we center the pooled training embeddings, estimate
the leading principal components on the training split, and remove the selected
components from both train and test embeddings. ABTT itself is not new; here it
serves as a retrieval-specific repair for anisotropic Latin manuscript
representations.
```

Patch-ready results prose:

```latex
At the diagnostic collapse layers, ABTT-D10 sharply reverses the low-rank
geometry. For LaTa, PhilTa, and mT5-base, the strongest-collapse layers have
PC1 variance shares of 0.956, 0.862, and 1.000 before correction; after
ABTT-D10 these fall to 0.036, 0.039, and 0.044. Effective rank rises from
1.34, 1.83, and 1.00 to 168.40, 155.02, and 151.74. This supports the repair
interpretation: ABTT does not add supervision or retrain the encoder, but it
removes a dominant global direction that was suppressing cosine separation.
```

Evidence:

- `docs/analyses/layer_geometry_diagnostics.md`
- `runs/active/resubmit/layer_diagnostics/geometry_retrieval_join_main.csv`

Main text should use the three-model numbers above. Appendix should include the
same diagnostics for LaBSE, Qwen3-0.6B, and KaLM-mini, plus the correlation
summary. The safe correlation claim is: PC1 dominance and low effective rank
are strongly associated with ABTT gains over baseline in the main-model layer
grid, but this association is explanatory rather than a standalone selector.

## Attribution Setup

Target location: Methods attribution subsection and Results attribution
subsection.

Instruction: update the current attribution framing from example-only IG
toward quantitative two-view attribution. Treat IG and retrieval-adapted MaRC
as equal paper-facing views. The current main text says quantitative
faithfulness comparisons are "accompanying work"; that should be replaced by
the Run 3/Run 4 quantitative evidence.

Patch-ready prose:

```latex
We evaluate local explanations for a fixed query-candidate retrieval score
using two complementary attribution views. Integrated gradients provides a
standard gradient-based attribution of the pairwise cosine. Retrieval-adapted
MaRC provides a learned-mask view: for each pair, a soft mask is optimized on
one side while the partner representation is held fixed, and the preserved
scalar is the same baseline or ABTT cosine used for retrieval. Both views are
computed at the predeclared operational attribution layers.
```

Patch-ready setup sentence for the table:

```latex
The attribution experiment uses 200 positive query-candidate pairs per model
for LaTa, PhilTa, and mT5-base, with baseline and ABTT variants for both
integrated gradients and retrieval-adapted MaRC.
```

Evidence:

- `runs/active/ig_examples_200pos_run3_operational/manifest.json`
- `runs/active/ig_examples_200pos_run3_operational/artifact_inventory.csv`
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv`

Main text should include the setup and headline table. Appendix can include
artifact completeness, supplemental methods, and example visualizations.

## MaRC Retrieval Adaptation

Target location: Methods attribution subsection.

Instruction: make clear which part is MaRC and which part is this paper's
adaptation. Avoid saying "we use MaRC" without qualification. Use
"retrieval-adapted MaRC" or "MaRC-style learned mask adapted to retrieval."

Patch-ready prose:

```latex
Our learned-mask attribution is adapted from MaRC
\citep{brinnerzarriess2023marc}, which optimizes a soft input mask for a frozen
classifier so that the masked input preserves the target class score while
regularizers encourage sparse and smooth rationales. We keep the per-instance
mask-optimization idea, but replace the classification target with pairwise
retrieval cosine. Given a query-candidate pair, the mask is optimized so that
the masked query representation preserves $\cos(E_v(q),E_v(c))$ with the
candidate representation held fixed; candidate-side attribution is obtained by
swapping the two sides. This adapts MaRC's objective to bi-encoder retrieval
without retraining the encoder.
```

Patch-ready equation lead-in:

```latex
Here $v$ denotes the embedding variant being explained, either raw mean-pooled
cosine or the ABTT-corrected cosine. The attribution target is therefore the
same scalar function used by retrieval, not a classifier logit.
```

Evidence:

- `docs/research/run1_metric_provenance.md`
- `src/direct_logit_attribution.py`
- `src/attribution_metrics.py`
- `runs/active/ig_examples_200pos_run3_operational/retrieval_mark/` when
  artifact directories are present.

Main text should describe the adaptation and cite MaRC. Appendix can include
optimization hyperparameters, masking constraints, and examples.

## Metric Provenance

Target location: Methods metrics paragraph, main attribution table caption, and
bibliography patch.

Instruction: separate method provenance from metric provenance. The rewrite
should say IG and ABTT are borrowed, MaRC is adapted, and the retrieval-cosine
target plus retrieval metric adaptation are ours.

Patch-ready metrics prose:

```latex
For faithfulness, we adapt rationale-evaluation metrics to the retrieval
cosine. Sufficiency keeps the top-ranked query tokens and measures how much of
the full pair cosine remains; comprehensiveness removes those tokens and
measures the resulting drop. We also report the smallest retained token
fraction needed to recover 80\% of the full cosine, labelled MinFrac@0.80.
As a perturbation-based ranking check, $\rho_{\mathrm{LOO}}$ is the Spearman
correlation between attribution magnitude and the leave-one-out drop in the
same retrieval score.
```

Patch-ready citation/provenance prose:

```latex
Integrated gradients is borrowed from \citet{sundararajan2017axiomatic};
ABTT is borrowed from \citet{mu2018allbutthetop}; and the learned-mask view is
adapted from MaRC \citep{brinnerzarriess2023marc}. The rationale metrics follow
the ERASER-family sufficiency and comprehensiveness framing
\citep{deyoung2020eraser}, with the classification score replaced by retrieval
cosine.
```

Bibliography instruction:

- Confirm `brinnerzarriess2023marc` has DOI `10.18653/v1/2023.findings-acl.867`
  and ACL Anthology URL.
- Add `deyoung2020eraser` if the final paper uses Sufficiency and
  Comprehensiveness provenance.
- Add `jain2019attention` only if the leave-one-out rank-correlation paragraph
  needs a broader saliency/perturbation citation. Do not overstate it as the
  exact source of this metric.

Main text should define metric directions in the table caption: higher is
better for `rho_LOO`, Sufficiency, and Comprehensiveness; lower is better for
MinFrac. Appendix should include threshold grids and any denominator/floor
policy.

## Main Attribution Result

Target location: Results attribution subsection and possibly a short paragraph
in Discussion.

Instruction: the main attribution claim should be about `rho_LOO`, not a broad
"ABTT improves attribution quality" claim. ERASER-style metrics must remain
visible as mixed evidence.

Patch-ready prose:

```latex
The clearest quantitative attribution result is rank faithfulness. Across the
three headline models and both attribution views, ABTT improves
$\rho_{\mathrm{LOO}}$ in all six model-method cells: LaTa IG
$-0.001\rightarrow0.396$, LaTa MaRC $0.042\rightarrow0.401$, PhilTa IG
$0.182\rightarrow0.607$, PhilTa MaRC $0.190\rightarrow0.331$, mT5-base IG
$0.164\rightarrow0.597$, and mT5-base MaRC $0.250\rightarrow0.392$. Thus,
tokens assigned higher attribution under the ABTT-corrected score better align
with leave-one-out drops in that same score.
```

Patch-ready caveat continuation:

```latex
The ERASER-style metrics qualify this result. At the global headline
thresholds, ABTT improves only 6 of 18 Sufficiency, Comprehensiveness, and
MinFrac comparisons. Sufficiency and Comprehensiveness improve for some
model-view pairs, especially mT5 and LaTa with MaRC, while MinFrac@0.80
worsens in all six cells. We therefore interpret attribution as supporting the
ABTT repair story through rank faithfulness, with threshold-dependent tradeoffs
rather than uniform improvement across every rationale metric.
```

Required LaTeX inclusions:

```latex
\input{tables/attribution_metrics_main}
```

and optionally:

```latex
\input{figures/fig_attribution_rho_loo_main}
```

Evidence:

- `overleaf_drafts/tables/attribution_metrics_main.tex`
- `overleaf_drafts/figures/fig_attribution_rho_loo_main.tex`
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv`

Main text should lead with `rho_LOO` and show the mixed metrics in the same
table or adjacent caption. Appendix should show the sweep and supplemental
methods.

## Appendix Support

Target location: new or revised appendix sections after existing per-layer and
SIF appendices.

Instruction: add appendices that make the headline choices auditable without
overloading main text.

Patch-ready appendix outline:

```latex
\section{Layer Diagnostics And Attribution Layer Selection}
\label{app:layer_diagnostics}

This appendix reports the full layerwise geometry diagnostics used to motivate
ABTT and the predeclared retrieval-layer rule used for attribution. The
diagnostic collapse layers identify where geometry is most anisotropic; the
operational attribution layers are selected separately by training-set
retrieval accuracy.

\section{Attribution Metric Sweeps}
\label{app:attribution_sweeps}

This appendix expands the main attribution table across Sufficiency and
Comprehensiveness thresholds of 10\%, 25\%, and 50\%, and reports available
MinFrac recovery thresholds. The main text uses a single global midpoint
threshold rather than selecting model-specific or method-specific thresholds.
```

Required or candidate appendix inclusions:

```latex
\input{tables/attribution_metrics_sweep_main_methods}
\input{tables/attribution_metrics_sweep_supplemental_methods}
```

Appendix caveat:

```latex
The current checked-in Run 3 summary contains MinFrac@0.80 but not the full
0.70/0.90/0.95 recovery-threshold grid. Missing threshold cells are marked as
missing rather than interpolated. A rerun of the metric stage from cached NPZ
artifacts is required before claiming the full MinFrac sweep.
```

Evidence:

- `docs/runs/run4_agent2_appendix_sweeps.md`
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_sweep_long.csv`
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/appendix_sweep_completeness.json`

## Caveats To Keep In Main Text

Use short versions of these caveats in main prose, table captions, or
limitations:

- Intrinsic geometry diagnoses collapse and ABTT recovery, but it is not a
  standalone best-layer selector.
- The operational attribution layers were selected by a train-only retrieval
  rule before Run 3 attribution metrics were computed.
- ABTT changes the scalar function being explained: baseline attribution
  explains raw cosine, while ABTT attribution explains ABTT-corrected cosine.
- `rho_LOO` is the only metric that improves consistently across all six
  headline IG/MaRC cells.
- Sufficiency, Comprehensiveness, and MinFrac show mixed effects; ABTT does not
  uniformly improve attribution quality.
- MinFrac should not be called MaRC compactness. It is a rank-based retained
  fraction needed to recover a cosine threshold.
- MaRC has built-in sparsity pressure, so compactness-style comparisons between
  MaRC and IG are descriptive rather than proof that one method is globally
  better.
- Attribution is local to a query-candidate score, not a global explanation of
  model behavior or a substitute for scholarly judgment.

Patch-ready limitations prose:

```latex
These attribution results are local diagnostics for fixed query-candidate
pairs. They do not provide a global explanation of model behavior, and
cross-variant comparisons are descriptive because ABTT changes both the
embedding geometry and the cosine function being explained. The consistent
effect is improved leave-one-out rank faithfulness; other rationale metrics
remain threshold-dependent and should not be summarized as a uniform
attribution-quality gain.
```

## Claims To Avoid

Do not write:

- "ABTT uniformly improves attribution quality."
- "Geometry diagnostics choose the best layer."
- "The attribution layer was selected from test attribution results."
- "MaRC proves the ABTT explanation result by itself."
- "Compactness shows MaRC is better than IG."
- "ABTT makes rationales more compact."
- "Diagnostic collapse layers are the operational retrieval layers."
- "The 20-pair, 80-pair, 200-random, LaTa-L4, or PhilTa-L6 artifacts are the
  current headline attribution evidence."

## Suggested Rewrite Order

1. Update Introduction and contribution bullets to make the layerwise geometric
   failure mode and ABTT repair the central story.
2. Patch Methods with the two-stage layer rule, train-split ABTT fitting, IG,
   retrieval-adapted MaRC, and metric provenance.
3. Patch Results so Task A/Task B layerwise retrieval leads naturally into the
   operational attribution layers.
4. Replace the current attribution-results language with the Run 4 main table,
   optional `rho_LOO` figure, and the narrow `rho_LOO` claim.
5. Add appendix sections for layer diagnostics and attribution sweeps.
6. Update limitations to keep the caveats above visible.
