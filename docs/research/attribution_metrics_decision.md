# Attribution Metric Expansion and Selection

Generated: 2026-09-06. Issue #124, part of epic #109. Decision 5 of
`docs/research/plan_20260906.md`.

**Honesty rule (Ian, 2026-09-06): nothing is deleted. Every metric computed here
appears in this memo with its full result. Metrics that do not make the main
table go to the appendix with one plain sentence saying why.**

This memo does two things, in this order and deliberately not the other way
round. Part A reports every attribution-quality metric we now have, old and new,
with the baseline-versus-ABTT outcome in every cell. Part B states the selection
criteria for the main table, then applies them. The criteria are written down
before the selection is made, and one of them is explicitly a validity criterion
that can disqualify a metric that favours ABTT.

The paper `.tex` is not touched here. Issue #120 regenerates the tables and
prose from this memo.

## Provenance

| Item | Value |
|---|---|
| Artifacts | `runs/active/ig_examples_200pos_run3_operational/artifacts/` (600 NPZs) |
| Models | LaTa (layer 7), PhilTa (layer 1), mT5-base (layer 1) |
| Pairs | 200 positive query-candidate pairs per model |
| Views | IG, retrieval-adapted MaRC (`retrieval_mark`) |
| Variants | `baseline`, `abtt` (D = 10) |
| New summary | `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv` |
| Long form | `.../attribution_metrics/summary_v2_sweep_long.csv` |
| Wins table | `.../attribution_metrics/metric_wins_v2.csv` |
| Per-pair cache | `.../attribution_metrics/v2_hidden/<slug>/*.json` (gitignored, regenerable) |
| Prior summary | `.../attribution_metrics/summary.csv` (unchanged, still the source of the currently published table) |
| Compute | CPU only, no GPU hours spent |

Reproduce with:

```bash
python scripts/ig/run_attribution_metrics.py --backend hidden \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --out_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --out_subdir v2_hidden \
  --summary_out runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv \
  --sweep_summary_out runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2_sweep_long.csv \
  --tex_out "" --require_artifacts

python scripts/ig/build_attribution_metric_decision_tables.py \
  --wins_csv runs/active/ig_examples_200pos_run3_operational/attribution_metrics/metric_wins_v2.csv
```

## What was added

Five registry functions in `src/attribution_metrics.py`, one per metric.

| Metric | Definition on our decision scalar | Direction | Source |
|---|---|---|---|
| AOPC-Suff | mean over k = 1..n of S_v(top-k kept) / S_v(full) | higher | DeYoung et al. 2020 (ERASER), AOPC from Samek et al. 2017 |
| AOPC-Comp | mean over k = 1..n of [S_v(full) - S_v(top-k removed)] / S_v(full) | higher | same |
| DelAUC | area under S_v(remaining)/S_v(full) as tokens are removed most-important-first | lower | Petsiuk et al. 2018 (RISE) |
| InsAUC | area under S_v(kept)/S_v(full) as tokens are inserted most-important-first | higher | same |
| DelAUC gap, InsAUC gap | the same AUC minus (or minus-from) a 5-draw random-order reference | higher | same |
| tau_LOO | Kendall tau-b between abs(a) and the per-token LOO delta | higher | tie-corrected companion to the existing Spearman rho_LOO |
| Rand gap | real metric minus the mean over 5 permutations of the attribution vector | higher | control in the spirit of Adebayo et al. 2018 |

Two notes on what these are and are not.

**The AUC gap, not the raw AUC, is the part that measures the ranking.** The
height of a deletion or insertion curve is dominated by how redundant a query
is: a 250-token fragment keeps most of its cosine no matter which tokens you
drop. Scoring the same pair under 5 uniformly random orderings and reporting the
difference is what isolates the attribution from the query. All methods within a
pair are scored against the same reference orderings, so the comparison is
paired.

**The randomization check is the score-permutation control, not Adebayo's
parameter randomization.** Adebayo et al. randomize model weights and labels;
we permute the attribution vector, which preserves its distribution exactly and
destroys only the token-to-score assignment. It answers the narrower question
that matters here: on this data, does a metric separate a real attribution from
a fake one with the same score distribution? A metric that cannot is
uninformative regardless of which variant it favours. We will describe it in the
paper as a shuffled-attribution control, not as a sanity check in Adebayo's
sense.

## Method note: how the masked cosine is computed

Every metric here is built on S_v(masked query, candidate). There are two ways
to erase a token and they do not give the same number.

* **Input-level erasure** (the pre-existing `--backend model` path): replace the
  token with PAD, zero its attention, and re-run the encoder. The surviving
  tokens are re-contextualised, so the drop mixes "this token mattered" with
  "removing this token changed the other tokens".
* **Representation-level erasure** (the new `--backend hidden` path): drop the
  token's row from the mean pool over the cached layer-L hidden states. Nothing
  is re-contextualised.

This run uses representation-level erasure, for two reasons. It is CPU-only,
which is what let us afford the full k = 1..n curve that AOPC and the AUCs need;
and it is exact for the unmasked cosine. The largest mean deviation between our
recomputed full cosine and the `cos_orig_*` value stored in the NPZ at build
time, across all 3 models x 2 variants, is 2.5e-07 (section A4). That is float32
noise, which is what pins both backends to the same decision scalar.

The consequence is that `summary_v2.csv` is internally consistent (one operator
for every column, old metrics included) but is **not** row-for-row comparable
with the published `summary.csv`. That is why the old metrics were recomputed
rather than copied across. The size of the difference is quantified in A4.

---

# Part A. Full results, every metric

All values are means over 200 pairs. Cells read `baseline -> ABTT`; **A** marks
an ABTT win in that metric's better direction, `b` a baseline win.

| Metric | Dir | LaTa/IG | LaTa/MaRC | PhilTa/IG | PhilTa/MaRC | mT5-base/IG | mT5-base/MaRC |
|---|---|---|---|---|---|---|---|
| rho_LOO | up | 0.013 -> 0.298 **A** | 0.070 -> 0.397 **A** | 0.144 -> 0.577 **A** | 0.179 -> 0.367 **A** | 0.138 -> 0.626 **A** | 0.272 -> 0.415 **A** |
| Suff@10% | up | 0.949 -> 0.483 b | 0.419 -> 0.723 **A** | 0.946 -> 0.832 b | 0.913 -> 0.671 b | 0.788 -> 0.807 **A** | 0.907 -> 0.660 b |
| Suff@25% | up | 0.965 -> 0.729 b | 0.556 -> 0.847 **A** | 0.965 -> 0.941 b | 0.952 -> 0.883 b | 0.885 -> 1.007 **A** | 0.936 -> 0.840 b |
| Suff@50% | up | 0.970 -> 0.923 b | 0.603 -> 0.931 **A** | 0.983 -> 1.030 **A** | 0.953 -> 0.997 **A** | 0.937 -> 1.148 **A** | 0.963 -> 1.017 **A** |
| Comp@10% | up | 0.304 -> 0.076 b | 0.114 -> 0.107 b | 0.702 -> 0.301 b | 0.649 -> 0.199 b | 0.013 -> 0.328 **A** | 0.011 -> 0.284 **A** |
| Comp@25% | up | 0.658 -> 0.194 b | 0.329 -> 0.272 b | 0.988 -> 0.546 b | 0.951 -> 0.443 b | 0.055 -> 0.653 **A** | 0.047 -> 0.591 **A** |
| Comp@50% | up | 0.802 -> 0.372 b | 0.466 -> 0.519 **A** | 1.026 -> 0.783 b | 0.971 -> 0.692 b | 0.275 -> 0.999 **A** | 0.139 -> 0.880 **A** |
| MinFrac@0.70 | down | 0.040 -> 0.243 b | 0.375 -> 0.145 **A** | 0.028 -> 0.088 b | 0.058 -> 0.216 b | 0.067 -> 0.172 b | 0.015 -> 0.261 b |
| MinFrac@0.80 | down | 0.042 -> 0.330 b | 0.398 -> 0.291 **A** | 0.031 -> 0.144 b | 0.058 -> 0.296 b | 0.134 -> 0.232 b | 0.032 -> 0.340 b |
| MinFrac@0.90 | down | 0.045 -> 0.456 b | 0.423 -> 0.447 b | 0.045 -> 0.242 b | 0.060 -> 0.386 b | 0.342 -> 0.317 **A** | 0.094 -> 0.441 b |
| MinFrac@0.95 | down | 0.057 -> 0.539 b | 0.455 -> 0.539 b | 0.084 -> 0.319 b | 0.075 -> 0.451 b | 0.605 -> 0.368 **A** | 0.231 -> 0.500 b |
| tau_LOO | up | 0.007 -> 0.212 **A** | 0.059 -> 0.292 **A** | 0.106 -> 0.432 **A** | 0.133 -> 0.258 **A** | 0.102 -> 0.465 **A** | 0.186 -> 0.291 **A** |
| AOPC-Suff | up | 0.972 -> 0.833 b | 0.617 -> 0.898 **A** | 0.979 -> 0.970 b | 0.964 -> 0.926 b | 0.908 -> 1.040 **A** | 0.953 -> 0.930 b |
| AOPC-Comp | up | 0.718 -> 0.395 b | 0.458 -> 0.504 **A** | 0.939 -> 0.711 b | 0.891 -> 0.643 b | 0.472 -> 0.882 **A** | 0.153 -> 0.766 **A** |
| DelAUC | down | 0.291 -> 0.614 b | 0.550 -> 0.505 **A** | 0.069 -> 0.297 b | 0.117 -> 0.365 b | 0.534 -> 0.124 **A** | 0.853 -> 0.241 **A** |
| DelAUC gap | up | 0.504 -> 0.178 b | 0.244 -> 0.287 **A** | 0.806 -> 0.420 b | 0.758 -> 0.353 b | 0.394 -> 0.548 **A** | 0.075 -> 0.431 **A** |
| InsAUC | up | 0.963 -> 0.824 b | 0.609 -> 0.889 **A** | 0.971 -> 0.962 b | 0.956 -> 0.918 b | 0.901 -> 1.034 **A** | 0.946 -> 0.924 b |
| InsAUC gap | up | 0.153 -> 0.032 b | -0.201 -> 0.098 **A** | 0.118 -> 0.244 **A** | 0.104 -> 0.200 **A** | -0.028 -> 0.353 **A** | 0.017 -> 0.243 **A** |
| Rand gap (rho) | up | 0.015 -> 0.297 **A** | 0.072 -> 0.391 **A** | 0.139 -> 0.570 **A** | 0.174 -> 0.361 **A** | 0.136 -> 0.630 **A** | 0.271 -> 0.416 **A** |
| Rand gap (tau) | up | 0.008 -> 0.211 **A** | 0.060 -> 0.288 **A** | 0.103 -> 0.427 **A** | 0.129 -> 0.254 **A** | 0.102 -> 0.467 **A** | 0.185 -> 0.292 **A** |
| Rand gap (AOPC-S) | up | 0.172 -> 0.035 b | -0.189 -> 0.097 **A** | 0.116 -> 0.238 **A** | 0.107 -> 0.199 **A** | -0.030 -> 0.354 **A** | 0.019 -> 0.253 **A** |
| Rand gap (AOPC-C) | up | 0.509 -> 0.179 b | 0.240 -> 0.285 **A** | 0.795 -> 0.418 b | 0.752 -> 0.353 b | 0.394 -> 0.551 **A** | 0.075 -> 0.444 **A** |

## A1. Wins per metric

| Metric | Dir | Family | LaTa/IG | LaTa/MaRC | PhilTa/IG | PhilTa/MaRC | mT5-base/IG | mT5-base/MaRC | ABTT wins |
|---|---|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| rho_LOO | higher | existing | A | A | A | A | A | A | 6/6 |
| Suff@10% | higher | existing | b | A | b | b | A | b | 2/6 |
| Suff@25% | higher | existing | b | A | b | b | A | b | 2/6 |
| Suff@50% | higher | existing | b | A | A | A | A | A | 5/6 |
| Comp@10% | higher | existing | b | b | b | b | A | A | 2/6 |
| Comp@25% | higher | existing | b | b | b | b | A | A | 2/6 |
| Comp@50% | higher | existing | b | A | b | b | A | A | 3/6 |
| MinFrac@0.70 | lower | existing | b | A | b | b | b | b | 1/6 |
| MinFrac@0.80 | lower | existing | b | A | b | b | b | b | 1/6 |
| MinFrac@0.90 | lower | existing | b | b | b | b | A | b | 1/6 |
| MinFrac@0.95 | lower | existing | b | b | b | b | A | b | 1/6 |
| tau_LOO | higher | new | A | A | A | A | A | A | 6/6 |
| AOPC-Suff | higher | new | b | A | b | b | A | b | 2/6 |
| AOPC-Comp | higher | new | b | A | b | b | A | A | 3/6 |
| DelAUC | lower | new | b | A | b | b | A | A | 3/6 |
| DelAUC gap | higher | new | b | A | b | b | A | A | 3/6 |
| InsAUC | higher | new | b | A | b | b | A | b | 2/6 |
| InsAUC gap | higher | new | b | A | A | A | A | A | 5/6 |
| Rand gap (rho) | higher | new | A | A | A | A | A | A | 6/6 |
| Rand gap (tau) | higher | new | A | A | A | A | A | A | 6/6 |
| Rand gap (AOPC-S) | higher | new | b | A | A | A | A | A | 5/6 |
| Rand gap (AOPC-C) | higher | new | b | A | b | b | A | A | 3/6 |

Machine-readable copy: `metric_wins_v2.csv`.

## A2. Controls

The `random` row is 5 seeds of uniform per-token scores; the `inverse` row uses
`1 / (eps + abs(IG))`, so its ranking is the reverse of IG's. A metric is only
interpretable if `random` sits at its no-information value and `inverse` sits
below it.

| Model | Row | Variant | rho_LOO | tau_LOO | AOPC-Suff | AOPC-Comp | DelAUC gap | InsAUC gap |
|---|---|---|--:|--:|--:|--:|--:|--:|
| LaTa | IG | baseline | 0.013 | 0.007 | 0.972 | 0.718 | 0.504 | 0.153 |
| LaTa | IG | abtt | 0.298 | 0.212 | 0.833 | 0.395 | 0.178 | 0.032 |
| LaTa | MaRC | baseline | 0.070 | 0.059 | 0.617 | 0.458 | 0.244 | -0.201 |
| LaTa | MaRC | abtt | 0.397 | 0.292 | 0.898 | 0.504 | 0.287 | 0.098 |
| LaTa | random | baseline | -0.002 | -0.001 | 0.810 | 0.218 | 0.004 | -0.009 |
| LaTa | random | abtt | -0.008 | -0.006 | 0.800 | 0.217 | 0.000 | -0.000 |
| LaTa | inverse | baseline | -0.013 | -0.007 | 0.299 | 0.045 | -0.169 | -0.520 |
| LaTa | inverse | abtt | -0.298 | -0.212 | 0.622 | 0.184 | -0.033 | -0.178 |
| PhilTa | IG | baseline | 0.144 | 0.106 | 0.979 | 0.939 | 0.806 | 0.118 |
| PhilTa | IG | abtt | 0.577 | 0.432 | 0.970 | 0.711 | 0.420 | 0.244 |
| PhilTa | MaRC | baseline | 0.179 | 0.133 | 0.964 | 0.891 | 0.758 | 0.104 |
| PhilTa | MaRC | abtt | 0.367 | 0.258 | 0.926 | 0.643 | 0.353 | 0.200 |
| PhilTa | random | baseline | 0.001 | 0.000 | 0.880 | 0.152 | 0.019 | 0.019 |
| PhilTa | random | abtt | -0.008 | -0.006 | 0.724 | 0.287 | -0.003 | -0.002 |
| PhilTa | inverse | baseline | -0.144 | -0.106 | 0.077 | 0.037 | -0.096 | -0.784 |
| PhilTa | inverse | abtt | -0.577 | -0.432 | 0.305 | 0.046 | -0.245 | -0.421 |
| mT5-base | IG | baseline | 0.138 | 0.102 | 0.908 | 0.472 | 0.394 | -0.028 |
| mT5-base | IG | abtt | 0.626 | 0.465 | 1.040 | 0.882 | 0.548 | 0.353 |
| mT5-base | MaRC | baseline | 0.272 | 0.186 | 0.953 | 0.153 | 0.075 | 0.017 |
| mT5-base | MaRC | abtt | 0.415 | 0.291 | 0.930 | 0.766 | 0.431 | 0.243 |
| mT5-base | random | baseline | -0.000 | -0.000 | 0.938 | 0.078 | 0.000 | 0.002 |
| mT5-base | random | abtt | -0.005 | -0.004 | 0.687 | 0.332 | -0.002 | 0.000 |
| mT5-base | inverse | baseline | -0.138 | -0.102 | 0.540 | 0.105 | 0.027 | -0.395 |
| mT5-base | inverse | abtt | -0.626 | -0.465 | 0.130 | -0.028 | -0.362 | -0.557 |

Reading of the control table:

* `random` sits within 0.02 of zero for rho_LOO, tau_LOO, DelAUC gap and InsAUC
  gap in all six cells (largest deviation 0.019, PhilTa baseline). Those four
  metrics have a calibrated zero.
* `random` for AOPC-Suff runs from 0.687 to 0.938 and for AOPC-Comp from 0.078
  to 0.332, depending on model and variant. Those two have no fixed
  no-information value, so an absolute AOPC number cannot be compared across
  models or across variants without also quoting its floor.
* `inverse` is below `random` on rho_LOO and tau_LOO in all six cells, and on
  both AUC gaps in five of six. The exception is mT5-base baseline, where the
  inverse DelAUC gap is +0.027 against a random floor of 0.000: at that layer
  the baseline cosine is so redundant that removing the *least* IG-salient
  tokens first still costs slightly more than removing tokens at random.
  `inverse` is also below `random` on both AOPC halves in all six cells. Nothing
  is inverted.

## A3. Shuffled-attribution control

Gap = real minus the mean of 5 permutations of the same attribution vector.
Positive means the real attribution beats a fake one drawn from its own score
distribution. Twelve cells: 3 models x 2 views x 2 variants.

| Metric | Cells with a positive gap | Range |
|---|---|---|
| rho_LOO | 12/12 | +0.015 to +0.630 |
| tau_LOO | 12/12 | +0.008 to +0.467 |
| AOPC-Comp | 12/12 | +0.075 to +0.795 |
| AOPC-Suff | **10/12** | -0.189 to +0.354 |

AOPC-Suff fails the control in two baseline cells: LaTa/MaRC (-0.189) and
mT5-base/IG (-0.030). In those cells the real attribution's sufficiency curve is
no better, and for LaTa/MaRC materially worse, than a shuffle of itself.

The `random` rows give the same gaps to within 0.008 of zero in every cell, so
the check is calibrated rather than biased upward by construction.

The shuffled-attribution gap and the random-order AUC gap agree numerically to
about 0.02 in every cell: compare the `Rand gap (AOPC-S)` and `InsAUC gap` rows
in Part A. They are two routes to the same correction, which is a consistency
check on both.

## A4. Effect of the erasure operator

Same 600 pairs, same stored attributions, two masking operators. `model` is the
published `summary.csv` (input-level PAD masking, re-running the encoder);
`hidden` is `summary_v2.csv` (representation-level masking).

| Metric | model backend | hidden backend | Conclusion |
|---|---|---|---|
| rho_LOO | 6/6 ABTT | 6/6 ABTT | stable |
| Suff@25% | 3/6 ABTT | 2/6 ABTT | one cell flips |
| Comp@25% | 3/6 ABTT | 2/6 ABTT | one cell flips |
| MinFrac@0.80 | 0/6 ABTT | 1/6 ABTT | one cell flips |

The largest per-cell shift in rho_LOO is 0.098 (LaTa/IG ABTT: 0.396 -> 0.298);
the sign and the direction of the ABTT effect are unchanged in all six cells.
Comprehensiveness moves most (LaTa/IG ABTT: 0.581 -> 0.194), which is expected:
input-level masking also re-contextualises the surviving tokens, so it charges
the removed token for a second effect.

Largest mean deviation between our recomputed full cosine and the stored
`cos_orig_*`, over all cells: **2.5e-07**. Both backends explain the same
decision scalar; only the erasure differs.

## A5. Two pairs of metrics are the same statistic

`AOPC-Comp = 1 - DelAUC` and `AOPC-Suff = InsAUC`, up to the difference between
a mean over k and a trapezoidal integral (observed gap about 0.008 in every
cell). Their win patterns in A1 are identical, cell for cell. Reporting both as
if they were independent evidence would be double counting, and the appendix
should say so.

---

# Part B. Selection for the main table

## B1. Criteria, stated before the selection

A metric earns a main-table column if it satisfies all of the following. These
are properties of a measurement instrument, not of the answer it gives.

1. **Standard.** It has a canonical published definition in the faithfulness
   literature, so a reviewer does not have to take our word for what it means.
2. **Threshold-free.** It summarises the whole attribution ranking rather than
   one arbitrary cut-off k. Our own sweep shows the threshold changes the
   answer (Sufficiency gives ABTT 2/6 at 10 and 25 percent but 5/6 at 50
   percent), so a single-threshold column would be a separate choice we would
   have to defend.
3. **Sensitive to sufficiency, not only to rank.** At least one main column must
   ask whether the selected tokens actually carry the score. A table made only
   of rank correlations does not answer the question a reader asks about a
   rationale.
4. **Calibrated.** The `random` control must sit at a known, model-independent
   value, so a number is interpretable without also knowing the query length and
   the anisotropy of the layer.
5. **Passes the shuffled-attribution control in every cell.** A metric on which a
   real attribution does not beat a permutation of itself is not measuring
   attribution on this data. Applied symmetrically: it disqualifies a metric
   whichever variant that metric happens to favour.
6. **Robust to the erasure operator**, where testable. The choice between
   input-level and representation-level masking is ours, not the model's, so a
   conclusion that flips with it is our artefact. Only the four legacy metrics
   exist under both backends, so this is a tiebreaker among them rather than a
   filter on the new ones.

## B2. Applying them

| Metric | 1 std | 2 free | 3 suff | 4 calib | 5 shuffle | 6 operator | Verdict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| rho_LOO | yes | yes | no | yes | 12/12 | stable | **main** (rank) |
| InsAUC gap | yes | yes | yes | yes | yes | n/a | **main** (sufficiency) |
| DelAUC gap | yes | yes | no | yes | yes | n/a | **main** (comprehensiveness) |
| tau_LOO | yes | yes | no | yes | 12/12 | n/a | appendix |
| AOPC-Comp | yes | yes | yes | **no** | 12/12 | n/a | appendix |
| AOPC-Suff | yes | yes | yes | **no** | **10/12** | n/a | appendix |
| DelAUC, InsAUC raw | yes | yes | yes | **no** | n/a | n/a | appendix |
| Suff@k | yes | **no** | yes | no | n/a | **flips** | appendix |
| Comp@k | yes | **no** | yes | no | n/a | **flips** | appendix |
| MinFrac@tau | partly | **no** | yes | no | n/a | **flips** | appendix |

Notes on the individual calls.

* **tau_LOO** is dropped for redundancy, not weakness. Across all 54 summary
  rows it correlates 0.9995 with rho_LOO and never disagrees in sign. It belongs
  in the appendix as a tie-corrected robustness check, which matters because
  MaRC produces many exactly-zero scores and Spearman handles ties by averaging
  ranks rather than correcting for them.
* **AOPC-Suff** fails criterion 5 in two cells. Both failures are on the
  *baseline* side, so keeping AOPC-Suff would have made ABTT look better, not
  worse. It is dropped anyway.
* **AOPC-Comp** passes the shuffled control everywhere but fails calibration: its
  random floor runs from 0.078 (mT5-base baseline) to 0.332 (mT5-base ABTT), so
  the same number means different things in the two columns being compared. It
  is fully represented in the main table by DelAUC gap, which is the same
  statistic with the floor subtracted (A5).
* **The ERASER trio** goes to the appendix, as decision 5 of the 2026-09-06 plan
  anticipated. The sweep in A1 and the backend comparison in A4 now give the
  plain reason: the verdict depends on the threshold and on the erasure operator.

## B3. The obvious objection

The three main columns give ABTT 6/6, 5/6 and 3/6. The middle column is the one
where the chance correction changed the outcome, since raw InsAUC gives ABTT 2/6
and the chance-corrected gap gives 5/6. That has to be defended, not asserted.

The correction is demanded by criterion 4, fixed before the numbers were looked
at, and its mechanism is measurable rather than assumed. The random-attribution
insertion floor is 0.853 (PhilTa) and 0.929 (mT5-base) under baseline, but 0.718
and 0.681 under ABTT. In a collapsed, anisotropic space an arbitrary subset of
tokens already reproduces most of the full cosine, so raw sufficiency near 1.0 at
baseline is a property of the geometry rather than of the attribution.
Subtracting the floor is what stops the baseline being credited for the very
collapse this paper is about.

Three things keep this from being a result-driven choice.

1. The same correction applied to the deletion side does **not** move the answer:
   raw DelAUC gives ABTT 3/6 and DelAUC gap gives ABTT 3/6, the same six cells.
   The correction is not a uniform ABTT bonus.
2. LaTa is a built-in counterexample. Its insertion floor barely moves (0.810
   baseline, 0.792 ABTT), the correction changes nothing there, and LaTa/IG stays
   a baseline win in both the raw and the corrected column.
3. The main table deliberately keeps DelAUC gap, a column where ABTT wins only
   half the cells. A table selected to flatter ABTT would not contain it.

If a reviewer prefers the uncorrected view, the appendix carries raw InsAUC and
AOPC-Suff with their 2/6, and the caption quotes the floor.

## B4. Recommended main table

Three columns, replacing the current four. All threshold-free, all with a
calibrated zero.

| Column | Key in `summary_v2.csv` | Direction | ABTT wins |
|---|---|---|---|
| rho_LOO | `loo_rho_mean` | higher | 6/6 |
| InsAUC gap | `ins_auc_gap_mean` | higher | 5/6 |
| DelAUC gap | `del_auc_gap_mean` | higher | 3/6 |

The caption must carry the random-attribution floor (`ins_auc_random_mean`,
`del_auc_random_mean`) so the gap is readable, and must repeat the standing
caveat that cross-variant comparisons are descriptive because ABTT changes both
the attribution scores and the cosine being explained.

Appendix, nothing deleted: Suff@{10,25,50}%, Comp@{10,25,50}%,
MinFrac@{0.70,0.80,0.90,0.95}, tau_LOO, AOPC-Suff, AOPC-Comp, raw DelAUC and
InsAUC, all shuffled-attribution gaps, and the `random` and `inverse` controls.
One plain sentence for the trio:

> The threshold-based ERASER metrics are reported for completeness. Their
> baseline-versus-ABTT verdict depends on the threshold and on the erasure
> operator, which is why the main table uses threshold-free, chance-corrected
> metrics instead.

## B5. The honest headline sentence

This is the sentence the paper should carry:

> Post-processing with ABTT improves the rank faithfulness of both attribution
> views in all six model-view cells (rho_LOO and Kendall tau, 6/6) and improves
> chance-corrected insertion faithfulness in five of six, but chance-corrected
> deletion faithfulness improves in only three of six, and the threshold-based
> ERASER metrics improve in between one and five of six depending on the
> threshold. We therefore claim that ABTT yields better-ranked and more
> sufficient rationales, not that it improves every faithfulness metric.

If issue #120 instead reports the uncorrected AUCs in the main table, the honest
sentence becomes:

> ABTT improves rank faithfulness in all six model-view cells, but raw
> sufficiency and comprehensiveness improve in only two and three of six. Those
> raw numbers are confounded: for PhilTa and mT5-base a random attribution
> already recovers 85 and 93 percent of the baseline cosine but only 72 and 68
> percent of the ABTT cosine, so the baseline columns are inflated by the same
> anisotropy the correction removes. For LaTa the floor barely moves (0.81 to
> 0.79) and the confound does not arise.

Either sentence is publishable and neither overclaims. The first is preferred
because it puts the confound-corrected numbers in the table rather than only in
the prose.

## B6. Action items for issue #120

1. Regenerate `overleaf_drafts/tables/attribution_metrics_main.tex` from
   `summary_v2.csv`, not `summary.csv`. Every number in the current table changes
   because the erasure operator changes; rho_LOO moves by at most 0.098 and keeps
   all six ABTT wins.
2. Add the erasure-operator sentence to the method text: masking is done at the
   representation level over cached layer-L hidden states.
3. Move the ERASER trio to the appendix with the sentence in B4.
4. Keep the shuffled-attribution control as one prose sentence in the main text.
   It is a validity statement, not a result column; the full gap table goes to
   the appendix.
5. Do not describe that control as an Adebayo sanity check. Call it a
   shuffled-attribution control.
6. Nothing in this memo licenses the claim "ABTT improves attribution quality"
   without a metric-specific qualifier. The guardrail list in
   `docs/runs/run4_interpretation_caveat_memo.md` still stands.
