# Attribution Metric Expansion and Selection

Generated: 2026-09-06. Issue #124, part of epic #109. Decision 5 of
`docs/research/plan_20260906.md`.

**Revision, 2026-09-06.** The independent review of PR #135 found that selection
criterion 5 had been applied to one metric and waived, on no computed number, for
its own chance-corrected twin. The control now covers both AUC gaps, the
criterion is applied identically to every candidate, and the recommended main
table changed as a result: it is now `rho_LOO` and `DelAUC gap`, and the
5/6 `InsAUC gap` column is in the appendix. B1 records what the rule was and why
it was not rewritten to save the column; B5 records what the headline sentence no
longer says.

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
| Random-order draws | 5 (`DEFAULT_RANDOM_ORDER_DRAWS`); shuffle draws 5 |
| Pairs per cell | 200 baseline, 191-195 ABTT for the ratio metrics (see A6) |
| Operator spot check | `.../attribution_metrics` is representation-level throughout; A8 adds a 20-pair input-level check |
| Compute | CPU only, `beto-delta-cpu` partition, no GPU hours spent |

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

The A8 spot check, which uses the *other* erasure operator and whose numbers must
never be mixed into the tables above:

```bash
python scripts/ig/run_attribution_metrics.py --backend model --device cpu \
  --models bowphs/LaTa --methods ig --max_pairs_per_model 20 \
  --metrics insertion_auc,deletion_auc --skip_pseudo_baselines \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --out_root <scratch> --out_subdir model20 \
  --summary_out <scratch>/model20_summary.csv \
  --sweep_summary_out <scratch>/model20_sweep.csv \
  --tex_out "" --require_artifacts
# then the same command with --backend hidden --out_subdir hidden20
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

The shuffled control covers six keys: `loo_rho`, `loo_tau`, `aopc_suff_ratio`,
`aopc_comp_ratio` and, added in response to the independent review of PR #135,
`ins_auc_gap` and `del_auc_gap` (`rand_ins_auc_gap_gap`,
`rand_del_auc_gap_gap`). Six, not four, because criterion 5 in Part B can only
be applied to a candidate that has a number.

**AOPC-Suff is the complement of DeYoung's sufficiency, not a sign error.**
DeYoung et al. report sufficiency as the *drop* caused by keeping only the
rationale, so lower is better there. We report the retained fraction, so higher
is better here, which puts it on the same scale and in the same direction as
every other ratio column in this memo. A reader who knows ERASER should read our
direction arrow as a re-parameterisation, not a mistake. The same applies to
`AOPC-Comp`, which is DeYoung's comprehensiveness with our ratio normalisation.

**Two ranking conventions apply to every curve metric.** Both are fixed in
`rank_order` in `src/attribution_metrics.py` and applied identically to every
method, every variant and every control row, but they are choices and the paper
has to state them.

* Tokens are ranked by **|a|**, not by signed importance. A token with a
  strongly negative contribution therefore ranks alongside a strongly positive
  one. RISE and ERASER rank by signed importance. We rank by magnitude because
  the question this paper asks is which tokens *carry* the pair cosine, and
  MaRC and IG both produce two-signed scores whose negative tail is not noise.
  The consequence is that our sufficiency curves can be reached by a token that
  pushes the cosine down; the effect is identical across variants, so it does
  not distort the baseline-versus-ABTT comparison, but it does make our absolute
  numbers non-comparable with a signed-ranking implementation.
* **Ties are broken by token position** (lowest index first, `np.argsort(...,
  kind="stable")`). Position is not a neutral order, and it matters most for
  MaRC, which produces many exactly-zero scores: for those tokens the ranking is
  effectively reading order. Kendall tau-b is in the appendix precisely because
  it is the tie-corrected companion to `rho_LOO` under that condition.

Two more notes on what these are and are not.

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
time, across all 3 models x 2 variants, is 2.5e-07 (section A4).

**What the 2.5e-07 number does and does not prove.** It establishes that our
pooling and our ABTT re-implementation reproduce the full-query cosine the model
actually produced at artifact-build time, so both backends explain the *same*
decision scalar. It says nothing whatever about any *masked* cosine, because
masking is exactly where the two operators diverge by construction. This number
must never appear in the paper next to a faithfulness claim, and must never be
cited as evidence that representation-level masking approximates input-level
masking.

**The interpretive limitation, which belongs in the paper.** Representation-level
erasure holds contextualisation fixed. A removed token's content still survives
inside its neighbours' layer-L states, so what these numbers measure is the
faithfulness of the **pooled read-out at layer L**, not of the encoder's
input-to-output map. A4 shows how large that difference is: comprehensiveness
moves from 0.581 to 0.194 on LaTa/IG between the two operators. Every
faithfulness statement derived from `summary_v2.csv` has to carry that scope.

The consequence is that `summary_v2.csv` is internally consistent (one operator
for every column, old metrics included) but is **not** row-for-row comparable
with the published `summary.csv`. That is why the old metrics were recomputed
rather than copied across. The size of the difference is quantified in A4, and
a bounded input-level spot check on the new AUC columns is in A8.

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
| Rand gap (InsAUC gap) | up | 0.172 -> 0.035 b | -0.189 -> 0.097 **A** | 0.116 -> 0.238 **A** | 0.107 -> 0.199 **A** | -0.030 -> 0.354 **A** | 0.019 -> 0.253 **A** |
| Rand gap (DelAUC gap) | up | 0.509 -> 0.179 b | 0.240 -> 0.285 **A** | 0.795 -> 0.418 b | 0.752 -> 0.353 b | 0.394 -> 0.551 **A** | 0.075 -> 0.444 **A** |

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
| Rand gap (InsAUC gap) | higher | new | b | A | A | A | A | A | 5/6 |
| Rand gap (DelAUC gap) | higher | new | b | A | b | b | A | A | 3/6 |

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

Every metric that is a candidate for a main-table column appears here. In the
first version of this memo the two AUC-gap rows were missing and the selection
table asserted a pass for them anyway; `randomization_check` now computes
`rand_ins_auc_gap_gap` and `rand_del_auc_gap_gap` directly, so the criterion is
read off a number in every row.

| Metric | Cells with a positive gap | Range | Failing cells |
|---|---|---|---|
| rho_LOO | 12/12 | +0.015 to +0.630 | none |
| tau_LOO | 12/12 | +0.008 to +0.467 | none |
| AOPC-Comp | 12/12 | +0.075 to +0.795 | none |
| **DelAUC gap** | 12/12 | +0.075 to +0.795 | none |
| AOPC-Suff | **10/12** | -0.189 to +0.354 | LaTa/MaRC baseline (-0.189), mT5-base/IG baseline (-0.030) |
| **InsAUC gap** | **10/12** | -0.189 to +0.354 | LaTa/MaRC baseline (-0.189), mT5-base/IG baseline (-0.030) |

`InsAUC gap` and `AOPC-Suff` do not merely agree: their shuffle gaps are the
same number. Over all 54 summary rows,
`max |rand_ins_auc_gap_gap - rand_aopc_suff_ratio_gap| = 2.2e-16` and
`max |rand_del_auc_gap_gap - rand_aopc_comp_ratio_gap| = 1.1e-16`. The random-order
reference is a property of the pair and is identical for the real and every
shuffled attribution, and the mean-over-k versus trapezoid offset `1/(2n)` is a
constant; both cancel exactly in the gap. A verdict that passes one and fails
the other is therefore not a judgement call but an error, which is what the
first version of B2 contained.

In the two failing cells the real attribution's sufficiency curve is no better,
and for LaTa/MaRC materially worse, than a shuffle of its own scores. Both are
baseline cells, so the reading is "the baseline attribution is worse than chance
on the sufficiency side here", not "ABTT looks bad". That does not save the
column: criterion 5 is a property of the instrument over the cells we measured,
and it is not satisfied.

The `random` rows give the same gaps to within 0.008 of zero in every cell, so
the check is calibrated rather than biased upward by construction.

The shuffled-attribution gap and the random-*order* AUC gap remain two different
corrections that agree numerically to about 0.02 in every cell: compare the
`Rand gap (AOPC-S)` and `InsAUC gap` rows in Part A. That agreement is a
consistency check on both, and it is distinct from the exact identity above,
which is between two shuffled-attribution gaps.

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
a mean over k = 1..n and a trapezoidal integral over k = 0..n. That difference
is the analytic constant `1/(2n)`: with `n_q_mean` between 83.6 and 111.2 it is
0.0045 to 0.0060 per side, and the largest observed discrepancy over all 54
summary rows is 0.0086. Their win patterns in A1 are identical, cell for cell.
Reporting both as if they were independent evidence would be double counting,
and the appendix should say so.

The consequence that matters for Part B: because the offset is a *constant*, it
cancels in any real-minus-shuffled difference, as does the random-order
reference. `AOPC-Suff` and `InsAUC gap` therefore cannot receive different
verdicts under criterion 5, and neither can `AOPC-Comp` and `DelAUC gap`.

## A6. Unequal n across variants (disclosure)

Every ratio metric is NaN when `|S_v(full)| < FULL_COS_FLOOR = 0.05`, and ABTT
pushes a handful of pairs below that floor while the baseline keeps all 200. The
AUC and AOPC columns therefore average 191 to 195 ABTT pairs against 200
baseline pairs, per model; the exact counts are in `summary_v2.csv` as
`ins_auc_gap_n`, `del_auc_gap_n` and their siblings. The `n` column of the
summary is the pair count *before* per-metric NaN filtering, so it does not show
this on its own.

We checked that it changes nothing. Recomputing every cell on the
both-variants-valid intersection, from the per-pair JSONs rather than the
summary means, reproduces the same winner in **every** cell for `loo_rho`
(200/200 pairs, no filtering), `ins_auc_gap`, `del_auc_gap` and both AUC-gap
shuffle columns. Zero verdict mismatches out of 42 metric-by-cell comparisons.
The disclosure is required; the correction is not.

## A7. Effect sizes for the main-table candidates

Win counts alone hide how big a win is. Paired per-pair differences, ABTT minus
baseline, over the both-variants-valid pairs:

| Cell | rho_LOO | DelAUC gap (main) | InsAUC gap (appendix) |
|---|--:|--:|--:|
| LaTa/IG | +0.285 +/- 0.019 (15.2 SE) | -0.336 +/- 0.031 (-10.9 SE) | -0.122 +/- 0.016 (-7.8 SE) |
| LaTa/MaRC | +0.327 +/- 0.027 (12.0 SE) | **+0.047 +/- 0.039 (1.2 SE)** | +0.304 +/- 0.033 (9.1 SE) |
| PhilTa/IG | +0.433 +/- 0.023 (19.0 SE) | -0.383 +/- 0.022 (-17.5 SE) | +0.125 +/- 0.016 (7.9 SE) |
| PhilTa/MaRC | +0.188 +/- 0.027 (7.0 SE) | -0.406 +/- 0.027 (-15.1 SE) | +0.096 +/- 0.023 (4.1 SE) |
| mT5-base/IG | +0.489 +/- 0.032 (15.3 SE) | +0.153 +/- 0.022 (7.0 SE) | +0.381 +/- 0.025 (15.5 SE) |
| mT5-base/MaRC | +0.143 +/- 0.026 (5.6 SE) | +0.357 +/- 0.026 (13.5 SE) | +0.225 +/- 0.030 (7.5 SE) |

`rho_LOO` is 5.6 SE or more in all six cells, so its 6/6 is not a run of narrow
wins. `DelAUC gap` is not so clean: one of its three wins, LaTa/MaRC, is inside
the noise at about 1.2 SE, while the other five cells are at least 7 SE from
zero. **A `3/6` for `DelAUC gap` is in truth two significant wins, one
statistical tie, and three significant losses**, and the paper's caption has to
say so rather than let the fraction imply six decided cells.

## A8. Erasure-operator spot check for the AUC columns

Criterion 6 could not be evaluated for `DelAUC gap` or `InsAUC gap` in the first
version of this memo: the curve metrics had only ever been run under
representation-level erasure. Since the headline claim now rests on
`DelAUC gap`, that `n/a` was not acceptable. This section closes it with a
bounded input-level run: **one model (LaTa), IG only, the first 20 pairs, both
variants, `--backend model --metrics insertion_auc,deletion_auc`**, on the CPU
partition, no GPU hours. The same 20 pairs are re-run under `--backend hidden`
so the two operators are compared on an identical subset rather than against the
200-pair table.

Means over the 19 pairs valid under both variants (one of the 20 falls below
`FULL_COS_FLOOR` under ABTT), with the paired ABTT-minus-baseline difference and
its standard error:

| Operator | Metric | baseline | ABTT | ABTT - baseline | Winner |
|---|---|--:|--:|--:|:-:|
| input-level (`model`) | InsAUC | 0.977 | 0.792 | -0.183 +/- 0.026 | baseline |
| input-level | InsAUC random floor | 0.784 | 0.582 | -0.195 +/- 0.040 | -- |
| input-level | **InsAUC gap** | 0.193 | 0.210 | **+0.011 +/- 0.045 (0.3 SE)** | ABTT |
| input-level | DelAUC | 0.311 | 0.288 | +0.011 +/- 0.102 | ABTT |
| input-level | DelAUC random floor | 0.850 | 0.597 | -0.252 +/- 0.048 | -- |
| input-level | **DelAUC gap** | 0.539 | 0.309 | **-0.263 +/- 0.095 (-2.8 SE)** | baseline |
| representation-level (`hidden`) | InsAUC | 0.971 | 0.825 | -0.144 +/- 0.020 | baseline |
| representation-level | InsAUC random floor | 0.797 | 0.777 | -0.013 +/- 0.035 | -- |
| representation-level | **InsAUC gap** | 0.174 | 0.048 | **-0.131 +/- 0.040 (-3.3 SE)** | baseline |
| representation-level | DelAUC | 0.358 | 0.589 | +0.263 +/- 0.087 | baseline |
| representation-level | DelAUC random floor | 0.857 | 0.788 | -0.068 +/- 0.026 | -- |
| representation-level | **DelAUC gap** | 0.499 | 0.198 | **-0.331 +/- 0.088 (-3.8 SE)** | baseline |

**`DelAUC gap`, the main-table column, agrees.** Both operators make LaTa/IG a
baseline win, and both make it a significant one (-2.8 SE and -3.8 SE). That
matches the 200-pair verdict for this cell and turns criterion 6 for
`DelAUC gap` from `n/a` into a pass, on the one cell we could afford to test.
`DelAUC` raw and `InsAUC` raw are not the columns that carry the claim; `DelAUC`
raw does flip.

**`InsAUC gap` is not confirmed.** Its nominal winner flips between operators,
but the input-level difference is +0.011 +/- 0.045, about 0.3 SE, which is a
statistical tie rather than an ABTT win. The honest reading is that on this cell
the input-level operator does not determine a winner at all, so the spot check
neither confirms nor contradicts the representation-level result. It is an
additional reason not to put the column in the main table, not the reason: the
column is in the appendix because it fails criterion 5.

The floors are worth a note of their own. The input-level operator's random
floors move far more between variants (`ins_auc_random` 0.784 to 0.582,
`del_auc_random` 0.850 to 0.597) than the representation-level operator's
(0.797 to 0.777, 0.857 to 0.788). Re-contextualisation is itself variant
dependent, which is a further reason the two operators' absolute numbers must
never be mixed in one table.

The check is bounded and should be read as such: 20 pairs, one model, one view.
It tests whether the *sign* of the baseline-versus-ABTT difference survives the
change of operator, not whether the magnitudes agree.

**The remaining limitation stands regardless of how this check comes out.** All
600-pair numbers in this memo come from representation-level masking, which
holds contextualisation fixed. They are faithfulness statements about the pooled
read-out at layer L, not about the encoder's input-to-output map: a masked
token's content still survives inside its neighbours' layer-L states. The
2.5e-07 full-cosine drift in A4 validates the *unmasked* reproduction only and
is not evidence about masking of any kind.

Two smaller caveats on the reference itself. `DEFAULT_RANDOM_ORDER_DRAWS = 5`,
so the random-order reference carries a Monte-Carlo error of roughly +/-0.01 on
a cell mean; that is small against the win margins but comparable to the
"`random` sits within 0.02 of zero" calibration claim in A2 and to the smallest
reported gaps. It was left at 5 rather than raised so that every number the
independent review of PR #135 reproduced stays bit-identical; the
`--random_order_draws` flag exists and a table-generating run for the paper
should use 20.

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
   exist under both backends at full scale, so this is a tiebreaker among them
   rather than a filter on the new ones; A8 adds a bounded input-level spot
   check for the AUC columns, which `DelAUC gap` passes.

### How criterion 5 is applied, and how it was applied wrongly first

The first version of this memo (PR #135, commit `4aa888b`) applied criterion 5
to `AOPC-Suff` and recorded `yes` for `InsAUC gap` and `DelAUC gap`. That `yes`
had no number behind it: `randomization_check` only shuffled for `loo_rho`,
`loo_tau`, `aopc_suff_ratio` and `aopc_comp_ratio`, so no column of
`summary_v2.csv` reported a shuffled-attribution gap for the AUC gaps at all.
The independent review of that PR caught it. The consequence was not neutral:
the un-evidenced pass sat on the column ABTT wins 5/6, and the evaluated failure
struck the column ABTT wins 2/6.

`randomization_check` now computes the gap for `ins_auc_gap` and `del_auc_gap`
directly (`rand_ins_auc_gap_gap`, `rand_del_auc_gap_gap` in `summary_v2.csv`),
so criterion 5 is evaluated from a number for every candidate. Two rules fix how
it is read, and both are stated here before the new numbers are consulted.

* **The criterion text is not rewritten.** It says "in every cell", so it is a
  per-cell rule over all twelve cells (3 models x 2 views x 2 variants), and a
  metric with any non-positive cell fails. Rewriting it into a pooled
  discrimination test would be a defensible criterion in the abstract, but
  choosing it *after* learning that the per-cell version strikes the column ABTT
  wins is exactly the tuning this memo exists to avoid. The criterion was
  pre-registered; only its application is being corrected.
* **Metrics that are the same statistic share one verdict.** A5 already shows
  `AOPC-Suff = InsAUC` and `AOPC-Comp = 1 - DelAUC`. For the shuffled control
  the identity is exact rather than approximate: the random-order reference is a
  property of the pair, identical for the real and every shuffled attribution,
  and the mean-over-k versus trapezoid offset `1/(2n)` is a constant, so both
  cancel in the gap. `rand_ins_auc_gap_gap` equals `rand_aopc_suff_ratio_gap`
  and `rand_del_auc_gap_gap` equals `rand_aopc_comp_ratio_gap` to machine
  precision (A3), and it is not possible to pass one and fail the other.

## B2. Applying them

Criterion 5 is now read from a computed number for every row (A3), and the
`n/a`s that used to sit in that column are gone.

Column 3 is headed "not-only-rank" rather than "suff" because that is what
criterion 3's own justification sentence asks for: a table not made only of rank
correlations. Read narrowly, as "at least one main column asks whether the
selected tokens on their own carry the score", **criterion 3 is not satisfied by
the recommended table**, because criterion 5 strikes every sufficiency-side
candidate. The criterion text is left as written and the shortfall is recorded
in B3 rather than defined away.

| Metric | 1 std | 2 free | 3 not-only-rank | 4 calib | 5 shuffle | 6 operator | Verdict |
|---|:-:|:-:|:-:|:-:|:-:|:-:|---|
| rho_LOO | yes | yes | no | yes | 12/12 | stable | **main** (rank) |
| DelAUC gap | yes | yes | yes | yes | 12/12 | **agrees** (A8) | **main** (comprehensiveness) |
| InsAUC gap | yes | yes | yes | yes | **10/12** | not confirmed (A8) | appendix |
| tau_LOO | yes | yes | no | yes | 12/12 | n/a | appendix (robustness twin of rho_LOO) |
| AOPC-Comp | yes | yes | yes | **no** | 12/12 | n/a | appendix |
| AOPC-Suff | yes | yes | yes | **no** | **10/12** | n/a | appendix |
| DelAUC, InsAUC raw | yes | yes | yes | **no** | n/a | n/a | appendix |
| Suff@k | yes | **no** | yes | no | n/a | **flips** | appendix |
| Comp@k | yes | **no** | yes | no | n/a | **flips** | appendix |
| MinFrac@tau | partly | **no** | yes | no | n/a | **flips** | appendix |

Notes on the individual calls.

* **InsAUC gap fails criterion 5, exactly as AOPC-Suff does, and for the same
  reason: they are the same statistic (A5).** The two failing cells are
  LaTa/MaRC baseline and mT5-base/IG baseline. Both are baseline cells, and both
  are cells where ABTT wins the InsAUC gap, so striking the column costs ABTT a
  5/6 and this memo its most favourable number. That is what applying a
  pre-registered validity criterion symmetrically means. Under the earlier,
  asymmetric application the criterion struck the column ABTT loses (AOPC-Suff,
  2/6) and passed the column ABTT wins (InsAUC gap, 5/6) on no evidence at all.
* **A negative shuffle gap is a statement about that cell, not only about the
  metric.** In LaTa/MaRC baseline the real attribution's sufficiency curve is
  materially worse (-0.189) than a permutation of its own scores; in mT5-base/IG
  baseline it is marginally worse (-0.030). That is a genuine finding about
  baseline attributions on those two cells, and it goes in the appendix with the
  number. It is not a reason to keep the column: a column whose reading in two of
  twelve cells is "this attribution is worse than its own shuffle" cannot be a
  headline measurement.
* **tau_LOO** is dropped for redundancy, not weakness. Across all 54 summary
  rows it correlates 0.9995 with rho_LOO and never disagrees in sign. It goes to
  the appendix as the **tie-corrected robustness twin of rho_LOO**, not as a
  second main column, which matters because MaRC produces many exactly-zero
  scores and Spearman handles ties by averaging ranks rather than correcting for
  them. The paper must not quote "rho and tau, 6/6" as if that were two
  independent results.
* **AOPC-Comp** passes the shuffled control everywhere but fails calibration: its
  random floor runs from 0.078 (mT5-base baseline) to 0.332 (mT5-base ABTT), so
  the same number means different things in the two columns being compared. It
  is fully represented in the main table by DelAUC gap, which is the same
  statistic with the floor subtracted (A5).
* **The ERASER trio** goes to the appendix, as decision 5 of the 2026-09-06 plan
  anticipated. The sweep in A1 and the backend comparison in A4 now give the
  plain reason: the verdict depends on the threshold and on the erasure operator.
* **Criterion 4 is satisfied by construction for the gap metrics, so it is not
  independent evidence for them.** A random attribution *is* a random ordering,
  so its gap is zero by definition. Criterion 4 discriminates between gaps and
  raws (it is what sends raw InsAUC/DelAUC and both AOPC halves to the
  appendix); among the gap metrics it carries no information.

## B3. What the symmetric application costs, and the objection it removes

Applying criterion 5 symmetrically strikes the sufficiency side of the table
entirely. `AOPC-Suff`, `InsAUC` and `InsAUC gap` are all in the appendix, so no
main column asks "do the top-ranked tokens on their own reproduce the score".
That is a real loss and the paper has to own it: **on this data no
sufficiency-side metric survives our own validity check in every cell.**

Criterion 3 is still met, on its own stated rationale. Its justification
sentence is that "a table made only of rank correlations does not answer the
question a reader asks about a rationale", and `DelAUC gap` is a perturbation
metric, not a rank correlation: it asks whether removing the top-ranked tokens
destroys the score. What is missing from the main table is the narrower
sufficiency half specifically, and A1 plus the appendix carry it in full.

The earlier draft of this memo spent a section defending `InsAUC gap` against
the charge that its chance correction manufactured the 5/6, since raw `InsAUC`
gives 2/6. That defence was sound as far as it went, and the independent review
of PR #135 strengthened it: a headroom-normalised skill score
`(InsAUC - random)/(1 - random)`, which removes the ceiling advantage rather
than subtracting it, gives the same 5/6 in the same cells. The mechanism is
also measurable rather than assumed: the random-attribution insertion floor is
0.853 (PhilTa) and 0.929 (mT5-base) at baseline but 0.718 and 0.681 under ABTT,
so raw sufficiency near 1.0 at baseline reflects the collapsed geometry rather
than the attribution, while for LaTa the floor barely moves (0.810 to 0.792) and
the confound does not arise.

None of that rescues the column, because criterion 5 is prior to it. A metric
that does not beat a permutation of itself in two of twelve cells is not
measured well enough on this data to carry a headline claim, whatever the
correction does elsewhere. The argument above is kept because it is the reason
the appendix reports `InsAUC gap` rather than raw `InsAUC` as the sufficiency
number of record.

The one thing this rewrite does **not** change is the most credibility-buying
decision in the selection: `DelAUC gap` stays in the main table at 3/6 (and, per
A7, two significant wins, one tie and three significant losses). A table
selected to flatter ABTT would not contain it, and would not have lost the 5/6
column. The bounded input-level spot check in A8 is a small piece of positive
evidence for keeping it: on LaTa/IG, the one cell we could afford to test under
the other erasure operator, `DelAUC gap` gives the same baseline win under both
operators and does so significantly under both.

## B4. Recommended main table

**Two columns**, replacing the current four.

| Column | Key in `summary_v2.csv` | Direction | ABTT wins |
|---|---|---|---|
| rho_LOO | `loo_rho_mean` | higher | 6/6 |
| DelAUC gap | `del_auc_gap_mean` | higher | 3/6 |

Both are threshold-free and both have a calibrated zero. `tau_LOO` is **not** a
third column: it is reported in the appendix as the tie-corrected robustness
twin of `rho_LOO`, with which it correlates 0.9995.

The caption must carry:

* the random-order floor (`del_auc_random_mean`) so the gap is readable;
* the per-cell standard errors, or a sentence saying that the LaTa/MaRC
  `DelAUC gap` win is inside the noise at about 1.2 SE (A7);
* the pair counts, since ABTT contributes 191 to 195 pairs against the
  baseline's 200 (A6);
* the standing caveat that cross-variant comparisons are descriptive, because
  ABTT changes both the attribution scores and the cosine being explained;
* the scope of the faithfulness claim: representation-level masking over cached
  layer-L hidden states, so these are statements about the pooled read-out at
  layer L rather than about the encoder's input-to-output map.

Appendix, nothing deleted: Suff@{10,25,50}%, Comp@{10,25,50}%,
MinFrac@{0.70,0.80,0.90,0.95}, tau_LOO, AOPC-Suff, AOPC-Comp, raw DelAUC and
InsAUC, **InsAUC gap**, all shuffled-attribution gaps, and the `random` and
`inverse` controls. Two plain sentences:

> The threshold-based ERASER metrics are reported for completeness. Their
> baseline-versus-ABTT verdict depends on the threshold and on the erasure
> operator, which is why the main table uses threshold-free, chance-corrected
> metrics instead.

> Chance-corrected insertion faithfulness favours ABTT in five of six cells, but
> we do not report it in the main table: in two of the twelve cells, both of
> them baseline cells, the real attribution does not beat a permutation of its
> own scores, so the measurement does not meet the validity bar we set for a
> headline column.

## B5. The honest headline sentence

This is the sentence the paper should carry:

> Post-processing with ABTT improves rank faithfulness in all six model-view
> cells (rho_LOO 6/6, with tie-corrected Kendall tau-b agreeing in all six), but
> chance-corrected deletion faithfulness improves in only three of six, and one
> of those three wins is within 1.2 standard errors of zero. The sufficiency-side
> metrics are reported in the appendix rather than the main table because they
> fail our shuffled-attribution control in two baseline cells. We therefore claim
> that ABTT yields better-ranked rationales, not that it improves faithfulness on
> every axis.

If issue #120 prefers to lead with the uncorrected view, the honest sentence
becomes:

> ABTT improves rank faithfulness in all six model-view cells, but raw
> sufficiency and comprehensiveness improve in only two and three of six. Those
> raw numbers are confounded: for PhilTa and mT5-base a random attribution
> already recovers 85 and 93 percent of the baseline cosine but only 72 and 68
> percent of the ABTT cosine, so the baseline columns are inflated by the same
> anisotropy the correction removes. For LaTa the floor barely moves (0.81 to
> 0.79) and the confound does not arise.

Either sentence is publishable and neither overclaims. The first is preferred
because it reports the corrected numbers in the table rather than only in prose.

Note what the first sentence no longer says. The version in PR #135 commit
`4aa888b` claimed that ABTT "improves chance-corrected insertion faithfulness in
five of six" as a main-table result, and cited "rho_LOO and Kendall tau, 6/6" as
though those were two independent measurements. Both are gone: the first because
the column did not survive criterion 5, the second because tau is a twin of rho,
not a second witness.

## B6. Action items for issue #120

1. Regenerate `overleaf_drafts/tables/attribution_metrics_main.tex` from
   `summary_v2.csv`, not `summary.csv`. Every number in the current table changes
   because the erasure operator changes; rho_LOO moves by at most 0.098 and keeps
   all six ABTT wins.
2. Add the erasure-operator sentence to the method text: masking is done at the
   representation level over cached layer-L hidden states, **so the faithfulness
   statements are about the pooled read-out at layer L, not about the encoder's
   input-to-output map** (a removed token's content survives inside its
   neighbours' states). Never cite the 2.5e-07 full-cosine drift as evidence
   about masking; it validates the unmasked reproduction only.
3. Move the ERASER trio to the appendix with the first sentence in B4, and
   `InsAUC gap` with the second.
4. Keep the shuffled-attribution control as one prose sentence in the main text.
   It is a validity statement, not a result column; the full gap table goes to
   the appendix.
5. Do not describe that control as an Adebayo sanity check. Call it a
   shuffled-attribution control.
6. State the two ranking conventions (magnitude ranking, positional tie-break)
   in the method text, and report the unequal pair counts (A6) and the per-cell
   standard errors (A7) in the caption.
7. Nothing in this memo licenses the claim "ABTT improves attribution quality"
   without a metric-specific qualifier. The guardrail list in
   `docs/runs/run4_interpretation_caveat_memo.md` still stands.
