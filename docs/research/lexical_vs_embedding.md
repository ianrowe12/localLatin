# Where do embeddings beat surface overlap?

Issue #132, part of epic #109. Rewritten 2026-09-06 against the **corrected** benchmark v1
split at `runs/active/resubmit/data/phase_resubmit_split.csv` (post-#131) and the
configurations in `runs/active/resubmit/results/phase_resubmit_results.csv`.

Regenerate everything below with one CPU-only command from the repo root:

```
python scripts/resubmit/lexical_vs_embedding.py
```

Outputs: `runs/active/resubmit/results/lexical_vs_embedding.csv` (78 rows, one per method
per stratum) and `overleaf_drafts/figures/fig_lexical_vs_embedding.pdf`. Both are gitignored.

## The question

PR #130 (issue #122) found that a character 3-5-gram TF-IDF cosine matches or beats every
embedding configuration on Task A and Task B, corpus-wide. That is an average, and an
average can hide a division of labour: the plausible story is that surface overlap carries
the easy near-duplicate witnesses while embeddings earn their place on the pairs whose
wording has drifted. This note tests that story directly.

## Setup

**Stratifying variable.** The raw character 3-5-gram TF-IDF cosine of a pair, vectoriser
fitted on train files only (the `tfidf_char35` scorer of `scripts/resubmit/lexical_baselines.py`,
read before its train min-max rescaling; the rescaling is monotone, so it moves no boundary).

**Pairs.** The 596 same-directory pairs among the 858 test files, cut into overlap terciles
of 199 / 198 / 199 pairs. Boundaries: low `< 0.526`, mid `0.526-0.799`, high `> 0.799`; the
full range of positive-pair overlap is `0.051` to `0.982`. A fourth `hard` slice holds the
122 positives below the TF-IDF Task-A threshold, `tau = 0.417`, i.e. the pairs the lexical
scorer itself would route as new. The hard slice is not a fourth tercile; it sits inside the
low tercile by construction. (PR #130 rescales the score matrix using the train block, and
that block runs from exactly 0.0 to 1.0 here, since some train pair shares no character
n-gram at all and another is an exact duplicate after normalisation. The rescaling is
therefore the identity on this matrix, and the rescaled and raw thresholds coincide.)

**Embedding rows are aligned by filename, not by row index.** The cached matrices under
`runs/active/resubmit_bases/phase9_bases/` are keyed by position, and #131's key corrections
renumbered 17 `file_id`s in the window 1554-1570 without changing a byte of text (see
`docs/research/benchmark_v1.md`). Reading the caches positionally against the corrected split
therefore hands seventeen files each other's vectors. This script instead reorders every
cache through `phase9_bases/row_order.csv`, matching on filename, which is unique across the
corpus. On the pre-correction split that reordering is the identity, so it changes none of
the original #132 numbers; on the corrected split it moves exactly those 17 rows. The
misalignment is not cosmetic: read positionally, seven of the twelve configurations learn a
`tau` that disagrees with `phase_resubmit_results.csv` (mT5-base ABTT lands at 0.4020 against
0.4372), and low-tercile recall@1 for LaBSE+ABTT reads 0.623 instead of 0.598.

**Methods.** The lexical scorer, plus per model the paper's Baseline configuration
(mean-pool, no correction) and the paper's ABTT configuration (mean-pool + `EmbeddingCleaner`
fitted on train, no SIF), each at the layer the headline tables pick, which is the layer of
highest train AUROC. Layer, `D`, representation and pooling are read from
`phase_resubmit_results.csv` rather than re-tuned, and the run prints each learned `tau`
against the CSV: **all twelve match to four decimals**, so these are the paper's
configurations and not a re-derivation of them. That check is now fatal rather than advisory:
a disagreement aborts the run unless `--allow-tau-drift` is passed.

| Model | Baseline layer | ABTT layer | ABTT `D` |
|---|---|---|---|
| LaTa | 12 | 12 | 10 |
| PhilTa | 1 | 1 | 10 |
| mT5-base | 12 | 1 | 10 |
| LaBSE | 12 | 11 | 10 |
| Qwen3-0.6B | 28 | 2 | 10 |
| KaLM-mini | 23 | 1 | 10 |

**Metrics.** A positive pair gives two directed observations, one per endpoint as query.
`recall_at_1` and `mrr` rank the partner among the test files with the query's *other*
partners dropped from the candidate pool. That residual convention matters here: low-overlap
positives concentrate in the large directories, where a query with three partners can put at
most one of them at literal rank 1, so the literal number (`recall_at_1_all` in the CSV)
pins every method near 0.10 in the low tercile and measures directory size rather than
retrieval quality. `frac_above_tau` is the share of positive pairs scored at or above the
method's own train-learned threshold, which is the routing question rather than the ranking
question.

**Significance.** Every method is compared against the lexical scorer with an exact two-sided
McNemar test, paired observation by observation inside the stratum. `p` columns in the tables
below are those tests: `mcnemar_p` on the ranking outcome, `routing_p` on the routing outcome.
The CSV also carries the discordant counts `b` (only TF-IDF gets it) and `c` (only the
embedding gets it), which are the whole content of the test. All p-values are uncorrected;
twelve methods are compared in each stratum, so a p around 0.03 is not a finding that would
survive a multiplicity correction.

## Ranking: partner recall@1 by overlap tercile

`p` is the exact McNemar p-value against char TF-IDF in that stratum, with the discordant
counts in brackets as (only TF-IDF, only this method).

| Method | low | p (low) | mid | high | all 596 | p (all) |
|---|---|---|---|---|---|---|
| **char TF-IDF 3-5** | **0.646** | | **0.990** | **1.000** | **0.878** | |
| LaBSE + ABTT | 0.598 | 0.023 (41, 22) | 0.965 | 1.000 | 0.854 | 0.0011 (52, 23) |
| PhilTa + ABTT | 0.548 | 1.2e-06 (52, 13) | 0.970 | 1.000 | 0.839 | 2.3e-08 (60, 13) |
| Qwen3-0.6B + ABTT | 0.503 | 1.2e-09 (74, 17) | 0.955 | 0.997 | 0.818 | 3.2e-12 (92, 20) |
| mT5-base + ABTT | 0.500 | 1.4e-10 (72, 14) | 0.967 | 0.997 | 0.821 | 1.4e-12 (83, 15) |
| KaLM-mini + ABTT | 0.485 | 1.6e-10 (84, 20) | 0.972 | 0.997 | 0.818 | 8.7e-12 (94, 22) |
| LaTa + ABTT | 0.470 | 9.1e-13 (86, 16) | 0.927 | 0.992 | 0.796 | 9.9e-19 (116, 18) |
| LaBSE baseline | 0.525 | 1.9e-07 (67, 19) | 0.904 | 0.985 | 0.805 | 1.6e-15 (109, 21) |
| KaLM-mini baseline | 0.487 | 6.4e-10 (85, 22) | 0.939 | 0.997 | 0.808 | 1.1e-13 (109, 25) |
| Qwen3-0.6B baseline | 0.412 | 2.7e-19 (106, 13) | 0.876 | 0.965 | 0.751 | 1.2e-33 (167, 15) |
| PhilTa baseline | 0.339 | 1.2e-31 (127, 5) | 0.833 | 0.995 | 0.722 | 4.0e-49 (192, 6) |
| LaTa baseline | 0.261 | 2.0e-34 (167, 14) | 0.712 | 0.950 | 0.641 | 1.9e-69 (298, 15) |
| mT5-base baseline | 0.098 | 2.0e-57 (225, 7) | 0.424 | 0.726 | 0.416 | 1.1e-153 (559, 8) |

The predicted division of labour does not appear. The lexical scorer is first in every
tercile, and its margin over the best embedding configuration is *largest* in the low
tercile (4.8 points over LaBSE+ABTT) rather than smallest. Every method converges on the
high tercile, where near-duplicate wording makes the task trivial for anything and no
embedding is distinguishable from TF-IDF at all.

What the terciles do show clearly is ABTT's own effect, which is an internal-geometry result
and is untouched by the lexical comparison: correction lifts the low tercile by 21 points for
LaTa (0.261 to 0.470), 21 for PhilTa and 40 for mT5-base, and it compresses the six models
from a 43-point spread at baseline (0.098 to 0.525) into a 13-point band (0.470 to 0.598).
That is the anisotropy-dip story, and it survives intact.

## The hard slice: 122 positives the lexical scorer would reject

| Method | recall@1 | p vs TF-IDF | MRR | above own tau |
|---|---|---|---|---|
| char TF-IDF 3-5 | **0.443** | | **0.530** | 0.000 (by construction) |
| LaBSE + ABTT | 0.377 | 0.033 (33, 17) | 0.447 | **0.246** |
| PhilTa + ABTT | 0.352 | 0.0016 (34, 12) | 0.439 | 0.090 |
| LaTa + ABTT | 0.299 | 3.3e-06 (46, 11) | 0.376 | 0.115 |
| KaLM-mini + ABTT | 0.275 | 1.0e-06 (56, 15) | 0.349 | 0.139 |
| Qwen3-0.6B + ABTT | 0.283 | 7.5e-07 (51, 12) | 0.353 | 0.049 |
| mT5-base + ABTT | 0.254 | 6.9e-09 (56, 10) | 0.352 | 0.016 |
| LaBSE baseline | 0.332 | 0.00036 (41, 14) | 0.378 | 0.090 |
| KaLM-mini baseline | 0.316 | 0.00015 (48, 17) | 0.376 | 0.066 |
| Qwen3-0.6B baseline | 0.242 | 3.2e-10 (57, 8) | 0.311 | 0.066 |
| PhilTa baseline | 0.164 | 1.1e-18 (70, 2) | 0.230 | 0.000 |
| LaTa baseline | 0.156 | 1.1e-14 (80, 10) | 0.221 | 0.016 |
| mT5-base baseline | 0.016 | 3.4e-28 (107, 3) | 0.054 | 0.000 |

This slice is selected as the pairs the lexical scorer scores lowest, so it is stacked
against TF-IDF, and TF-IDF still ranks them best. The `above own tau` column looks like the
one place an embedding does something the lexical scorer cannot, but the zero it is measured
against is definitional, not empirical: the slice *is* the set of pairs TF-IDF puts below its
threshold. Quoting LaBSE+ABTT's 0.246 as a 25-point win, or its p of 1.9e-09, would be
dishonest. The threshold-free version of that comparison is the next section.

## Routing at matched operating points

The routing comparison only means something if the thresholds being compared are the same
operating point. Each method sits at its own train-learned `tau`; here is where that `tau`
lands on the whole test block (596 positive pairs, 367,057 negative pairs).

| Method | tau | TPR | FPR | precision | false positives |
|---|---|---|---|---|---|
| char TF-IDF 3-5 | 0.417 | 0.795 | 0.00029 | 0.816 | 107 |
| LaBSE + ABTT | 0.452 | 0.792 | 0.00026 | 0.830 | 97 |
| KaLM-mini + ABTT | 0.412 | 0.752 | 0.00023 | 0.839 | 86 |
| PhilTa + ABTT | 0.417 | 0.758 | 0.00029 | 0.810 | 106 |
| LaTa + ABTT | 0.422 | 0.711 | 0.00050 | 0.697 | 184 |
| Qwen3-0.6B + ABTT | 0.452 | 0.676 | 0.00022 | 0.831 | 82 |
| mT5-base + ABTT | 0.437 | 0.671 | 0.00020 | 0.844 | 74 |

TF-IDF and LaBSE+ABTT are at the same operating point in every respect that matters: equal
overall TPR to three decimals, LaBSE marginally *tighter* on false positives (97 against
107) and marginally higher precision. So LaBSE+ABTT's low-tercile routing figure is not
bought with a looser threshold.

On the low tercile, which is defined without reference to any threshold, LaBSE+ABTT carries
0.427 of the positives above its `tau` against TF-IDF's 0.387. Paired, that is 30
observations only LaBSE routes correctly against 22 only TF-IDF routes correctly, exact
McNemar **p = 0.33**. The edge is real in sign and it is not a threshold artefact, but on 199
pairs it is not distinguishable from parity. Every other ABTT model is below TF-IDF on the
same measure and several significantly so: KaLM-mini 0.312 (p = 0.044), PhilTa 0.286
(p = 0.0029), LaTa 0.256 (p = 0.00054), Qwen3-0.6B 0.166 (p = 1.0e-09), mT5-base 0.106
(p = 3.2e-15). Over all 596 positives LaBSE+ABTT and TF-IDF are indistinguishable
(0.792 against 0.795, p = 0.90).

## Reverse slice: negatives with high surface overlap

The 36,706 different-directory test pairs in the top decile of TF-IDF overlap (cosine
`>= 0.148`), giving 47,773 directed observations after dropping endpoints with no test
partner. A method is confused when it scores the impostor above every true partner of that
query. `p` is the exact McNemar test on those confusions, so a low `p` with the ABTT rate
below TF-IDF's means the embedding is confused significantly *less* often.

| Model | baseline | ABTT | ABTT p vs TF-IDF |
|---|---|---|---|
| char TF-IDF 3-5 | 0.0055 | | |
| LaBSE | 0.0169 | **0.0045** | 0.0024 (134, 88) |
| PhilTa | 0.0187 | 0.0046 | 0.0011 (105, 62) |
| LaTa | 0.0144 | 0.0047 | 0.019 (137, 100) |
| mT5-base | 0.0549 | 0.0060 | 0.14 (129, 155) |
| Qwen3-0.6B | 0.0114 | 0.0064 | 0.0048 (95, 139) |
| KaLM-mini | 0.0134 | 0.0067 | 0.00048 (101, 158) |

Near-duplicate wording across genuinely different fragments is a small problem for everyone.
Uncorrected embeddings are the most vulnerable (mT5-base gets it wrong ten times as often as
TF-IDF, p effectively 0 for all six), and ABTT removes most of that gap. After correction the
picture splits three ways: LaBSE, PhilTa and LaTa reject these impostors significantly more
often than TF-IDF, mT5-base is at parity, and Qwen3-0.6B and KaLM-mini are significantly
worse. The significant margins are tiny in absolute terms, around one confusion in a
thousand observations, and it takes 47,773 observations to see them; this is the one metric
here on which the *paired* test resolves what the raw rates cannot, and it is also the
metric that moved most under #131's two-file relabelling, so treat it as fragile.

## Verdict

**Redundant, not complementary.** On this corpus a character 3-5-gram TF-IDF cosine ranks
the true partner at least as well as every embedding configuration in every overlap stratum,
and the corpus-wide lead over the best embedding, LaBSE+ABTT, is solid on a paired test
(0.878 against 0.854, McNemar p = 0.0011). On the two slices where the hoped-for division of
labour would have to appear, the lead is smaller and the evidence is weaker but still points
the same way: 0.646 against 0.598 in the
low-overlap tercile (p = 0.023) and 0.443 against 0.377 on the hard slice of positives TF-IDF
itself would reject (p = 0.033). Those two p-values are uncorrected and would not survive
correction for the twelve methods compared in each stratum, so the honest reading is that
TF-IDF is never worse anywhere and is probably better in the low-overlap regime too, not that
its low-overlap lead is established at the strength of the corpus-wide one. What is
established is the absence of the effect this analysis was built to find: there is no overlap
regime in which any embedding configuration ranks better than surface overlap, and the
lexical margin is widest, not narrowest, where the wording has drifted furthest. The
exception is routing rather than ranking, and it is one model: LaBSE+ABTT carries 0.427 of
the lowest-overlap positives above its threshold against TF-IDF's 0.387, at an operating
point matched on TPR and slightly tighter on false positives, so it is not a threshold
artefact, but paired it is p = 0.33 and cannot be told apart from parity on 199 pairs. On
high-overlap impostors the corrected embeddings are mixed: LaBSE, PhilTa and LaTa with ABTT
are confused significantly less often than TF-IDF (0.0045 to 0.0047 against 0.0055,
p = 0.0011 to 0.019), mT5-base is at parity, and Qwen3-0.6B and KaLM-mini are significantly
worse, all of it inside a band of about one confusion per thousand observations. The
retrieval framing in F6 and F8 should therefore not claim that embeddings add retrieval
signal beyond surface overlap on this corpus, and should present the lexical baseline as the
operating point a practitioner would actually pick. What the stratification does support, and
support strongly, is the paper's actual subject: ABTT lifts low-overlap recall by up to 40
points and compresses a 43-point spread across six models into 13, which is a claim about
what post-processing does to a model's internal geometry and is independent of whether the
corrected embeddings then beat a bag of character n-grams.

## Caveats

- One 50/50 split, no seed variation. PR #130's 5-seed Task B protocol puts the char TF-IDF
  standard deviation at 0.5 points, so the corpus-level ordering is stable, but the
  199-pair strata here are not resampled and their binomial standard errors are around 3.5
  points. The figure's error bars use the pair count rather than the directed count, since
  the two directed observations of a pair share a text and are not independent.
- All p-values are uncorrected. Each stratum carries twelve comparisons against TF-IDF, so
  the two marginal results (low tercile p = 0.023, hard slice p = 0.033) would not survive a
  Bonferroni or Holm correction at that family size. They are reported as they came out, not
  as findings.
- Terciles are cut on the TF-IDF cosine itself, which is also one of the methods under test.
  That favours TF-IDF in the high tercile (where everything saturates anyway) and works
  against it in the low tercile and the hard slice, which is the direction that matters for
  the verdict.
- `phase_resubmit_results.csv` is still the pre-correction evaluation. Every `tau`, layer and
  `D` it holds reproduces exactly on the corrected split, so the configurations used here are
  unaffected, but its own accuracy columns will move slightly when issue #113 re-runs the
  evaluator. Nothing in this note is read from those columns.
- Only `abtt_optimal` and `baseline` are compared. SIF and SIF+ABTT are not in this analysis,
  and neither is the fine-tuned LaTa of issue #123.
- Re-run the single command above whenever the split or PR #130's scorer changes. Issue #131
  already forced one such re-run: it relabels two directories, which adds a positive test
  pair (595 to 596) and renumbers part of the embedding cache.
