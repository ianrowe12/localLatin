# Where do embeddings beat surface overlap?

Issue #132, part of epic #109. Written 2026-09-06 against the split at
`runs/active/resubmit/data/phase_resubmit_split.csv` and the configurations in
`runs/active/resubmit/results/phase_resubmit_results.csv`.

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

**Pairs.** The 595 same-directory pairs among the 858 test files, cut into overlap terciles
of 198 / 198 / 199 pairs. Boundaries: low `< 0.527`, mid `0.527-0.799`, high `> 0.799`;
the full range of positive-pair overlap is `0.051` to `0.982`. A fourth `hard` slice holds
the 121 positives below the TF-IDF Task-A threshold, `tau = 0.417`, i.e. the pairs the
lexical scorer itself would route as new. The hard slice is not a fourth tercile; it sits
inside the low tercile by construction. (PR #130 rescales the score matrix using the train
block, and that block runs from exactly 0.0 to 1.0 here, since some train pair shares no
character n-gram at all and another is an exact duplicate after normalisation. The rescaling
is therefore the identity on this matrix, and the rescaled and raw thresholds coincide.)

**Methods.** The lexical scorer, plus per model the paper's Baseline configuration
(mean-pool, no correction) and the paper's ABTT configuration (mean-pool + `EmbeddingCleaner`
fitted on train, no SIF), each at the layer the headline tables pick, which is the layer of
highest train AUROC. Layers and `D` are read from `phase_resubmit_results.csv` rather than
re-tuned, and the run prints each learned `tau` against the CSV: all twelve match to four
decimals, so these are the paper's configurations and not a re-derivation of them.

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

## Ranking: partner recall@1 by overlap tercile

| Method | low | mid | high | all 595 |
|---|---|---|---|---|
| **char TF-IDF 3-5** | **0.639** | **0.990** | **1.000** | **0.876** |
| LaBSE + ABTT | 0.601 | 0.965 | 1.000 | 0.855 |
| PhilTa + ABTT | 0.540 | 0.970 | 1.000 | 0.837 |
| Qwen3-0.6B + ABTT | 0.495 | 0.955 | 0.997 | 0.816 |
| mT5-base + ABTT | 0.492 | 0.967 | 0.997 | 0.819 |
| KaLM-mini + ABTT | 0.485 | 0.972 | 0.997 | 0.818 |
| LaTa + ABTT | 0.467 | 0.927 | 0.992 | 0.796 |
| LaBSE baseline | 0.528 | 0.904 | 0.985 | 0.806 |
| KaLM-mini baseline | 0.490 | 0.939 | 0.997 | 0.809 |
| Qwen3-0.6B baseline | 0.414 | 0.876 | 0.965 | 0.752 |
| PhilTa baseline | 0.331 | 0.833 | 0.995 | 0.720 |
| LaTa baseline | 0.263 | 0.712 | 0.950 | 0.642 |
| mT5-base baseline | 0.098 | 0.424 | 0.726 | 0.417 |

The predicted division of labour does not appear. The lexical scorer is first in every
tercile, and its margin over the best embedding configuration is *largest* in the low
tercile (3.8 points over LaBSE+ABTT) rather than smallest. Every method converges on the
high tercile, where near-duplicate wording makes the task trivial for anything.

What the terciles do show clearly is ABTT's own effect, which is an internal-geometry result
and is untouched by the lexical comparison: correction lifts the low tercile by 20 points for
LaTa (0.263 to 0.467), 21 for PhilTa and 39 for mT5-base, and it compresses the six models
from a 43-point spread at baseline into a 13-point band. That is the anisotropy-dip story,
and it survives intact.

## The hard slice: 121 positives the lexical scorer would reject

| Method | recall@1 | MRR | above own tau |
|---|---|---|---|
| char TF-IDF 3-5 | **0.430** | **0.516** | 0.000 (by construction) |
| LaBSE + ABTT | 0.380 | 0.447 | **0.248** |
| PhilTa + ABTT | 0.339 | 0.422 | 0.091 |
| LaTa + ABTT | 0.293 | 0.365 | 0.116 |
| KaLM-mini + ABTT | 0.273 | 0.340 | 0.140 |
| Qwen3-0.6B + ABTT | 0.269 | 0.337 | 0.050 |
| mT5-base + ABTT | 0.240 | 0.336 | 0.017 |
| LaBSE baseline | 0.335 | 0.381 | 0.091 |
| KaLM-mini baseline | 0.318 | 0.379 | 0.066 |
| Qwen3-0.6B baseline | 0.244 | 0.314 | 0.066 |
| LaTa baseline | 0.157 | 0.223 | 0.066 |
| PhilTa baseline | 0.149 | 0.215 | 0.000 |
| mT5-base baseline | 0.017 | 0.054 | 0.000 |

This slice is selected as the pairs the lexical scorer scores lowest, so it is stacked
against TF-IDF, and TF-IDF still ranks them best. The `above own tau` column is the one
place an embedding does something the lexical scorer cannot: LaBSE+ABTT puts 25 percent of
these pairs above its own threshold, so they would be routed to their directory rather than
flagged as new, while TF-IDF puts zero above its threshold. That zero is definitional, not
empirical, and quoting it as a 25-point win would be dishonest. The honest version of the
same comparison is the low tercile, which is defined without reference to any threshold:
there LaBSE+ABTT routes 0.429 of the positives above its `tau` against 0.389 for TF-IDF, a
4-point edge on 198 pairs, inside the roughly 3.5-point binomial standard error of either
estimate. Every other model's ABTT configuration is below TF-IDF on that measure
(KaLM-mini 0.313, PhilTa 0.288, LaTa 0.258, Qwen3-0.6B 0.167, mT5-base 0.106).

## Reverse slice: negatives with high surface overlap

The 36,706 different-directory test pairs in the top decile of TF-IDF overlap (cosine
`>= 0.148`), giving 47,775 directed observations after dropping endpoints with no test
partner. A method is confused when it scores the impostor above every true partner of that
query.

| Model | baseline | ABTT |
|---|---|---|
| char TF-IDF 3-5 | 0.0056 | |
| LaBSE | 0.0170 | **0.0047** |
| PhilTa | 0.0187 | 0.0047 |
| LaTa | 0.0144 | 0.0048 |
| mT5-base | 0.0549 | 0.0062 |
| Qwen3-0.6B | 0.0115 | 0.0065 |
| KaLM-mini | 0.0135 | 0.0068 |

Near-duplicate wording across genuinely different fragments is a small problem for everyone.
Uncorrected embeddings are the most vulnerable (mT5-base gets it wrong ten times as often as
TF-IDF), ABTT removes most of that gap, and the three corrected models at 0.0047 sit
marginally below TF-IDF's 0.0056. The differences here are real in the sense that 47,775
observations resolve 0.001, but 0.0009 is about 43 observations and should not be built into
a claim.

## Verdict

**Redundant, not complementary.** On this corpus a character 3-5-gram TF-IDF cosine ranks
the true partner better than every embedding configuration in every overlap stratum,
including the low-overlap tercile and the deliberately adversarial slice of positives the
lexical scorer itself scores below its own threshold, so the hoped-for division of labour
where embeddings recover the semantically related but lexically drifted witnesses is not
present in the data. The one measurable exception is routing rather than ranking, and it is
one model: LaBSE with ABTT carries 43 percent of the lowest-overlap positives above its
decision threshold against TF-IDF's 39 percent, and puts a quarter of the hard slice above
threshold where TF-IDF by construction puts none. That is a 4-point edge on 198 pairs from
one of six models, which is a hypothesis, not a result. The corrected models also reject
high-overlap impostors slightly more often than TF-IDF (0.0047 against 0.0056), by a margin
of the same negligible size. The retrieval framing in F6 and F8 should therefore not claim
that embeddings add retrieval signal beyond surface overlap on this corpus, and should
present the lexical baseline as the operating point a practitioner would actually pick. What
the stratification does support, and support strongly, is the paper's actual subject: ABTT
lifts low-overlap recall by up to 39 points and compresses a 43-point spread across six
models into 13, which is a claim about what post-processing does to a model's internal
geometry and is independent of whether the corrected embeddings then beat a bag of character
n-grams.

## Caveats

- One 50/50 split, no seed variation. PR #130's 5-seed Task B protocol puts the char TF-IDF
  standard deviation at 0.6 points, so the corpus-level ordering is stable, but the
  198-pair strata here are not resampled and their standard errors are around 3.5 points.
- Terciles are cut on the TF-IDF cosine itself, which is also one of the methods under test.
  That favours TF-IDF in the high tercile (where everything saturates anyway) and works
  against it in the low tercile and the hard slice, which is the direction that matters for
  the verdict.
- Only `abtt_optimal` and `baseline` are compared. SIF and SIF+ABTT are not in this analysis,
  and neither is the fine-tuned LaTa of issue #123, which had not landed when this ran.
- Issue #112 relabels two directories, which changes the positive-pair set. Re-run the single
  command above afterwards.
