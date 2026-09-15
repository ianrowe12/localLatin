# Sensitivity of the chance-corrected deletion AUC gap

Issue #195 (G9), part of epic #109. Companion to
`docs/research/attribution_metrics_decision.md`, which predeclared the setting,
and `docs/research/attribution_v1_resample.md`, which produced the numbers now
in Table 4.

**Status: analysis only. No main-table number and no paper sentence changes in
this PR.** One recommendation is put to Ian at the end; it would weaken a claim
rather than strengthen it.

## Why this run exists

At the 2026-09-14 meeting Prof. Siddique read the second main-table attribution
column, the chance-corrected deletion AUC gap, and asked why its cells sit so
far apart: 0.842 against 0.180 for LaTa/IG, 0.506 against 0.327 for LaTa/MaRC,
0.042 against 0.464 for mT5-base/MaRC. Differences that large, running in both
directions depending on the cell, are either a property of the two embedding
spaces or an artefact of how the metric is assembled. The metric is a
constructed statistic with several free choices behind it, and none of them had
ever been varied.

This memo varies them, over the same stored hidden states the published run
used, and reports what moves. It is a sensitivity analysis. The predeclared
setting is fixed by the earlier memo and is not re-chosen here on the basis of
its result.

## Provenance

| Item | Value |
|---|---|
| Artifacts | `runs/active/ig_examples_200pos_v1/artifacts/` (600 NPZs, benchmark v1) |
| Models and layers | LaTa 7, PhilTa 1, mT5-base 1, `D = 10` |
| Views | IG, retrieval-adapted MaRC (`retrieval_mark`) |
| Backend | `hidden` throughout (representation-level erasure), CPU only |
| Sweep script | `scripts/ig/run_delauc_sensitivity.py` |
| Table generator | `scripts/ig/build_delauc_sensitivity_table.py` |
| Outputs | `runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity/` |
| Configurations | 20 (1 predeclared, 13 one-knob arms, 6 combined) |
| Compute | job `22078824`, `cpu` partition, elapsed 00:30:03, reserved 00:45:00 |

Nothing under `runs/active/ig_examples_200pos_run3_operational/` was read or
written, and no GPU hours were spent.

### The sweep reproduces the published metric exactly

The predeclared configuration is not a second implementation of the metric, it
is the same computation re-expressed with the knobs exposed, and the run checks
that against the published per-pair cache:

```json
{"compared": 2382, "missing_from_cache": 0, "both_undefined": 18,
 "floor_mismatches": 0, "max_abs_diff": 1.78e-15,
 "mean_abs_diff": 4.86e-17, "tolerance": 1e-09, "agrees": true}
```

Every per-pair `del_auc_gap` under the predeclared setting lands on the
published value to floating-point noise, over all 2,382 defined (pair, view,
variant) rows; the 18 rows undefined below the cosine floor are undefined in
both, and no row is defined in one and not the other. The run raises rather
than writing a table if any of that fails. The paired cell statistics follow:

| Cell | base | ABTT | paired ABTT - base | Verdict |
|---|--:|--:|--:|:-:|
| LaTa/IG | 0.814 | 0.180 | -0.634 +/- 0.078 (-8.1 SE) | baseline |
| LaTa/MaRC | 0.486 | 0.327 | -0.159 +/- 0.089 (-1.8 SE) | **tie** |
| PhilTa/IG | 0.113 | 0.400 | +0.287 +/- 0.010 (27.7 SE) | ABTT |
| PhilTa/MaRC | 0.200 | 0.310 | +0.110 +/- 0.014 (7.7 SE) | ABTT |
| mT5-base/IG | 0.062 | 0.561 | +0.500 +/- 0.021 (23.3 SE) | ABTT |
| mT5-base/MaRC | 0.042 | 0.464 | +0.422 +/- 0.023 (18.3 SE) | ABTT |

which is the paired table of `attribution_v1_resample.md` to three decimals.
The per-variant column values differ slightly from Table 4 (0.814 here against
0.842 there for LaTa/IG baseline) for the reason A6 of the decision memo
records: Table 4 averages each variant over its own valid pairs, this memo
averages both variants over the pairs valid under both. A6 already checked that
the choice changes no verdict, and it changes none here either.

## Knob 1 of the question: what knobs are there?

Six, in the current implementation. The predeclared column of this table is the
setting behind Table 4, and is what every sweep arm moves away from.

| Knob | Where it lives | Predeclared setting | Swept values |
|---|---|---|---|
| Deletion step schedule | `attribution_metrics._curve_auc` over the length-`n+1` curve from `PairContext.curves` | every token: k = 0, 1, ..., n | 5%, 10%, 20% of the query per step |
| What a deleted token becomes | `HiddenPairEvaluator.prefix_curves` | dropped from the mean, denominator shrinks to `n-k` | zero vector with the denominator kept at `n`; the corpus mean vector with the denominator kept at `n` |
| Number of random orderings | `DEFAULT_RANDOM_ORDER_DRAWS` | 5 | 1, 20, 50 |
| Seed behind those orderings | `RANDOM_ORDER_SEED` | 20260906 | 20260101, 7 |
| Token filter | `--token_filter`, applied as a pooling mask | `tokenizer_empty`, matching the artifact generator | `all`, `no_empty` (both diagnostic, see below) |
| Side erased | `HiddenPairEvaluator`, which masks the query only | query only, candidate fixed | query and candidate together, matched by fraction |

A seventh choice, the erasure *operator* (representation-level masking of the
cached states against input-level re-encoding with PAD), is not swept here. It
is bounded by A8 of the decision memo, which ran both operators on 20 LaTa/IG
pairs and found `DelAUC gap` gives the same baseline verdict under each, at
-2.8 and -3.8 standard errors. Sweeping it over 600 pairs needs the model
backend and is not CPU-cheap.

Two implementation details that the sweep had to preserve to be comparable:

* **Ranking.** Tokens are ordered by `|a|` with ties broken by token position
  (`rank_order`). Unchanged in every arm.
* **The floor.** The curve is normalised by the full-query cosine, and the
  metric is undefined below `FULL_COS_FLOOR = 0.05`. Unchanged in every arm, so
  cells average 194 to 200 of 200 pairs throughout.

## The sweep

Twenty configurations: the predeclared setting, thirteen arms that move one
knob, and six that combine the three knobs which change the erasure operator
itself. Win/tie/loss counts the six model-view cells under the paper's caption
rule, a tie being a paired difference inside two standard errors of zero.

| Config | Knob | W/T/L | sign wins | base gap range | ABTT gap range |
|---|---|:-:|:-:|---|---|
| **predeclared** | - | **4/1/1** | 4 | 0.04 to 0.81 | 0.18 to 0.56 |
| 5% grid | schedule | 4/1/1 | 4 | 0.04 to 0.81 | 0.18 to 0.56 |
| 10% grid | schedule | 4/0/2 | 4 | 0.04 to 0.80 | 0.17 to 0.55 |
| 20% grid | schedule | 4/0/2 | 4 | 0.03 to 0.74 | 0.16 to 0.52 |
| zero vector | erasure | 3/1/2 | 3 | 0.04 to 0.81 | 0.08 to 0.27 |
| corpus mean vector | erasure | 4/1/1 | 5 | 0.01 to 0.43 | 0.18 to 0.56 |
| 1 draw | draws | 4/1/1 | 4 | 0.04 to 0.76 | 0.18 to 0.56 |
| 20 draws | draws | 4/0/2 | 4 | 0.04 to 0.83 | 0.18 to 0.56 |
| 50 draws | draws | 4/0/2 | 4 | 0.04 to 0.83 | 0.18 to 0.56 |
| seed 20260101 | seed | 4/0/2 | 4 | 0.04 to 0.83 | 0.18 to 0.56 |
| seed 7 | seed | 4/0/2 | 4 | 0.04 to 0.83 | 0.18 to 0.57 |
| no filter (`all`) | token filter | 1/1/4 | 1 | -0.21 to 0.42 | -0.13 to 0.01 |
| `no_empty` | token filter | 6/0/0 | 6 | 0.02 to 0.17 | 0.16 to 0.48 |
| both sides | side | 3/1/2 | 4 | 0.00 to 1.30 | 0.13 to 0.39 |
| 10% + zero | combined | 3/1/2 | 3 | 0.04 to 0.80 | 0.08 to 0.26 |
| 10% + corpus mean | combined | 4/1/1 | 5 | 0.01 to 0.41 | 0.17 to 0.55 |
| 10% + both sides | combined | 3/1/2 | 3 | 0.01 to 1.28 | 0.13 to 0.38 |
| zero + both sides | combined | 3/0/3 | 3 | 0.00 to 1.30 | 0.05 to 0.10 |
| corpus mean + both sides | combined | 4/1/1 | 4 | 0.00 to 0.39 | 0.13 to 0.39 |
| 10% + zero + both sides | combined | 3/0/3 | 3 | 0.01 to 1.28 | 0.05 to 0.09 |

"sign wins" is the published convention (the sign of the cell means, which is
what the `4/6` in the Table 4 caption counts); W/T/L applies the caption's
two-standard-error rule on top of it.

### Cell by cell

Verdicts for the six cells. The two token-filter rows are separated because
they are diagnostics rather than candidate settings.

| Config | LaTa/IG | LaTa/MaRC | PhilTa/IG | PhilTa/MaRC | mT5/IG | mT5/MaRC |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| **predeclared** | base | **tie** | ABTT | ABTT | ABTT | ABTT |
| 5% grid | base | tie | ABTT | ABTT | ABTT | ABTT |
| 10% grid | base | base | ABTT | ABTT | ABTT | ABTT |
| 20% grid | base | base | ABTT | ABTT | ABTT | ABTT |
| zero vector | base | base | ABTT | tie | ABTT | ABTT |
| corpus mean vector | base | tie | ABTT | ABTT | ABTT | ABTT |
| 1 draw | base | tie | ABTT | ABTT | ABTT | ABTT |
| 20 draws | base | base | ABTT | ABTT | ABTT | ABTT |
| 50 draws | base | base | ABTT | ABTT | ABTT | ABTT |
| seed 20260101 | base | base | ABTT | ABTT | ABTT | ABTT |
| seed 7 | base | base | ABTT | ABTT | ABTT | ABTT |
| both sides | base | base | ABTT | tie | ABTT | ABTT |
| 10% + zero | base | base | ABTT | tie | ABTT | ABTT |
| 10% + corpus mean | base | tie | ABTT | ABTT | ABTT | ABTT |
| 10% + both sides | base | base | ABTT | tie | ABTT | ABTT |
| zero + both sides | base | base | ABTT | base | ABTT | ABTT |
| corpus mean + both sides | base | tie | ABTT | ABTT | ABTT | ABTT |
| 10% + zero + both sides | base | base | ABTT | base | ABTT | ABTT |
| *no filter (`all`)* | *base* | *ABTT* | *base* | *tie* | *base* | *base* |
| *`no_empty`* | *ABTT* | *ABTT* | *ABTT* | *ABTT* | *ABTT* | *ABTT* |

**Four of the six cells never move.** LaTa/IG is a baseline win in all eighteen
valid configurations, PhilTa/IG, mT5-base/IG and mT5-base/MaRC are ABTT wins in
all eighteen. LaTa/MaRC and PhilTa/MaRC are the only cells any knob can reach.

### How far each knob moves the number

Shift in the paired ABTT-minus-baseline difference against the predeclared
setting, and the Spearman correlation of the six cell values with theirs:

| Config | max shift | mean shift | rank correlation |
|---|--:|--:|--:|
| 5% grid | 0.007 | 0.003 | 1.000 |
| 10% grid | 0.018 | 0.008 | 1.000 |
| 20% grid | 0.054 | 0.025 | 1.000 |
| zero vector | 0.389 | 0.206 | 0.829 |
| corpus mean vector | 0.387 | 0.156 | 1.000 |
| 1 draw | 0.063 | 0.025 | 1.000 |
| 20 draws | 0.017 | 0.007 | 1.000 |
| 50 draws | 0.019 | 0.008 | 1.000 |
| seed 20260101 | 0.011 | 0.006 | 1.000 |
| seed 7 | 0.016 | 0.007 | 1.000 |
| both sides | 0.509 | 0.259 | 0.943 |
| no filter (`all`) | 0.968 | 0.484 | -0.771 |
| `no_empty` | 0.781 | 0.214 | 0.829 |

## What this says

**1. The step schedule, the draw count and the seed are inert.** A 5% grid
moves no cell's paired difference by more than 0.007 and a 20% grid by no more
than 0.054, on differences that run to 0.63. Changing the seed moves cells by
0.016 at most, and going from 5 to 50 random orderings by 0.019. The cell
ordering is preserved exactly (rank correlation 1.000) in every one of these
arms. Whatever produces the spread Prof. Siddique asked about, it is not the
deletion bookkeeping and it is not Monte-Carlo noise in the reference.

"Inert" here means no verdict, no sign and no ordering moves. It does not mean
the printed numbers are identical: a 0.017 shift is visible at the three
decimals Table 4 prints, and the recommendation section below states exactly
which cells that reaches.

**2. The draw count does move one thing: the standard error, and therefore the
one tie.** LaTa/MaRC is the cell the Table 4 caption names as a tie. Its
paired difference barely moves with the draw count, but its significance
converges:

| Draws | paired diff | SE ratio | verdict |
|---|--:|--:|:-:|
| 1 | -0.096 | -0.93 | tie |
| **5 (published)** | **-0.159** | **-1.78** | **tie** |
| 20 | -0.177 | -2.10 | baseline |
| 50 | -0.178 | -2.20 | baseline |

The reference is a Monte-Carlo estimate, and at five draws its own noise is
still a visible share of the per-pair difference. At twenty draws the estimate
has converged and the cell reads as a small but resolved baseline win.
Section A8 of the decision memo already anticipated this ("the
`--random_order_draws` flag exists and a table-generating run for the paper
should use 20") and the published run did not do it. This is the only place in
the sweep where a valid setting changes a verdict the paper states.

**3. The erasure operator is the one real knob, and each version of it moves
exactly one variant.** The two alternative replacements are each an exact
identity on one side of the comparison, verified per pair over all 600:

| Replacement | baseline cells | ABTT cells |
|---|---|---|
| zero vector, denominator kept | identical to `drop` (max per-pair difference 1.8e-15) | 0.18-0.56 shrinks to 0.08-0.27 |
| corpus mean vector, denominator kept | 0.04-0.81 compresses to 0.01-0.43 | identical to `drop` (max per-pair difference 6.1e-16) |

The algebra is short and worth writing down. Zeroing a deleted token scales the
pooled vector by `(n-k)/n`, and a cosine cannot see a positive rescaling, so
the baseline is blind to it; the ABTT cleaner subtracts the corpus mean *before*
normalising, so the same rescaling changes the cleaned direction. Replacing
with the corpus mean is the mirror image: after centring, the pooled vector
becomes `(n-k)/n` times the centred `drop` vector, so ABTT is blind to it while
the uncleaned baseline is not. **Both halves of the "huge gap" are therefore
partly a scale convention, and the convention favours a different variant
depending on which replacement is chosen.** That is the most substantive
methodological finding here, and it is a property of pairing a cosine read-out
with a centre-and-project cleaner, not of this dataset.

It does not, however, reverse anything. Under `zero` ABTT still wins 3 cells
and ties a fourth; under `corpus mean` it wins 4 and its sign-win count rises
to 5. The predeclared `drop` operator sits between them.

**4. Erasing both sides inflates the baseline, not ABTT.** With the candidate
masked in step with the query, LaTa/IG's baseline gap goes to 1.30 and its
paired difference to -1.14. The chance correction stops bounding the gap by 1
because the random reference curve now falls faster than the attribution one on
both axes at once. It is a defensible metric and it makes ABTT look worse, but
it answers a different question (how faithful is a joint rationale) than the
one the paper asks (how faithful is the query-side attribution).

**5. The token filter dominates everything else, and cannot be varied on these
artifacts.** `all` turns the column into a 1/6 for ABTT and `no_empty` into a
6/6, shifts of up to 0.97 where no other knob reaches 0.51, and `all` even
reverses the cell ordering (rank correlation -0.77). Neither is a legitimate
setting for this run: the generator applied `tokenizer_empty` inside the
pooling and stored unfiltered hidden states, so any other filter pools a
different document vector than the one the ABTT components were fitted on. The
pipeline refuses these arms in normal use (`check_token_filter`), and the sweep
only reaches them by asking for them explicitly. What they establish is the
shape of the problem: the deletion gap is far more sensitive to *which tokens
enter the pooled vector* than to any choice inside the deletion procedure. The
cell spread in Table 4 is a fact about the pooled geometry, not an artefact of
the metric's bookkeeping.

## Is the main-table setting representative?

**Yes, on the win count, and mildly optimistic on the loss count.**

Over the eighteen configurations that pool the vectors the components were
fitted on, ABTT's win count under the two-standard-error rule is 4 in twelve of
them and 3 in the other six; the median and the mode are both 4, and the
predeclared setting gives 4. On the published sign convention the counts are 4
in eleven arms, 3 in five and 5 in two, so the range is 3 to 5 with median 4,
and the predeclared setting again gives 4. The predeclared setting is a median
case, not an outlier, and not the most flattering one available: the
`corpus mean` erasure gives 5 sign wins and the `no_empty` filter would give 6.

The one caveat is the tie. The predeclared setting is the only valid arm
outside `1 draw`, `5% grid` and the two `corpus mean` arms in which LaTa/MaRC
reads as a tie rather than a baseline win, and finding 3 above shows why: at
five draws the reference is noisy enough to hide a -2.2 SE effect. The
predeclared setting therefore reports 4 wins, 1 tie, 1 loss where a
better-estimated reference reports 4 wins and 2 losses.

## Recommendation, for Ian to decide

**No change to the main table in this PR, and no change proposed to the metric
definition.** Five of six knobs are either inert or invalid to vary, and the
one that is neither (the erasure operator) leaves the verdict standing in both
directions.

**One change is worth considering, and it makes the paper's claim weaker.**
Raise `--random_order_draws` from 5 to 20 for the table-generating run, as the
decision memo's own A8 recommended before Table 4 was produced. The
consequences, all measured above:

* **Printed numbers do change.** Seven of the twelve cell means move at three
  decimals: LaTa/IG baseline 0.814 to 0.831, LaTa/MaRC baseline 0.486 to 0.504,
  PhilTa/IG 0.113/0.400 to 0.114/0.403, PhilTa/MaRC 0.200/0.310 to 0.201/0.313,
  mT5-base/MaRC ABTT 0.464 to 0.463. The largest move is 0.017, on the two LaTa
  baseline cells. Table 4 prints the per-variant convention rather than the
  paired one, so its own 0.842 and 0.506 move by the same amounts. The table
  would have to be regenerated, not just re-captioned.
* The caption's random-order reference range is itself a five-draw quantity and
  would be re-derived. At twenty draws it still prints as 0.692 to 0.961, but
  the individual cell references move by up to 0.018 (LaTa baseline 0.697 to
  0.715).
* The caption's tie clause loses `DelAUC gap for LaTa MaRC`, because that cell
  becomes a resolved -2.10 SE baseline win.
* The column therefore reads 4 wins and 2 losses instead of 4 wins, 1 tie and 1
  loss.
* **Nothing qualitative moves.** Every cell keeps its sign and its verdict, and
  the headline `4/6` is unaffected.

Doing it costs one CPU job plus a regeneration of Table 4 and its caption. Not
doing it is also defensible: the published number is the predeclared one, the
caption's tie is an honest statement of what five draws can resolve, and
re-running to move seven printed means by at most 0.017 and convert a tie into
a loss is not a change a reader benefits from mid-resubmission. **This memo
does not make the change. It records the finding and leaves the call to Ian.**

If the change is taken, `docs/research/attribution_metrics_decision.md` A8 and
`docs/research/attribution_v1_resample.md` both need a line saying the
published run used five draws and the table-generating run uses twenty, so the
two numbers in the repo are never confused.

## The appendix table

`overleaf_drafts/tables/attribution_delauc_sensitivity.tex` is generated from
`configs.csv` and `cells.csv` by
`scripts/ig/build_delauc_sensitivity_table.py`. It is **not** `\input` anywhere
yet, and `tests/test_delauc_sensitivity.py` asserts that it is not: whether the
appendix carries it is a separate decision from computing it. If it goes in, it
belongs in Appendix `app:attribution_sweeps`, next to the erasure-operator
caveat that already lives there.

## Reproduce

```bash
sbatch slurm/ig/delauc_sensitivity.sbatch
```

or, on a login node, roughly half an hour single-threaded:

```bash
OMP_NUM_THREADS=1 python scripts/ig/run_delauc_sensitivity.py \
  --run_dir runs/active/ig_examples_200pos_v1 \
  --out_dir runs/active/ig_examples_200pos_v1/attribution_metrics/sensitivity \
  --verify_cache runs/active/ig_examples_200pos_v1/attribution_metrics/v2_hidden

python scripts/ig/build_delauc_sensitivity_table.py
```

`per_pair.csv` (48,000 rows, 7 MB) is gitignored and rebuilt by the sweep;
`configs.csv`, `cells.csv` and `verification.json` are committed.
