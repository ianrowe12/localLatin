# Reframe experiment E2: token audit (issue #252)

Results for James's experiment E2 of the analysis reframe
(`reframe_handoff_20260927.md`), and the paper edits that put them into
`overleaf_drafts/acl_latex.tex`. Every number below is in
`runs/active/reframe/e2/facts_e2.md` (token audit) or
`runs/active/reframe/e2/facts_e2_zeroing.md` (zeroing follow-up), both
generated, or in a CSV beside them; the few numbers read from a CSV and not
from a facts file are marked "(CSV)". The facts files were read at commit
`cee4e8c` with a clean working tree.

Short version:

- **LaTa: the token account holds.** The token mix explains a median 0.985 of
  the PC1 score of test passages at the 10 collapsed layers (random directions
  0.18). At layer 6 the comma alone holds 0.962 of PC1's score variance over
  training passages. Mean pooling without three token types (comma, period,
  `</s>`) restores all 10 layers; frequency-matched random types restore none.
- **PhilTa: same leading token types, context-dependent values.** The token
  ablation restores all 9 collapsed layers, but 8 of them need the top 10 to 30
  types. The token-mix rule fails at 0 of 9 (median EV 0.42), and neither
  single-factor pooling arm rescues any layer: the expectation "SIF weights
  with special tokens kept rescue" is not met.
- **mT5-base: the account fails.** Token-mix EV 0.066, group shares close to
  the groups' token mass, token ablation restores 0 of 7, top-PC share 1.000
  under every pooling arm. The expectation that the tokens SIF keeps carry the
  direction is not met.
- **Length account: fails at 0 of 26.** LaTa's negative raw gap is not a
  length effect and stays unexplained.
- **Zeroing follow-up (post hoc, belongs with E1).** After the ten top
  coordinates are zeroed, the dominant direction is what is left of PC1
  outside them (remainder cosine 0.999). The surviving PC2 and PC3 are not why
  zeroing fails.
- **One gate failed.** The SIF arm equals the repository CLI's SIF pooling on
  the tracked split exactly, but not the published `sif_only` cells (up to
  0.015 AUROC in LaTa). The planned consequence was changed after the results
  were read; see "The SIF gate" and deviation 4. James must accept or reject
  that change in review.

## Provenance

| Item | Value |
|---|---|
| Scripts | `scripts/paper/reframe/e2_token_audit.py` (subcommands `audit`, `check`, `render`); `scripts/paper/reframe/e2_zeroing_followup.py` (`compute`, `check`, `render`) |
| Batch files | `slurm/reframe/reframe_e2.sbatch` (one GPU, 8 cores, `--time=00:55:00`); `slurm/reframe/reframe_e2_zeroing.sbatch` (CPU, 16 cores, `--time=00:20:00`) |
| Zeroing follow-up job | 22597130, `cpu-interactive`, 8 min 42 s, exit 0, code at `b84cd36` |
| Audit jobs | 22597913 (LaTa, PhilTa, mT5-base, LaBSE; 11 min 46 s) and 22598253 (Qwen3-0.6B, KaLM-mini; 13 min 2 s), `gpuA40x4-interactive`, code at `6628352`. Both exited 3, the gate code (CSVs written, gate 2a failed); neither crashed |
| Smoke job | 22597799, `gpuA40x4-interactive`, 4 min 56 s, exit 0: `LIMIT=64`, LaTa and Qwen3-0.6B, gates skipped, written to `runs/active/reframe/e2/smoke_limit64/` |
| SIF diagnostic jobs | 22598639 (11 min 2 s) and 22599010 (13 min 50 s), `gpuA40x4-interactive`; script and logs in `/projects/bimc/swong2/setup/e2_sif_diag/` (outside the repository) |
| Replication job | 22599178, `gpuA40x4-interactive`, 4 min 52 s, exit 0; `/projects/bimc/swong2/setup/e2_replicate.py`, log `/projects/bimc/swong2/setup/logs/e2_replicate_22599178.out` |
| Account | `beto-delta-gpu` / `beto-delta-cpu` for every job |
| Inputs | `runs/active/resubmit/data/phase_resubmit_split.csv`; the six models from the HuggingFace cache (transformers 4.57.6, offline); cached mean-pooled vectors `runs/active/resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/` and, for five models, SIF-pooled vectors `hidden_sif_tokempty/` (James's re-extraction; none for KaLM-mini); `phase_resubmit_results.csv` for the published cells |
| Outputs | `runs/active/reframe/e2/`: `e2_pooling_arms.csv` (500 rows: 5 arms at 100 model-layers), `e2_direction_audit.csv` (8,800 rows), `e2_carriers.csv` (9,000 rows: top 30 token types per PC), `e2_token_ablation.csv` (6,000 rows), `e2_length.csv`, `e2_gate_check.csv`, `facts_e2.md`; `e2_zeroing_followup.csv` (800 rows), `e2_zeroing_gate_check.csv`, `facts_e2_zeroing.md`; `overleaf_drafts/tables/e2_token_audit.tex` |
| Tests | `tests/test_e2_token_audit.py`, `tests/test_e2_zeroing_followup.py` |

Regenerate the table and the facts files from the committed CSVs with
`python scripts/paper/reframe/e2_token_audit.py render` and
`python scripts/paper/reframe/e2_zeroing_followup.py render` (no GPU, no
embeddings).

### Design, as revised after E1

The handoff's E2 row audits which tokens carry the top coordinates. E1 then
showed that zeroing up to ten coordinates restores none of the 26 collapsed T5
layers while removing three principal components restores all 26. The design
was revised after E1 and before any E2 number was read: the audit targets the
directions the projection removes (PC1 to PC3 of the training pooled vectors),
and keeps the three top-variance coordinates as a secondary readout. The ratio
r and the Qwen3-0.6B r test left E2 because E1 reports them.

### Protocol

- Six models, every layer: LaTa, PhilTa, mT5-base and LaBSE 12 layers each,
  Qwen3-0.6B 28, KaLM-mini 24 (100 model-layers). 847 train and 858 test
  passages. LaBSE, Qwen3-0.6B and KaLM-mini are the healthy contrast.
- Forward pass: each model loaded and tokenized as its extraction CLI does
  (max length 512, batch size 8, `tokenizer_empty` keep lookup, batches in the
  cache's row order), one forward with all hidden states.
- Directions are fit on the cached training vectors: the training mean, PC1 to
  PC3 (the components ABTT removes), the three coordinates of largest training
  variance, and 20 random unit directions (seed 233) orthogonal to the top 10
  PCs and to each other.
- **Pooling control.** Five arms, all with the CLI's pooling expression:
  `mean` (the cache); `mean_nospecial` (special tokens dropped);
  `sif_keepspecial` (SIF weights a / (a + p) on token types with a training
  probability, weight 1 on special tokens); `sif` (the CLI's SIF: frequency
  weights and zero weight on special tokens); `mean_nofreq100` (the 100 most
  frequent training token types dropped, special tokens kept).
- **Token audit.** The contribution of token t is c_t = (h_t - mu) . w; a
  passage's score is the mean of c_t over its kept tokens. Token-mix EV: one
  mean of c_t per token type fit on training tokens (a type unseen in training
  gets the mean over all training tokens), the passage's predicted score is
  the mean of its tokens' type means, EV = 1 - Var(s - s_hat) / Var(s) over
  the passages of a split. Group shares (special, 100 most frequent, other)
  are Cov(s_group, s) / Var(s) and sum to 1. Carriers are token types ranked
  by that share on training passages. The same quantities on the 20 random
  directions are the matched control; span(PC2, PC3) and span(PC1, PC2, PC3)
  get joint readouts that do not depend on the basis inside the subspace.
- **Token ablation.** Token types ranked on training passages by PC1 share
  (`pc1`) and by the mean of the PC1, PC2 and PC3 shares (`pc123`); mean
  pooling without the top m types, m in {1, 3, 10, 30, 100}. Control: five
  draws of random token types matched in training count.
- **Length.** Token count n under each model's tokenizer; Spearman of the PC1
  score with log n; mean |delta log n| over the same- and different-directory
  pairs that Task A scores.
- **Zeroing follow-up** (CPU, cached vectors only): for each of E1's zeroed
  coordinate sets, the loading of the original PCs on the zeroed coordinates,
  the variance left along them, angles and the remainder cosine between old
  and new components, score correlations, and intervention cells (zeroing
  followed by ABTT; original PCs 2 and 3 removed with PC1 kept).

## Frozen rules and expectations

Fixed after E1 and before any E2 number was read (constants at the top of the
script; facts file, "Frozen decision rules").

| Rule | Statement |
|---|---|
| Collapsed | published baseline test AUROC < 0.70 at a T5 layer (26 layers: LaTa 2 to 11, PhilTa 3 to 11, mT5-base 5 to 11) |
| R1 rescue | an arm rescues a collapsed layer if (AUROC_arm - AUROC_mean) / (AUROC_sif - AUROC_mean) >= 0.80, evaluated only where AUROC_sif - AUROC_mean >= 0.05; elsewhere "no SIF gain to recover" and the raw change |
| R2 | the token mix carries the direction at a layer if test EV for PC1 >= 0.5 |
| R3 | token ablation restores a collapsed layer if test AUROC >= 0.90 for some m <= 100, under either ranking |
| R4 | the length account holds if \|Spearman(s_PC1, log n)\| >= 0.5 on test and mean \|delta log n\| is larger for same-directory than for different-directory pairs |

Expectations recorded before the run: in LaTa and PhilTa frequent tokens carry
the direction (the `frequent` group has the largest share; `sif_keepspecial`
rescues, `mean_nospecial` does not); in mT5-base no arm rescues and the tokens
SIF keeps (`other`) carry it. The subspace readouts carry no rule.

## Gates

`e2_gate_check.csv`, facts section 0. Per model, largest difference:

| Gate | Reference | Tolerance | Outcome |
|---|---|---|---|
| 1a `mean` arm AUROC | published baseline cells | 1e-6 | PASS at all six models (largest 2.31e-7, Qwen3-0.6B) |
| 1b `mean` arm vectors | cached `hidden_mean_tokempty` | 1e-3 relative L2 | PASS, difference 0 at all six |
| 2a `sif` arm AUROC | published `sif_only` cells | 1e-6 | **FAIL at all six**: LaTa 1.50e-2, PhilTa 6.61e-3, mT5-base 1.06e-3, LaBSE 2.51e-4, Qwen3-0.6B 5.86e-4, KaLM-mini 9.47e-4 |
| 2b `sif` arm vectors | cached `hidden_sif_tokempty` (reported, not gated) | 1e-3 | difference 0 at the five models with a cache; no cache for KaLM-mini |
| 3 score identity | passage score from tokens against w . (pooled - mu) | 1e-9 relative | PASS at all six |
| 4 group shares sum to 1 | | 1e-8 | PASS at all six |

Also in section 0: the top three variance coordinates equal those of
`e1_top_coordinates.csv` at 100 of 100 model-layers; the pooling expression
equals the extraction CLI's own function on the first batch of each model;
two passages have no token under `mean_nospecial` and `sif` (under
`mean_nospecial` they fall back to the mean vector, under `sif` never, as in
the CLI: the one such test passage is a zero vector with cosine 0 to every
other, and leaving it out would move LaTa layer 6 from 0.8782 to 0.8779
according to the replication log).

Zeroing follow-up (`e2_zeroing_gate_check.csv`, its facts section 0): the
zeroed coordinate sets, the zero-only AUROC and the top-PC share of the zeroed
vectors equal E1's (difference 0), and base and ABTT D = 1, 2, 3, 10 AUROC
equal H1's within 5.4e-7. All pass.

## The SIF gate

Gate 2a compares the `sif` arm with the published `sif_only` cells that
Section 4 of the paper quotes. It fails at every model. Two facts bound the
failure:

1. The `sif` arm is the repository CLI's SIF pooling on the tracked split with
   training-only token probabilities. It equals James's re-extracted
   `hidden_sif_tokempty` vectors exactly (gate 2b), and the replication's
   independent pooling reproduces its AUROC (below). The local SIF caches had
   not been rescored against the published cells before this experiment.
2. The forward pass is not the cause: the `mean` arm reproduces the published
   baseline to 2.3e-7. What differs is the SIF weighting, most likely its
   token probabilities.

Two diagnostic jobs recomputed the CLI's SIF pooling for the three T5 encoders
under 13 token-probability sources and scored each against the published
cells. Largest |AUROC - published| over the 12 layers (logs
`sif_diag_22598639.out` and `sif_diag_22599010.out`; `train`, `train_nokeep`
and `batch8` appear in both with the same values):

| Source | Job | LaTa | PhilTa | mT5-base |
|---|---|---|---|---|
| `train` (training files, empty-token filter; the `sif` arm) | both | 1.50e-2 | 6.61e-3 | 1.06e-3 |
| `all_labelled` | 22598639 | 1.25e-2 | 5.92e-3 | 1.01e-3 |
| `test` | 22598639 | 1.02e-2 | 5.31e-3 | 1.34e-3 |
| `labelled_plus_unlabelled` | 22598639 | 1.12e-3 | 1.57e-3 | 2.87e-3 |
| `unlabelled` | 22598639 | 7.48e-3 | 2.03e-3 | 4.00e-3 |
| `train_nokeep` (training files, no empty-token filter in the count) | both | 1.21e-3 | 1.11e-3 | 8.33e-4 |
| `all_nokeep` (all labelled files, no filter) | 22599010 | 1.86e-3 | 1.94e-3 | 5.79e-4 |
| `labunlab_nokeep` (labelled and unlabelled, no filter) | 22599010 | 1.51e-2 | 7.37e-3 | 3.11e-3 |
| `p9train` (training files of `runs/phase9/phase9_split.csv`) | 22599010 | 2.65e-2 | 1.10e-2 | 3.15e-3 |
| `p9train_nokeep` | 22599010 | 1.56e-2 | 4.96e-3 | 2.63e-3 |
| `p9all` (all files of that split) | 22599010 | 1.78e-2 | 8.04e-3 | 1.90e-3 |
| `p9all_nokeep` | 22599010 | 4.55e-3 | 1.08e-3 | 1.53e-3 |
| `batch8` (probabilities from each batch of 8) | both | 3.46e-1 | 2.87e-1 | 1.07e-1 |

The definitions of the last eight rows are those of `sif_diag.py` as saved;
the first job ran an earlier version of the script that was overwritten, so
`all_labelled`, `test`, `labelled_plus_unlabelled` and `unlabelled` are read
from their column names.

No source reproduces the published cells to 1e-6: the smallest difference in
the table is 5.79e-4. No single source is closest for all three models
(`labelled_plus_unlabelled` for LaTa, `p9all_nokeep` for PhilTa, `all_nokeep`
for mT5-base). `train_nokeep`, training files counted without the empty-token
filter, has the smallest worst case: within 1.21e-3, 1.11e-3 and 8.33e-4.

**What was done instead of blocking.** The plan said a failed gate 2 blocks
the pooling-control conclusions. After the results were read, `render` was
changed (commit `cee4e8c`) to report R1 under both references: against the
job's own `sif` arm, and against the published `sif_only` cell as AUROC_sif.
The gate, its tolerance, the rules and every audit number are unchanged. Under
the published reference (facts section 3, last subsection):

- the `sif` arm is higher than the published cell at every collapsed layer:
  LaTa median +1.22e-2 (+7.43e-4 to +1.50e-2), PhilTa +4.79e-3, mT5-base
  +7.23e-4;
- all 78 R1 verdicts (26 collapsed layers, 3 arms) are the same;
- the largest change of the recovered share of the SIF gain is 0.049 (LaTa
  layer 8, `mean_nofreq100`), over the 57 cells where the share is evaluated.

**Open question for the first author.** Which token-probability source
produced the published `sif_only` cells of `phase_resubmit_results.csv`? The
tracked split, the repository CLI and the arguments of
`slurm/resubmit/resubmit_extract_*.sbatch` do not reproduce them to 1e-6 in
this checkout (Ian's cache is not readable from it). Until that is known, the
SIF numbers of Section 4, Figure 1 and the SIF appendix tables rest on cells
that cannot be regenerated here.

## Independent replication

`/projects/bimc/swong2/setup/e2_replicate.py` (job 22599178, 288 s) was
written from the definitions in the facts file without reading or importing
`e2_token_audit.py`. It uses the repository's basic library code (text
loading, keep lookup, token probabilities, `EmbeddingCleaner`, the alignment
resolver) and computes every metric itself. It recomputed 32 cells at LaTa
layer 6, PhilTa layer 10 and mT5-base layer 5; all 32 agree with the committed
CSVs (AUROC within 1e-6, audit quantities within 1e-4).

| Cells | Quantity | LaTa L6 | PhilTa L10 | mT5-base L5 |
|---|---|---|---|---|
| Pooling arms | AUROC `mean` (cache) | 0.4957 | 0.5380 | 0.6537 |
| | AUROC `mean_nospecial` | 0.4931 | 0.4959 | 0.6535 |
| | AUROC `mean_nofreq100` | 0.9036 | 0.6917 | 0.7258 |
| | AUROC `sif` | 0.8782 | 0.8793 | 0.6728 |
| Token audit of PC1 | token-mix EV, test | 0.9840 | 0.4387 | 0.0660 |
| | share special / frequent / other | 0.011 / 0.969 / 0.020 | 0.225 / 0.753 / 0.022 | -0.000 / 0.530 / 0.470 |
| Ablation | top 3 types of the `pc123` ranking | comma, period, `</s>` | `</s>`, comma, period | `</s>`, period, comma |
| | AUROC with them dropped | 0.9198 | 0.8551 | 0.6918 |
| Carrier | top PC1 carrier and its training share | comma, 0.9616 | | |

Two info lines (not counted) differ: a variant that gives an unseen token type
the mean of the type means instead of the mean over all training tokens moves
the test EV by 2.8e-3 (LaTa) and 1.5e-3 (PhilTa). The SIF diagnostic script's
`train` column is a third computation of the `sif` arm: its largest
differences from the published cells equal gate 2a's (1.50e-2, 6.61e-3,
1.06e-3), and for example LaTa layer 6 gives 0.878186 in both.

## Findings

Verdicts are those of facts sections 3 to 8 and 11. "Median (min to max)" is
over the collapsed layers of the model.

### R1, pooling control: rescue in LaTa only

AUROC by arm at the collapsed layers:

| Model | `mean` | `mean_nospecial` | `sif_keepspecial` | `sif` | `mean_nofreq100` |
|---|---|---|---|---|---|
| LaTa (10) | 0.502 | 0.498 | 0.861 | 0.878 (CSV) | 0.901 |
| PhilTa (9) | 0.542 | 0.500 | 0.653 | 0.897 (CSV) | 0.667 |
| mT5-base (7) | 0.657 | 0.657 | 0.683 | 0.682 (CSV) | 0.715 |

| Model | SIF gain | `mean_nospecial` rescues | `sif_keepspecial` rescues | `mean_nofreq100` rescues |
|---|---|---|---|---|
| LaTa | +0.375 (+0.355 to +0.411) | 0 of 10 (share of the gain -0.01) | 10 of 10 (0.96; 0.95 to 1.00) | 10 of 10 (1.06; 1.00 to 1.11) |
| PhilTa | +0.350 (+0.328 to +0.368) | 0 of 9 (-0.12) | 0 of 9 (0.31; 0.16 to 0.43) | 0 of 9 (0.35; 0.20 to 0.47) |
| mT5-base | +0.026 (+0.019 to +0.031) | not evaluable | not evaluable | not evaluable |

- LaTa: all three expectations met (frequent group largest at 10 of 10;
  `sif_keepspecial` rescues 10 of 10; `mean_nospecial` rescues 0 of 10).
- PhilTa: "`sif_keepspecial` rescues" is **NOT MET (0 of 9)**. Neither
  single-factor arm rescues; only full SIF does. "`mean_nospecial` does not
  rescue" is met (0 of 9; it lowers AUROC by a median 0.041).
- mT5-base: R1 is **not evaluable** under the frozen 0.05 floor, because SIF
  gains at most 0.031. Raw changes against mean pooling: `mean_nospecial`
  -0.000, `sif_keepspecial` +0.027 (+0.021 to +0.034), `mean_nofreq100` +0.057
  (+0.055 to +0.072), `sif` +0.026. Top-PC share of the training vectors stays
  at 1.000 under every arm (at least 0.9995 at all 7 layers, CSV); in LaTa its
  median falls from 0.941 to 0.153 under `sif`, in PhilTa from 0.845 to 0.121
  (CSV).
- **Caveat on `sif_keepspecial`.** The arm gives special tokens weight 1 while
  frequent tokens get much less, so the special-token share of the pooling
  weight is about double that of mean pooling (median 0.040 against 0.020;
  `mean_nofreq100` 0.039). It is full SIF with only the zero weight on special
  tokens removed, not mean pooling with only frequency weights added. PhilTa's
  failure under it therefore shows that neither change alone suffices; it does
  not show how much each contributes.

### R2, token mix: holds at 10 of 26 (LaTa 10 of 10, PhilTa 0 of 9, mT5-base 0 of 7)

| Model | PC1 test EV | Random directions, mean of 20 | PC2 test EV | PC3 test EV |
|---|---|---|---|---|
| LaTa | 0.985 (0.866 to 0.989) | 0.177 (0.119 to 0.401) | -0.113 | -0.119 (-1.052 to 0.658) |
| PhilTa | 0.424 (-0.968 to 0.444) | 0.200 (0.047 to 0.365) | -0.581 | -2.769 |
| mT5-base | 0.066 (0.066 to 0.067) | 0.160 (0.018 to 0.239) | 0.048 | 0.505 (0.400 to 0.656) |

- LaTa: R2 holds for PC1 only. One mean per token type does not predict the
  PC2 and PC3 scores (EV below 0), although one token type holds most of
  their variance share (carriers below).
- PhilTa: PC1 EV is above the random-direction mean at 8 of 9 layers but below
  0.5 at all 9. The contribution of `</s>` changes sign across passages (layer
  10: mean c -1.106e5 against mean |c| 1.561e5), so one mean per token type
  cannot predict it. The right reading is that the carrier's value depends on
  context, not that tokens do not carry PC1: the group shares and the ablation
  say they do.
- mT5-base: EV 0.066 is below the random-direction mean at 6 of 7 layers (at
  layer 7 the random mean is 0.018). PC3 has EV 0.40 to 0.66, but PC3 holds
  0.000 of the training variance there.

### Token groups and carriers

Share of the PC1 score variance of test passages by group, with the groups'
token mass for scale:

| Model | special | frequent | other | Mass: special / frequent / other |
|---|---|---|---|---|
| LaTa | 0.010 | 0.969 (0.968 to 0.970) | 0.020 | 0.021 / 0.482 / 0.497 |
| PhilTa | 0.235 (0.225 to 0.515) | 0.740 (0.455 to 0.753) | 0.022 | 0.020 / 0.499 / 0.481 |
| mT5-base | -0.000 | 0.530 | 0.470 | 0.015 / 0.482 / 0.503 |
| Random directions, all 26 | 0.023 | 0.430 | 0.546 | |

- LaTa: frequent largest at 10 of 10 (MET).
- PhilTa: frequent largest at 8 of 9 (**NOT MET**); at layer 3 the special
  group leads (0.515 against 0.455).
- mT5-base: `other` largest at 0 of 7 (**NOT MET**). The shares equal the
  groups' token mass to within 0.05; a random direction there gives 0.003,
  0.467 and 0.531.

Carriers at the worst layer (share of the score variance over training
passages; facts section 9):

| Model, layer | PC1 | PC2 | PC3 |
|---|---|---|---|
| LaTa 6 | comma 0.962, `</s>` 0.014, `·` 0.007 | period 0.916, `</s>` 0.034, comma 0.029 | comma 0.897, `</s>` 0.036, period 0.016 |
| PhilTa 10 | comma 0.562, `</s>` 0.260, period 0.151 | `</s>` 0.372, period 0.281, comma 0.233 | `</s>` 0.719, comma 0.191, period 0.023 |
| mT5-base 5 | period 0.082, `s` 0.033, `i` 0.031 | period 0.323, comma 0.104, `u` 0.059 | `</s>` 0.468, comma 0.018, period 0.017 |

- LaTa: the comma's contribution to PC1 has one sign (mean c -4.727e4, mean
  |c| 4.727e4). Its PC1 share is 0.957 to 0.966 at all 10 collapsed layers
  (CSV). The period leads PC2 from layer 4 on; at layers 2 and 3 the comma
  does (CSV).
- mT5-base: no token type holds more than 0.082 of PC1, and the 30 listed
  types sum to 0.585. A few rare types have very large contributions per
  token (`▁inser`, 6 training occurrences, mean c 2.198e5; `▁depo`, 22
  occurrences, 3.509e4; against a mean |c| of 1.257e4 for the period), with
  shares of 0.025 and 0.026.

### R3, token ablation: restores 19 of 26 (LaTa 10 of 10, PhilTa 9 of 9, mT5-base 0 of 7)

The matched random control reaches 0.90 at 0 of 26.

| Model | Ranking | Restored | Smallest m | AUROC m=1 | m=3 | m=10 | m=30 | m=100 | Control |
|---|---|---|---|---|---|---|---|---|---|
| LaTa | `pc1` | 2 of 10 | 1 (1 to 1) | 0.735 | 0.696 | 0.686 | 0.665 | 0.650 | 0.50 |
| LaTa | `pc123` | 10 of 10 | 3 (1 to 3) | 0.724 | 0.919 | 0.921 | 0.926 | 0.924 | 0.50 |
| PhilTa | `pc1` | 3 of 9 | 10 (3 to 100) | 0.671 | 0.873 | 0.889 | 0.891 | 0.892 | 0.54 |
| PhilTa | `pc123` | 9 of 9 | 10 (3 to 30) | 0.500 | 0.873 | 0.910 | 0.925 | 0.928 | 0.54 |
| mT5-base | `pc1` | 0 of 7 | | 0.667 | 0.675 | 0.678 | 0.721 | 0.719 | 0.656 |
| mT5-base | `pc123` | 0 of 7 | | 0.667 | 0.694 | 0.710 | 0.726 | 0.734 | 0.656 |

- LaTa: the three types dropped at m = 3 under `pc123` are the comma, the
  period and `</s>` at all 10 layers; AUROC 0.912 to 0.929 (CSV). They are
  0.113 of the test tokens (CSV). Ranking by PC1 share alone restores only
  layers 2 and 3, which is consistent with the comma leading PC1 and the
  period PC2 from layer 4 on. This parallels H1, where D = 1 restores 1 of 26
  and D = 3 all 26.
- PhilTa: dropping the top three types leaves AUROC at 0.847 to 0.904 (1 of 9
  restored, CSV). Layer 3 is restored at m = 3, layers 4 to 10 at m = 10,
  layer 11 at m = 30. Dropping `</s>` alone lowers AUROC to 0.50, as
  `mean_nospecial` does.
- mT5-base: best AUROC over rankings and m is 0.724 to 0.748, with 0.357 of
  the test tokens dropped at m = 100. Top-PC share stays at 1.000 (CSV).

### R4, length: holds at 0 of 26

- |Spearman(s_PC1, log n)| >= 0.5 at 0 of 26. LaTa -0.384 (-0.437 to +0.387),
  PhilTa -0.104, mT5-base -0.274. The sign of a component is fixed by its
  largest loading, so the sign of rho is not comparable across layers.
- Same-directory pairs differ less in log length than different-directory
  pairs in every model: on test pairs 0.322 against 0.779 (LaTa), 0.319
  against 0.772 (PhilTa), 0.326 against 0.771 (mT5-base). Length difference
  alone ranks pairs at AUROC 0.75 on test.
- LaTa's negative raw gap is therefore not a length effect. It stays
  unexplained.

### Healthy contrast

- Qwen3-0.6B layer 1: PC1 token-mix EV 0.987, carried by special tokens (share
  0.744) and almost entirely by the first token (0.693). AUROC is 0.858 and
  nothing collapses, and PC1 holds 0.100 of the training variance. A direction
  that the token mix explains is harmless when it holds little variance.
- R2 holds at 13 of 28 Qwen3-0.6B layers and 14 of 24 KaLM-mini layers, and at
  0 of 12 LaBSE layers. Random directions also reach high EV in these models
  (mean 0.710 at Qwen3-0.6B layer 1, 0.731 at LaBSE layer 1, against PC1 EV
  0.358 there). R2 therefore has to be read against its random-direction
  control and together with the component's variance share; alone it does not
  mark a collapse.
- At the worst layers of the three embedding-trained models no arm changes
  AUROC by more than 0.012, and the token ablation gains at most 0.027.
- Limit of the contrast: it shows that the audit's statistics do not fire
  spuriously as a collapse signal; it does not test why the T5 token types
  take large values.

### Zeroing follow-up (post hoc; reported with E1)

At the 26 collapsed layers, k = 10, ranking by variance, median (min to max):

| Measure | Value |
|---|---|
| Squared loading on the ten zeroed coordinates | PC1 0.918 (0.759 to 0.997), PC2 0.495 (0.271 to 0.726), PC3 0.178 (0.065 to 0.398) |
| Variance left along the original PC1 axis | 0.007 (0.000 to 0.060) |
| Remainder cosine of PC1 (new top direction against what is left of PC1 outside the ten coordinates) | 0.999 (0.972 to 1.000); above 0.9 at 26 of 26 |
| Share of the remaining variance along that remainder | 0.717 (0.322 to 0.995) |
| Score correlation, old against new top component, test passages | \|Pearson\| 0.991 (above 0.9 at 26 of 26); \|Spearman\| 0.867 (0.311 to 0.986; mT5-base 0.449) |
| Angle between old and new PC1 | 73.3 degrees (below 30 at 0 of 26) |

Intervention cells (added post hoc, no prediction), layers restored to AUROC
>= 0.90 of 26:

| Intervention | D = 1 | D = 2 | D = 3 |
|---|---|---|---|
| Zero ten coordinates, then ABTT fit on the zeroed vectors | 9 | 21 | 26 |
| ABTT on the unzeroed vectors (H1) | 1 | 13 | 26 |

Removing original PCs 2 and 3 while keeping PC1 restores 0 of 26 (median
AUROC 0.518). So the surviving PC2 and PC3 are not why zeroing fails: the new
top direction is the old PC1 with its ten largest entries removed, and
passages score on it almost as on the old one. The passage variable behind PC1
is also written on the other coordinates. With E2 this reconciles E1: the
nuisance in LaTa is carried by a few token types but is not localized on a few
coordinates.

## Deviations

1. **Design revised after E1.** The handoff row audits coordinates; the run
   audits PC1 to PC3 and keeps the coordinates as a secondary readout. Revised
   before any E2 number was read. The ratio r and the Qwen3-0.6B r test moved
   to E1.
2. **Additions requested by James before the run.** KaLM-mini; the joint
   subspace readouts for span(PC2, PC3) and span(PC1, PC2, PC3); the matched
   random controls (20 random directions, frequency-matched random token
   types); the extra measures of the zeroing follow-up. The
   `mean_nofreq100` arm and the two-ranking token ablation are part of the
   revised design.
3. **Zeroing follow-up is post hoc as a whole**, and its intervention cells
   (zeroing followed by ABTT; PCs 2 and 3 removed with PC1 kept) were added
   after its measures were written. They carry no prediction and no verdict.
4. **The consequence of a failed gate 2 was changed after the results were
   read.** Planned: a failed SIF gate blocks the pooling-control conclusions.
   Done: R1 is reported under both SIF references (commit `cee4e8c`, `render`
   only), and the paper states the mismatch and the change. The verdicts agree
   at 78 of 78 cells. This replaces a pre-registered consequence, so **James
   must accept it in review**; if he does not, the R1 sentences of the E2
   paragraph, the pooling columns of the table, the appended sentence of the
   Sec. 4 "SIF at the collapsed layers" paragraph and R1 in this document have
   to be withdrawn. R2, R3, R4, the carriers and the zeroing follow-up do not
   depend on the SIF arm.
5. **R1 is not evaluable for mT5-base** under the frozen 0.05 floor; the raw
   changes are reported instead, and the expectation "no arm rescues" gets no
   verdict.
6. **Handoff claim row "Massive-coordinate mechanism".** Its surviving wording
   ("constant per-token values become per-passage variation through mean
   pooling; SIF's partial rescue runs through the tokens carrying the
   outliers") is supported for LaTa: the comma's contribution has one sign and
   the token mix explains 0.985 of the PC1 score. It is supported in part for
   PhilTa (same token types, context-dependent values, both SIF changes
   needed) and not for mT5-base. The paper says so and makes no claim about
   pretraining.
7. **Findings length.** The E2 findings take 14 sentences where the brief
   asked for 9 to 13.

## Paper edits

All in `overleaf_drafts/acl_latex.tex`; line numbers are those before the
edit. James's region: the E1 and E2 paragraphs and their tables. The other
edits are outside it; James authorized them where an E2 result decides the
sentence.

| # | Where | Region | Change |
|---|---|---|---|
| 1 | Abstract, line 73 | outside | `\pending{E2: one-clause result}` removed; one sentence added: token carriers in LaTa and PhilTa (three token types restore all 10 LaTa layers, at most 30 all 9 PhilTa layers), none in mT5-base |
| 2 | Contribution (3), line 131 | outside | `\pending{E2: token audit result}` replaced by one sentence with the same result |
| 3 | Sec. 4, "The standardized gap", line 343 | outside | "We do not have an account of this yet, and the token audit ... tests one candidate (E2)" becomes "We do not have an account of this: the token audit ... tests a length account and rejects it (E2)" |
| 4 | Sec. 4, "SIF at the collapsed layers", after line 352 | outside | One sentence appended with the outcome: the first suggestion is supported (LaTa, PhilTa), the second is not (mT5-base). The two existing sentences and the SIF ranges are unchanged |
| 5 | Sec. 5, E1 paragraph, line 423 | James | "we did not test whether it is the same direction, or why it still dominates the remainder" replaced by three sentences from the zeroing follow-up, labelled as designed after the E1 results and, for the intervention cells, as post hoc |
| 6 | Sec. 5, E2 paragraph, lines 433 to 450 | James | Design text rewritten for the revised design (17 sentences, with the frozen thresholds and the expectations); `\pending{E2: ...}` replaced by 14 sentences of findings; `\input{tables/e2_token_audit}` after the paragraph |
| 7 | Sec. 5, Scope, lines 510 and 511 | outside | "For what causes the collapse, our claim therefore reduces to ..." becomes "For the coordinate account, ..."; "leaves one direction dominant (E1)" becomes "leaves the rest of that direction dominant (E1 and its follow-up)"; one sentence added: E2 adds token carriers in LaTa and PhilTa and none in mT5-base, for which the claim stays the geometric description and the repair, and why those token types take such values is outside the scope |
| 8 | Discussion, line 678 | outside | `\pending{E2: whether specific tokens carry these coordinates}` replaced by one sentence |
| 9 | Discussion, line 686 | outside | "the token audit (E2) tests a length account" becomes "the token audit rejects a length account (E2)" |

No P2x2 marker was touched: the `pending` lines went from 18 to 14, the four
removed being the E2 markers.

Build: `/projects/bimc/swong2/setup/build_paper.sh` gives 60 pages (58
before), no undefined reference or citation, no duplicate label. The
Discussion starts on page 18 (16 before). One overfull box, 5.5pt, in the
generated `tables/e2_token_audit.tex` (none before).

## For the first author: sentences not edited

1. **Sec. 4, "SIF at the collapsed layers"**: the ranges "0.84--0.94" (LaTa),
   "0.87--0.93" (PhilTa) and "0.67--0.69" (mT5-base) come from published
   `sif_only` cells that the tracked split and the CLI do not reproduce to
   1e-6. The recomputed arm gives 0.857 to 0.939, 0.877 to 0.929 and 0.673 to
   0.691 (CSV). The same holds for the SIF line of Figure 1, the caption of
   the gap figure and the SIF appendix tables. See "The SIF gate".
2. **Sec. 4, same paragraph**: "This pattern suggests that tokens SIF
   down-weights or drops carry the dominant direction in LaTa and PhilTa, and
   that tokens it keeps carry it in mT5-base." Left as the suggestion it was;
   the appended sentence gives the outcome. It could be shortened now.
3. **Sec. 5 preamble**: "The testable content is that this direction is
   aligned with a few coordinates and has token carriers." Unchanged. E1 and
   E2 now answer both halves: not confined to a few coordinates; token
   carriers in LaTa and PhilTa, none in mT5-base.
4. **Introduction, line 116**: "We test whether mean pooling turns these
   coordinates into values that vary from passage to passage and so drown out
   content". Still true as a statement of the test; the outcome could follow.
5. **Sec. 5 Scope, last sentence**: "whether T5 pretraining or the missing
   embedding objective produces them" (carried over from the E1 list).
6. **Appendix "Why whitening is excluded"**: "SIF acts before pooling, so it
   can only attenuate the frequent tokens that feed the dominant direction".
   E2 supports this for LaTa, in part for PhilTa (the end-of-sequence token
   also feeds it, and SIF drops that token), and not for mT5-base.
7. **Related work**: nothing presupposes token carriers. The frequency link of
   Puccetti et al. (2022) fits LaTa's carriers, which are among its most
   frequent token types; no sentence was added.
8. **Generated table `tab:e2_token_audit`** (James's region, render code): it
   is 5.5pt wider than the text block; its caption says "the published SIF
   cells", which inside the paper should name Section 4; and its rho column is
   signed although the sign of a component is fixed only by a convention (its
   largest loading is positive), so the sign of rho is not comparable across
   layers or models.
9. **Page budget**: the paper grew from 58 to 60 pages (the table and about
   one column of text). The E2 design text can be cut further once the
   findings are accepted.
10. **Untested, for a later look**: LaTa's PC1 score is close to the share of
    commas in a passage. Whether same-directory witnesses differ in
    punctuation more than different-directory pairs do, which would bear on
    the negative raw gap, was not measured.
