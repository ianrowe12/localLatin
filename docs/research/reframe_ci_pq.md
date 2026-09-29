# Reframe CI and PQ: bootstrap intervals and routing checks (issue #233)

Results memo for the two `\pending` notes that belong to Ian in
`overleaf_drafts/acl_latex.tex`: **CI** (Section 7, "Routing converges") and
**PQ** (Section 7, "The offset regime and the open-set decision"). The claim
scoping follows `docs/research/reframe_handoff_20260927.md`. `acl_latex.tex` is
not edited here; the replacement LaTeX is at the end of this memo, for the
integration pass.

## Verdict in five lines

1. Every printed headline cell reproduces exactly (132 of 132 cells, 49
   configurations). Only after that check passed were the cells resampled.
2. Per-cell 95% intervals are about ±0.02 AUROC and ±3 points of routing
   accuracy. Every ABTT routing gain excludes zero. So do the ABTT AUROC gains
   of the three T5 encoders and LaBSE. The AUROC gains of Qwen3-0.6B (+0.006)
   and KaLM-mini (+0.009) do not.
3. The drops that ABTT causes on the fine-tuned encoders are all within noise,
   **including LaTa's** (0.984 to 0.970: −0.014, CI −0.032 to +0.001). This
   contradicts the current sentence "ABTT lowers AUROC for every fine-tuned
   model" read as an effect.
4. The lexical reference is not distinguishable from the best ABTT cells.
   Qwen3-0.6B and KaLM-mini tie at 89.4 DirAcc@1, and TF-IDF leads each by
   +0.5 (95% CI −1.4 to 2.5 against Qwen3-0.6B, −1.3 to 2.3 against
   KaLM-mini). Only LaTa's ABTT cell is significantly below TF-IDF. "Above
   every ABTT cell" does not survive.
5. PQ: the routing gain of the embedding-trained models appears without any
   threshold and at the oracle threshold. It is not a grid artifact and is not
   produced by centering. It comes with the removed top components and with a
   large drop in hubness. The checks support "ABTT improves the separation of
   known from new witnesses". They **weaken** any attribution to the shared
   offset (mean vector). They leave open what the removed components encode and
   whether reducing hubness causes the gain.

## What was run

| | |
|---|---|
| Code | `scripts/paper/reframe/ci_pq.py` (`compute`, `sweep`, `publish`, `render`), `ci_pq_core.py` (vectorized evaluator, bootstrap, checks), `ci_pq_render.py` (tables) |
| Job | `slurm/reframe/reframe_ci_pq.sbatch`, CPU partition, `beto-delta-cpu`, 16 cores, 48 GB |
| Run of record | job **22535867** (code `5245ec3`), 32:03 elapsed, 6 h 59 min total CPU, 7.9 GB peak RSS. `compute` 1,718 s wall, `sweep` 175 s |
| Earlier jobs | 22535011 (LaBSE pilot, B = 200, cancelled during the sweep); 22535062 and 22535192 (stopped at the reproduction gate, see below); 22535244 (pilot, B = 1,000, 4:06 elapsed, 35 CPU-min, used to set `--time`); 22535292 (first full run, 29:07, 6 h 28 min CPU; its intervals are identical to the run of record, which only adds the PQ paired contrasts) |
| Raw outputs | `runs/active/reframe/ci_pq_v2/` (also `bootstrap_replicates.npz`: every replicate and the draw counts), `runs/active/reframe/ci_pq_v2_sweep/` (`sweep_per_layer.csv`, 2,076 rows) |
| Published copies | `docs/research/data/reframe_ci_pq/` (`headline_ci.csv`, `headline_ci_diffs.csv`, `pq_cells.csv`, `sweep_selected.csv`, `reproduction_*.csv`, `run_info.json`, `sweep_run_info.json`). `publish` changes labels only, and every number is copied as text: D is blank where nothing is fitted (Base, SIF), and the replicate-max rows are relabelled (see CI results) |
| Tables | `overleaf_drafts/tables/headline_ci.tex` (`tab:headline_ci`), `headline_ci_diffs.tex` (`tab:headline_ci_diffs`), `pq_routing.tex` (`tab:pq_routing`), `taskA_headline_ci.tex` and `taskB_headline_ci.tex` (compact drop-ins that keep the labels `tab:taskA_headline`/`tab:taskB_headline`) |
| Tests | `tests/test_reframe_ci_pq.py`: the fast evaluator against `run_resubmit_evaluate` on random fixtures, the bootstrap weights against an explicit resample with duplicated directories, byte-identical regeneration of all five tables from the committed CSVs, the stale-CSV guard, and the lexical-row path of the compact tables |

Inputs: `runs/active/resubmit/data/phase_resubmit_split.csv`,
`runs/active/resubmit/results/phase_resubmit_results.csv`, the three
`finetune_*_ceiling_comparison.csv` and `*_layer_results.csv`,
`lexical_baselines.csv`. The embedding caches are
`runs/active/resubmit_bases/phase9_bases/<slug>/hidden_{mean,sif}_tokempty/`
and `runs/active/resubmit_finetune_bases/phase9_bases/<slug>-ft/hidden_mean_tokempty/`.
Every matrix is loaded through `AlignmentResolver` (all caches
`verified-permuted`, 17 rows moved, as expected after the benchmark v1
corrections). SIF weights are the cached ones: a = 1e-3, p(w) from the
training texts (`sif_prob_source: corpus` with `--split_csv`).

Regenerate:

```bash
sbatch slurm/reframe/reframe_ci_pq.sbatch        # OUT_TAG=ci_pq_v3 for a new run; outputs are never overwritten
python scripts/paper/reframe/ci_pq.py publish --data_root <repo> --out_dir runs/active/reframe/<tag> --sweep_dir runs/active/reframe/<tag>_sweep
python scripts/paper/reframe/ci_pq.py render
```

## Method

**Directory-level bootstrap.** One replicate draws the 514 test directories
with replacement (multinomial counts, B = 10,000, `numpy.random.default_rng(233)`).
A drawn directory keeps all its files and pairs. A directory drawn c times
counts c times: each of its files and within-directory (positive) pairs gets
weight c, and each pair it forms with a directory drawn c' times gets weight
c·c'. Pairs between two copies of the same directory are not formed, because
a file is never paired with its own copy. A test checks that these weights
give exactly the AUROC of an explicit resample with duplicated directories.
Intervals are 95% **percentile** intervals. BCa was not used: the replicate
distributions are close to symmetric, and at B = 10,000 the percentile
endpoints are stable to the printed digit.

**What is held fixed.** The layer, τ, the ABTT components and D, the SIF
probabilities and the fine-tuned checkpoints all stay as fit on the fixed
training split. Only the test evaluation is resampled. The scores of every
pair and every file are computed once on the full test set (for Task B, each
file's candidate pool is the full test set), and a replicate reweights those
units. So the intervals cover test-set sampling. They do not cover the choice
of split, of layer or of τ. The five-seed Task B figures in the paper address
the split.

**Paired differences.** Both cells are recomputed on the same replicates, and
the interval is taken on the per-replicate difference. Each cell sits at its
own train-selected layer, so "ABTT − Base" compares printed cells, not one
layer. `share_le_0` in `headline_ci_diffs.csv` is the fraction of replicates
with a difference ≤ 0.

**Rounding.** Differences are computed at full precision and then rounded,
so they can differ by one unit in the last digit from the difference of the
two printed cells. Qwen3-0.6B's ABTT AUROC gain is +0.006 although its printed
cells, 0.966 and 0.973, differ by 0.007; LaBSE's is +0.030 against printed
0.956 and 0.987. The contrast table's caption says so. This memo and the
replacement LaTeX quote the full-precision difference with its interval.

**Spreads.** The largest minus the smallest of the six zero-shot cells, per
replicate. Under resampling the range of near-tied values is biased upward,
so the ABTT spread interval is conservative on the high side.

## Reproduction check (before any resampling)

For every configuration behind a headline cell (6 models × 4 settings × 2
tasks, 3 fine-tuned × 2 settings × 2 tasks, 3 lexical; 49 distinct
configurations), `compute` first re-runs the paper's own evaluator
(`run_resubmit_evaluate.evaluate_single`, which re-selects D for ABTT, or
`lexical_baselines.evaluate_single_split`). It compares the result with the
CSV behind the table, then rebuilds the per-pair and per-file units through
`ci_pq_core` and compares again. It refuses to resample on any mismatch.

| Quantity | paper evaluator vs CSV | fast units vs paper evaluator |
|---|---|---|
| τ, D, assignment acc., DirAcc@1, train DirAcc@1 | identical (46/46 D) | identical |
| AUROC | max \|Δ\| 3.4e−8 | 2.2e−16 |
| cosine gap | 1.2e−7 | 2.9e−8 |
| train AUROC | 6.7e−8 | n/a |

The 1e−8 AUROC differences come from float32 cosine matrices whose last bit
depends on the BLAS thread count. They reorder a few near-tied pairs and never
touch a printed digit. The first two full-scope pilots (22535062, 22535192)
stopped at this gate. The first used a 1e−9 tolerance, which this jitter
exceeds. The second compared exact counts at tolerance 0 and failed on the
last-ulp error of pandas' CSV float parser. The gate now uses 1e−6 on
continuous scores and 1e−12 on counts, τ and D. At printed precision all
**132** cells match (`reproduction_cells.csv`): 120 against the committed
`taskA_headline.tex`/`taskB_headline.tex`, and the 12 lexical cells against
`lexical_baselines.csv`, because the headline tables do not yet carry the
lexical rows. The `sweep` pass with the paper's 200-point grid re-derives
every published layer and D from scratch, and all 60 embedding cells match.

Route-to-Ian item 19 is confirmed: fine-tuned Qwen3-0.6B sits at layer 28
(baseline) and layer 27 (ABTT, D = 2) on both tasks.

## CI results

Full table: `tab:headline_ci` (every cell) and `tab:headline_ci_diffs`
(contrasts). The key contrasts:

| Contrast | AUROC | Assignment acc. | DirAcc@1 |
|---|---|---|---|
| ABTT − Base, LaTa | +0.034 (0.017, 0.054) | +14.7 (11.1, 18.3) | +14.0 (10.4, 17.6) |
| PhilTa | +0.043 (0.029, 0.060) | +20.3 (16.2, 24.5) | +19.0 (15.0, 23.1) |
| mT5-base | +0.137 (0.109, 0.166) | +43.1 (38.6, 47.6) | +41.8 (37.2, 46.4) |
| LaBSE | +0.030 (0.014, 0.049) | +7.5 (4.6, 10.4) | +7.0 (4.1, 10.0) |
| Qwen3-0.6B | **+0.006 (−0.010, 0.019)** | +9.2 (6.3, 12.2) | +9.1 (6.1, 12.2) |
| KaLM-mini | **+0.009 (−0.002, 0.021)** | +4.2 (1.7, 6.7) | +3.5 (1.0, 6.0) |
| Fine-tuned: ABTT − Base, LaTa | **−0.014 (−0.032, +0.001)** | +4.3 (1.7, 7.1) | +3.6 (1.0, 6.3) |
| Qwen3-0.6B (L28 vs L27) | −0.002 (−0.005, +0.001) | +0.7 (−0.7, 2.1) | +0.6 (−0.8, 2.0) |
| KaLM-mini | −0.003 (−0.007, +0.001) | +0.9 (−0.3, 2.2) | +0.8 (−0.5, 2.1) |
| Fine-tuned ABTT − zero-shot ABTT, LaTa | −0.002 (−0.010, 0.005) | −0.7 (−2.6, 1.2) | −0.9 (−2.8, 1.0) |
| Qwen3-0.6B | +0.022 (0.006, 0.044) | +0.6 (−1.6, 2.9) | +2.0 (0.0, 4.0) |
| KaLM-mini | +0.013 (0.002, 0.029) | +1.6 (−0.3, 3.6) | +3.1 (1.2, 5.2) |
| TF-IDF char 3–5 − ABTT, Qwen3-0.6B | +0.014 (0.002, 0.031) | +0.3 (−1.6, 2.3) | +0.5 (−1.4, 2.5) |
| TF-IDF char 3–5 − ABTT, KaLM-mini | +0.006 (−0.002, 0.016) | +0.1 (−1.6, 1.9) | +0.5 (−1.3, 2.3) |
| TF-IDF char 3–5 − ABTT, LaBSE | +0.000 (−0.004, 0.005) | +1.0 (−1.0, 3.1) | +1.4 (−0.8, 3.6) |
| Spread over six models, Base | 0.134 (0.110, 0.158) | 40.3 (35.7, 44.7) | 39.3 (34.6, 43.7) |
| Spread over six models, ABTT | 0.015 (0.006, 0.034) | 3.3 (1.8, 5.8) | 3.3 (1.8, 6.0) |

TF-IDF against each model's ABTT cell (DirAcc@1): LaTa +3.7 (1.4, 6.2) is the
only one that excludes zero. The best ABTT cells are Qwen3-0.6B's and
KaLM-mini's, tied at 89.4, and the fixed contrasts against them are the ones
quoted. On Task A, the best ABTT cell is LaBSE's, and the AUROC difference is
0.000 (−0.004, 0.005), a tie.

`headline_ci_diffs.csv` also carries a row group
`tfidf_minus_replicate_max_abtt`: TF-IDF minus the best ABTT cell re-chosen in
every replicate. It is labelled there as a max-over-models statistic and is
not used in the text. The replicate maximum sits above the maximum of the
point estimates (mean replicate max 89.84 against 89.39 DirAcc@1, about +0.45),
so its interval is centred near zero even though the point difference is
+0.47. The first version of this memo quoted it; the fixed contrasts above
replace it.

The fine-tuned drops: the replicate shares below zero are 96% (LaTa), 89%
(Qwen3-0.6B) and 92% (KaLM-mini). The direction is consistent across the
three, but no single drop is established at 95%.

The contrast the task statement asked for, fine-tuned+ABTT minus zero-shot
ABTT for Qwen3-0.6B and KaLM-mini, is in the table above. The 0.002–0.003
drops the handoff calls "within noise" are the other contrast, fine-tuned+ABTT
minus fine-tuned Base, and both are reported.

## PQ results (routing models at their Task B train-selected layers)

Table `tab:pq_routing` gives all six zero-shot models. For each it shows the
Base and ABTT cells, and baseline and centering at the ABTT cell's layer. The
CSV `pq_cells.csv` covers all 33 Task B headline cells plus the centering
configurations. The three embedding-trained models:

| | LaBSE | Qwen3-0.6B | KaLM-mini |
|---|---|---|---|
| Layer Base / ABTT | 11 / 11 | 28 / 5 | 23 / 3 |
| **(1) Existing-vs-new AUROC of the max cosine**, Base cell → ABTT cell | 0.922 → 0.960 | 0.900 → 0.954 | 0.931 → 0.958 |
| paired Δ (95% CI) | +0.037 (0.021, 0.055) | +0.055 (0.034, 0.077) | +0.027 (0.012, 0.043) |
| **(2) Oracle gap**: best-test-threshold assignment acc. − train-τ assignment acc., Base cell | +1.0 (0.1, 3.5) | +1.0 (0.2, 2.6) | 0.0 (0.0, 1.1) |
| same, ABTT cell | +0.6 (0.1, 1.6) | +0.7 (0.2, 1.7) | +0.2 (0.0, 1.6) |
| ABTT − Base at each side's oracle threshold | +7.0 (4.2, 9.3) | +8.9 (6.1, 11.7) | +4.4 (2.2, 6.8) |
| **(3) Hubness**: skewness of N₁₀, Base cell → ABTT cell | 2.00 → 0.58 | 2.51 → 1.07 | 1.97 → 0.87 |
| max N₁₀, Base cell → ABTT cell | 68 → 28 | 86 → 38 | 63 → 37 |
| **(4) Centering only (D = 0)**, assignment acc. − Base, at the Base layer | −4.7 (−7.4, −2.0) | +2.6 (−0.4, 5.5) | −1.9 (−3.7, −0.1) |
| ABTT − centering at the ABTT layer | +12.1 (8.8, 15.5) | +25.9 (22.0, 29.9) | +12.0 (9.1, 15.1) |
| centering, own train-selected layer (assign / DirAcc@1 / Task A AUROC) | 83.9 / 82.1 / 0.956 | 85.4 / 83.1 / 0.975 | 85.7 / 83.9 / 0.979 |
| **(5) τ refit, fixed layer and D**: Base assign (paper → exact cut) | 83.3 → 82.5 | 82.3 → 82.6 | 87.5 → 86.2 |
| ABTT assign (paper → exact cut) | 90.8 → 90.8 | 91.5 → 91.5 | 91.7 → 91.7 |
| SD of test pair cosines at the Base cell | 0.039 | 0.043 | 0.021 |
| same at the ABTT cell (after ABTT; raw at that layer) | 0.102 (0.039) | 0.086 (0.034) | 0.085 (0.054) |

Definitions and notes:

- (1) The AUROC of each test file's maximum cosine to any other test file,
  with existing files (a same-directory partner in test) as positives. It
  involves no threshold.
- (2) The best assignment accuracy over every cut of the test max-cosines,
  minus the printed value at train τ. The oracle is re-optimized in every
  replicate. It is an accuracy gap, not a distance between thresholds. At the
  three Base cells the point gap is at most 1.0 and the upper bounds are 1.1
  to 3.5.
- (1) and (3) compare the Base cell with the ABTT cell, each at its own
  train-selected layer. The layers are the same for LaBSE (11), but not for
  Qwen3-0.6B (28 vs 5) or KaLM-mini (23 vs 3). The rows at the ABTT layer in
  `tab:pq_routing` separate the layer change from the projection.
- (3) N_k(x) is the number of test files that have x among their k = 10
  nearest neighbours by cosine, excluding self. The statistic is the
  population skewness E[(N−μ)³]/σ³, the k-occurrence skewness of Radovanović
  et al. (2010, JMLR; `radovanovic2010hubs` is already in `custom.bib`). It
  is a point estimate with no interval.
- (4) Subtract the training mean and remove no component. The rows in the
  main table are at the Base and ABTT layers. The "own train-selected layer"
  row applies the paper's selection rule to centering across all layers (from
  `sweep_selected.csv`, point estimates). For comparison, ABTT gives 90.8 /
  88.5 / 0.987 (LaBSE), 91.5 / 89.4 / 0.973 (Qwen3-0.6B) and 91.7 / 89.4 /
  0.981 (KaLM-mini).
- (5) Fine grid = `linspace(0, 1, 10001)`. Exact cut = the best-F1 threshold
  over every distinct training pair score, which is the limit of any finer or
  quantile grid. A quantile grid over all train pairs was not used: the F1
  optimum sits in the top 0.2% of the 358k train pair scores (565 positives),
  so 200 quantiles would put about one grid point in the relevant tail and be
  coarser than the paper grid there.
- Qwen3-0.6B's 0.007 standard deviation quoted in the paper is at **layer 27**
  (`geometry_per_layer.csv`, raw view). Layer 27 is not a headline cell. At
  the Base cell (layer 28) the SD of test pair cosines is 0.043, and that of
  the per-file max cosine is 0.031. One grid step (0.005) is about 0.12 SD
  there.

Threshold grid across all printed Task B numbers (66 numbers from 33 cells):

- **Fixed layer and D.** The fine grid moves 45 of the 66 printed routing numbers, by a median of 3 files (0.35 points); ABTT, SIF+ABTT and fine-tuned cells move by at most 0.8, and Base and SIF cells by up to 2.6. The
  exact cut moves 48. The fine-grid moves, in files (1 file = 0.117 points),
  are distributed as {1: 14, 2: 3, 3: 10, 5: 1, 6: 3, 7: 4, 8: 2, 11: 2,
  13: 2, 15: 2, 17: 1, 22: 1}. Moves above 1 point: mT5-base Base
  +2.6/+2.3 (fine/exact), KaLM-mini SIF +2.0/+2.1, LaBSE SIF −1.5/−1.7,
  KaLM-mini Base −1.3/−1.3, mT5-base SIF +1.3/+1.4. The mean signed change is
  +0.15.
- **Layer and D re-selected under the new grid as well** (60 embedding
  cells). The fine grid moves 42 and the exact cut 44. The largest moves are
  mT5-base Base +5.0 (layer 12 → 1), LaBSE SIF −2.3 (layer 12 → 11), KaLM-mini
  SIF +2.1, PhilTa SIF+ABTT −2.0 (layer 1 → 2) and fine-tuned KaLM-mini ABTT
  −1.2 (D 10 → 5).

**Recommendation.** Keep the published grid. Every grid variant optimizes the
same training objective (best F1 over same- vs different-directory pairs), and
the 200-point grid is simply its published resolution. Refitting would trade
one set of one-to-few-file differences for another, and it would not change
any conclusion. Disclose the sensitivity instead, with the `Exact` column of
`tab:pq_routing` and the Limitations sentence below.

Collapsed T5 baselines (for completeness): their oracle gaps are large (LaTa
+5.6, PhilTa +7.0, mT5-base +19.6), so part of their baseline deficit is a
train-to-test threshold shift. ABTT still wins by 9.7 to 24.5 points when both
sides use their oracle thresholds.

## Honest reading against Section 7, "The offset regime and the open-set decision"

The paragraph makes, or implies, six claims.

1. *"ABTT still changes their routing: 83.3 → 90.8, 82.3 → 91.5,
   87.5 → 91.7."* **Supported.** All three gains exclude zero (lowest bound
   +1.7 for KaLM-mini).
2. *"ABTT also improves their ranking: LaBSE's AUROC rises from 0.956 to
   0.987, and DirAcc@1 gains 3.5 to 9.1 points."* **Supported for LaBSE and
   for DirAcc@1.** The Task A AUROC gains of Qwen3-0.6B and KaLM-mini are
   within noise. The sentence does not claim them, but "their ranking"
   invites the reader to assume them. By contrast, the threshold-free
   existing-vs-new AUROC (check 1) rises significantly for all three.
3. *"If an offset only added a constant … a shared offset vector need not act
   as such a constant … LaBSE's AUROC gain … is consistent with this, and the
   D = 0 check isolates the mean."* **Weakened.** Removing the mean alone
   leaves LaBSE's AUROC unchanged (−0.002, CI −0.009 to 0.004). The whole
   gain comes with the ten removed components (+0.029, 0.013 to 0.048).
   Centering also lowers LaBSE's routing (−4.7) and KaLM-mini's (−1.9). For
   Qwen3-0.6B and KaLM-mini, centering at its own selected layer does reach
   ABTT's Task A AUROC (0.975 vs 0.973; 0.979 vs 0.981), but it recovers at
   most 3 of the 4 to 9 routing points. The shared offset, read as the mean
   vector, does not account for the routing gain in any of the three.
4. *"A cost to the open-set decision beyond ranking would need offsets that
   differ by query, such as hub vectors … or a shift between the training and
   test score distributions."* **The shift is ruled out and hubs are
   consistent.** At every embedding-trained Base cell, the train-fit τ gives
   assignment accuracy within 1.0 point of the best test threshold (upper
   bounds 1.1 to 3.5), and ABTT's gain holds at the oracle threshold
   (check 2). Hubness falls sharply from the Base cell to the ABTT cell
   (skewness 2.0–2.5 → 0.6–1.1; the largest hub reaches 63–86 files at the
   Base cell and 28–38 at the ABTT cell), while centering alone at the ABTT
   layer leaves it near 1.8. This is correlational.
   The checks do not show that reducing hubness causes the routing gain.
5. *"The threshold grid offers a mundane alternative … one grid step can
   span most of a standard deviation, and ABTT could help routing simply by
   spreading the cosines."* **Not supported as an explanation of the gain.**
   The 0.007 SD is not at a headline cell. With τ refit exactly, the three
   ABTT cells do not move and the baselines move by −1.3 to +0.3, so the gap
   grows or shrinks by about a point. The gain is present with no threshold
   at all (check 1). The grid does matter for the **precision of the printed
   one-decimal routing numbers**: the fine grid moves 45 of the 66 printed routing numbers, by a median of 3 files (0.35 points); ABTT, SIF+ABTT and fine-tuned cells move by at most 0.8, and Base and SIF cells by up to 2.6 (5.0 with re-selection).
   That belongs in the appendix or Limitations, not in the offset argument.
6. *"Until they run, we do not attribute the routing gain to the offset
   geometry."* The checks attribute the gain to what ABTT removes beyond the
   mean. The supported wording is "a gain in separating known from new
   witnesses by their best match, carried by the removed top components and
   accompanied by lower hubness". The "offset regime" label for these models
   can stay as a description of their geometry (high mean pairwise cosine,
   variance spread over many directions). It should not be offered as the
   cause of the routing gain.

**Left open:** what the removed components encode in the embedding-trained
models (E3's subspace split would answer it); whether hub reduction is causal
(one test would be a hubness-only correction such as mutual-proximity or CSLS
rescoring on the Base cells); and whether the intervals survive a different
split (the five-seed spread speaks to that only for Task B).

## Contradictions with the current text

1. Section 7, "Where the repair costs": "On pairwise ranking, ABTT lowers
   AUROC for every fine-tuned model." All three drops, LaTa's included, have
   95% intervals that reach zero. Handoff row "ABTT hurts models fine-tuned on
   the target task" says "consistent with, not replicates". It now needs
   "in direction" as well.
2. Same paragraph: "The same projection still raises their routing …" holds
   only for fine-tuned LaTa (+3.6, 1.0 to 6.3). The Qwen3-0.6B (+0.6) and
   KaLM-mini (+0.8) rises are within noise.
3. Section 7, "Routing converges", and Section 3, "Lexical reference": "89.9
   … above every ABTT cell (86.1 to 89.4)" and "above every pre-trained one".
   TF-IDF is not distinguishable from the best ABTT cells: +0.5 (−1.4 to 2.5
   against Qwen3-0.6B, −1.3 to 2.3 against KaLM-mini), so the supported word
   is "level with". Only LaTa's ABTT cell is significantly below TF-IDF. The
   generated caption of `taskB_headline.tex` with `--lexical_csv` already
   says "level with", from its one-point tolerance.
4. "Practitioner rule": "… and hurt ranking in contrastively fine-tuned ones"
   is stronger than the data. "Did not help ranking" survives.
5. Section 7: "The test cosines of Qwen3-0.6B spread with a standard
   deviation as small as 0.007." True of layer 27 only. The routing cells are
   at layers 28 (SD 0.043) and 5 (0.086 after ABTT; 0.034 raw).
6. The generated caption of `taskB_headline.tex` places the fine-tuned
   Qwen3-0.6B and KaLM-mini ABTT rows "above every zero-shot ABTT cell".
   On the single split their assignment accuracy exceeds their own zero-shot
   ABTT cell by +0.6 (−1.6, 2.9) and +1.6 (−0.3, 3.6). Only DirAcc@1 for
   KaLM-mini (+3.1, 1.2 to 5.2) is outside noise. The caption comes from
   `build_headline_tables.py` and compares points against the zero-shot
   range. It is not an inference claim, but a reader will read it as one.
   Routed to Ian; not changed here.
7. Limitations: "The headline cells carry no confidence intervals until …"
   is superseded (replacement below).
8. Section 7, "Routing converges": "Surface overlap remains the strongest
   router on this corpus." Replace with "Surface overlap routes as well as the
   best repaired encoder on this corpus."
9. "Practitioner rule": "… a character n-gram reference routes better than
   every pre-trained configuration on this corpus." Replace "routes better
   than every" with "routes as well as the best".
10. "Where the repair costs": "Why the projection costs ranking in fine-tuned
    models is the subspace question …". Replace "Why the projection costs
    ranking in fine-tuned models" with "Whether the projection costs ranking
    in fine-tuned models", since no drop is established.

Supported as written: the abstract's "cuts the cross-model spread in routing
accuracy from 39.3 to 3.3 points" (34.6–43.7 → 1.8–6.0); "ABTT moves every
model into a band from 0.971 to 0.987"; "mT5-base gains 0.137"
(0.109–0.166); "the repair matches the lexical reference but does not exceed
it" (TF-IDF minus LaBSE's ABTT cell, the best on Task A: 0.000, −0.004 to 0.005). The five-seed claim "fine-tuned Qwen3-0.6B
with ABTT routes at 92.3 against 90.5 …" is a different protocol. The
single-split DirAcc@1 contrasts point the same way (+2.0, 0.0 to 4.0;
+3.1, 1.2 to 5.2).

## Replacement LaTeX

Each block replaces the quoted text in `overleaf_drafts/acl_latex.tex`. Table
inputs: add `\input{tables/headline_ci}`, `\input{tables/headline_ci_diffs}`
and `\input{tables/pq_routing}` to an appendix (suggested title "Bootstrap
Intervals and Routing Checks", label `app:ci_pq`). To show intervals in the
main text, replace `\input{tables/taskA_headline}` /
`\input{tables/taskB_headline}` with `taskA_headline_ci` / `taskB_headline_ci`.
They keep the same labels, and they cost one extra row per model. If the
headline tables are regenerated with `--lexical_csv` (handoff TODO), run
`python scripts/paper/reframe/ci_pq.py render` afterwards. The compact tables
copy the headline tables line by line, the renderer adds interval rows to the
lexical rows, and it stops if a printed value and the CI CSV disagree.
No new bibliography entry is needed. Line numbers below refer to `main` at
`9beeeec` and shift after #239; match on the quoted strings.

**CI, line 624.** Replace
`The single-split cells in both headline tables carry no intervals yet: \pending{CI: directory-level bootstrap intervals on every headline cell}.`
with:

```latex
Directory-level bootstrap intervals (Appendix~\ref{app:ci_pq}, Table~\ref{tab:headline_ci}) span about $\pm 0.02$ AUROC and $\pm 3$ points of routing accuracy per cell.
Every routing gain from ABTT excludes zero, as do the AUROC gains of the three T5 encoders and LaBSE; those of Qwen3-0.6B ($+0.006$, 95\% CI $-0.010$ to $0.019$) and KaLM-mini ($+0.009$, $-0.002$ to $0.021$) do not (Table~\ref{tab:headline_ci_diffs}).
The spread in DirAcc@1 falls from 39.3 (34.6 to 43.7) to 3.3 (1.8 to 6.0) points, and the lexical reference is not distinguishable from the best ABTT cells ($+0.5$; 95\% CI $-1.4$ to $2.5$ against Qwen3-0.6B, $-1.3$ to $2.3$ against KaLM-mini); only LaTa's ABTT cell is significantly below it.
```

and in the same paragraph change
`The lexical reference routes at 89.9 on the single split, above every ABTT cell (86.1 to 89.4), and at 91.1 over five seeds, above every SIF+ABTT cell (88.9 to 90.6).`
to
`The lexical reference routes at 89.9 on the single split, level with the best ABTT cells (86.1 to 89.4), and at 91.1 over five seeds, above every SIF+ABTT cell (88.9 to 90.6).`
(The five-seed clause is outside this bootstrap and is left as written.)
In the next sentence, change `Surface overlap remains the strongest router on this corpus.` to `Surface overlap routes as well as the best repaired encoder on this corpus.`
In Section 3, "Lexical reference", change `and 89.9 DirAcc@1 on Task~B, above every pre-trained one` to `and 89.9 DirAcc@1 on Task~B, level with the best pre-trained ones`.

**PQ, lines 641–642.** Replace
`Five checks on cached embeddings test these readings: \pending{PQ: ...}.`
and `Until they run, we do not attribute the routing gain to the offset geometry.`
with:

```latex
Five checks on cached embeddings test these readings (Table~\ref{tab:pq_routing}).
The gain does not depend on the threshold.
With no threshold at all, the AUROC of each file's best-match cosine for existing against new files rises from the baseline cell to the ABTT cell by 0.037 (95\% CI 0.021 to 0.055) for LaBSE, 0.055 (0.034 to 0.077) for Qwen3-0.6B, and 0.027 (0.012 to 0.043) for KaLM-mini.
At their baseline cells the train-fit $\tau$ gives assignment accuracy within 1.0 point of the best test threshold (upper bounds 1.1 to 3.5), and the gain survives when both settings use their best test thresholds (7.0, 8.9, and 4.4 points).
Refitting $\tau$ as the exact best-F1 cut leaves the three ABTT cells unchanged and moves the baselines by $-1.3$ to $+0.3$ points; the 0.007 standard deviation belongs to Qwen3-0.6B's layer 27, and at its selected layer 28 the test cosines spread with 0.043.
Nor does the shared offset produce the gain.
At each baseline cell's layer, centering alone ($D=0$) changes assignment accuracy by $-4.7$ points ($-7.4$ to $-2.0$) for LaBSE, $+2.6$ ($-0.4$ to $5.5$) for Qwen3-0.6B, and $-1.9$ ($-3.7$ to $-0.1$) for KaLM-mini, and at the ABTT layer of each of the three the removed components add 12.0 to 25.9 points on top of centering.
From the baseline cell to the ABTT cell hubness also falls \citep{radovanovic2010hubs}: the skewness of the 10-occurrence distribution drops from 2.0--2.5 to 0.6--1.1, while centering alone at the ABTT layer leaves it near 1.8.
We therefore read the routing gain as better separation of known from new witnesses by their best match, carried by the top components beyond the mean; whether the reduced hubness causes it remains untested.
```

Earlier in the same paragraph, change
`Whether centering alone, which removes the shared offset, accounts for this is what the $D=0$ check below tests.`
to
`Centering alone, which removes the shared offset, does not account for this (below).`
and change
`LaBSE's AUROC gain under ABTT, which removes the mean and ten components, is consistent with this, and the $D=0$ check isolates the mean.`
to
`LaBSE's AUROC gain under ABTT does not come from the mean: centering alone leaves its AUROC unchanged ($-0.002$, $-0.009$ to $0.004$), and the gain arrives with the ten removed components ($+0.029$, $0.013$ to $0.048$).`

**"Where the repair costs", lines 648–651.** Replace the first four sentences
(`On pairwise ranking, ABTT lowers AUROC ...` through `... and from 91.7 to 92.5 for KaLM-mini (Table~\ref{tab:finetune_ceiling}).`) with:

```latex
ABTT does not help fine-tuned ranking: AUROC falls by 0.002 to 0.014 in point estimate, and every 95\% interval includes zero.
Fine-tuned LaTa falls from 0.984 to 0.970 ($-0.014$, $-0.032$ to $+0.001$), and Qwen3-0.6B and KaLM-mini fall by 0.002 and 0.003 ($-0.005$ to $+0.001$; $-0.007$ to $+0.001$); for Qwen3-0.6B the two rows also sit at different train-selected layers (28 and 27).
The same projection raises fine-tuned LaTa's DirAcc@1 from 81.6 to 85.2 ($+3.6$, 1.0 to 6.3); the rises for Qwen3-0.6B (90.8 to 91.4) and KaLM-mini (91.7 to 92.5) are within noise (Table~\ref{tab:finetune_ceiling}).
```

and in the next sentence change `The ranking loss is consistent with \citet{rajaee2021finetuning}` to `The direction of the ranking loss is consistent with \citet{rajaee2021finetuning}`.
Later in the paragraph change `Why the projection costs ranking in fine-tuned models is the subspace question` to `Whether the projection costs ranking in fine-tuned models is the subspace question`.
Optionally, after `... and KaLM-mini at 93.0 against 90.9; for LaTa the two tie at 87.7 (Appendix~\ref{app:reference_systems}).`, add:
`On the single split the DirAcc@1 differences are $+2.0$ (0.0 to 4.0) and $+3.1$ (1.2 to 5.2).`

**"ABTT restores every layer", line 606.** Change
`The gain is largest where the baseline is weakest: mT5-base gains 0.137, while KaLM-mini gains 0.009.`
to
`The gain is largest where the baseline is weakest: mT5-base gains 0.137 (0.109 to 0.166), while KaLM-mini gains 0.009 ($-0.002$ to $0.021$), within noise.`

**"Practitioner rule", lines 660 and 662.** Change `and hurt ranking in contrastively fine-tuned ones` to `and did not help ranking in contrastively fine-tuned ones`, and `a character n-gram reference routes better than every pre-trained configuration on this corpus` to `a character n-gram reference routes as well as the best pre-trained configuration on this corpus`.

**Limitations, line 694.** Replace
`The headline cells carry no confidence intervals until the directory-level bootstrap is run (CI), so small differences, such as the ABTT drops on fine-tuned Qwen3-0.6B and KaLM-mini, should not be read as effects.`
with:

```latex
The bootstrap intervals resample test directories with the layer, threshold, and projections held fixed, so they cover test-set sampling but not the choice of split; they span about $\pm 0.02$ AUROC and $\pm 3$ routing points, and differences below that, including the ABTT drops on the fine-tuned encoders and the lexical reference's lead, are not effects. The printed routing accuracies are precise to a few files: refitting the threshold on a finer grid moves 45 of the 66 printed routing numbers, by a median of 3 files (0.35 points); ABTT, SIF+ABTT, and fine-tuned cells move by at most 0.8, and Base and SIF cells by up to 2.6 (Table~\ref{tab:pq_routing}).
```

## Suggested appendix text (`app:ci_pq`)

```latex
\section{Bootstrap Intervals and Routing Checks}
\label{app:ci_pq}

\paragraph{Intervals.} A bootstrap replicate draws the 514 test directories with replacement ($B=10{,}000$, seed 233). A drawn directory keeps every file and pair: a directory drawn $c$ times weights its files and within-directory pairs by $c$ and its pairs with a directory drawn $c'$ times by $cc'$, and no pair is formed between two copies of one directory. The layer, threshold, ABTT components, $D$, and SIF weights stay fit on the training split, and every score is computed once on the full test set, so a replicate only reweights test pairs and files. Intervals are 95\% percentiles; a paired difference recomputes both cells on each replicate. Before resampling, every cell was recomputed with the paper's evaluation code and matched its printed value. Table~\ref{tab:headline_ci} gives every cell and Table~\ref{tab:headline_ci_diffs} the contrasts quoted in Section~\ref{sec:repair}.

\paragraph{Routing checks.} Table~\ref{tab:pq_routing} reports, at the Task~B cells, the AUROC of each test file's maximum cosine for existing against new files (no threshold), the gap between the best test threshold and the training threshold, the skewness of the 10-occurrence distribution \citep{radovanovic2010hubs}, centering alone ($D=0$), and assignment accuracy with $\tau$ refit as the exact best-F1 cut over all training pair scores.
```
