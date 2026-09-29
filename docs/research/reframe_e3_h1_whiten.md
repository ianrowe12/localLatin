# Reframe experiments E3, H1 and WHITEN (issue #232)

Results for Ian's three CPU experiments in the analysis reframe
(`reframe_handoff_20260927.md`), with the replacement LaTeX for every
`\pending{E3 ...}`, `\pending{H1 ...}` and `\pending{WHITEN ...}` in
`overleaf_drafts/acl_latex.tex`. That file is not edited here; the text below is
for the single integration pass.

Short version:

- **H1 fails its own test.** The nuisance at the collapsed T5 layers is not
  rank-1. Removing one component recovers a median 45 percent of the D=10 AUROC
  gain and passes the 80 percent bar at 1 of 26 collapsed layers. Three
  components pass it at all 26 (median 96 percent). Centering alone recovers
  nothing. This holds even though PC1 carries a median 94 percent of the
  centered training variance there. The "one direction" wording has to go
  wherever it describes what breaks ranking.
- **E3 does not trigger the third falsifier.** In every pre-trained model the
  removed subspace ranks well below the retained one (at chance, median 0.489,
  at the collapsed layers; 0.868 against 0.971 at LaTa layer 12). Fine-tuning
  narrows the gap at the same layer and D=10 to under 0.02 in all three
  fine-tuned encoders, which fits the Rajaee and Pilehvar account of the ABTT
  drop after fine-tuning.
- **WHITEN: reduced-rank whitening performs on par with ABTT.** With k=128 or
  256 components, fit on train, whitening lands within about 0.01 of ABTT at
  each model's train-selected layer (0.973 to 0.984 against 0.971 to 0.987).
  At the same layer, whitening with k >= 128 is within -0.009 to +0.021 of
  ABTT at all 100 model-layers, and it also restores every collapsed layer
  (0.966 or more). Only full-rank whitening fails. The paper can no longer
  present ABTT as the only projection that works; the finding is that removing
  or rescaling the few dominant directions restores ranking.

## Provenance

| Item | Value |
|---|---|
| Script | `scripts/paper/reframe/abtt_subspace_whiten.py` (subcommands `h1`, `e3`, `whiten`, `render`) |
| Batch file | `slurm/paper/reframe_e3_h1_whiten.sbatch` (CPU partition, `beto-delta-cpu`) |
| Job | 22534981, COMPLETED, 16 CPUs, 48 GB |
| Wall time | elapsed 00:05:27 of 00:25:00 reserved (reserved 16 x 25 min = 6.7 core-hours; allocated 16 x 5:27 = 1.45 core-hours; TotalCPU 1:11:10) |
| Pilot | one model-layer per subcommand on the login node, about 40 s in all (H1 23 s, E3 7 s, WHITEN 8 s per model-layer at 4 BLAS threads), used to size `--time` |
| Inputs | `runs/active/resubmit/data/phase_resubmit_split.csv` (tracked since #231); cached mean-pooled hidden states `runs/active/resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/` and `runs/active/resubmit_finetune_bases/phase9_bases/<slug>-ft/hidden_mean_tokempty/` from the HPC working copy |
| Alignment | the job's alignment gate (`verify_embedding_alignment.py`) passed: 19 caches manifest-verified, 38 byte-identical spot checks. Every matrix is loaded through `AlignmentResolver` (17 rows moved per cache, as in benchmark v1) |
| Outputs | `runs/active/reframe/h1/h1_d_ablation.csv` (1,200 rows), `runs/active/reframe/e3/e3_subspace_split.csv` (198 rows), `runs/active/reframe/whiten/whiten_reduced.csv` (400 rows), `runs/active/reframe/facts_e3_h1_whiten.md`, `runs/active/reframe/reproduction_check.csv`; `overleaf_drafts/tables/{d_ablation,e3_subspace_split,whiten_reduced}.tex`; `overleaf_drafts/figures/fig_d_ablation.pdf` |
| Tests | `tests/test_reframe_e3_h1_whiten.py`: synthetic-data checks run in CI; the byte-identical regeneration of the three tables and the figure, and the recomputation of one E3 row from the cached vectors, run where `runs/` and the caches exist and skip otherwise |

Regenerate the tables, the figure and the facts file from the committed CSVs
with `python scripts/paper/reframe/abtt_subspace_whiten.py render` (seconds,
no embeddings needed). The fine-tuned rows of the reproduction check in the
facts file need the gitignored `finetune_*_layer_results.csv`; pass them with
`--ft_csv` (the batch file does). Rerun everything with
`sbatch slurm/paper/reframe_e3_h1_whiten.sbatch` from the repo root.

### Protocol

- Mean-pooled hidden states (`hidden_mean_tokempty`), layers 1..L, 847 train and
  858 test passages. The train mean, the principal components, the whitening
  transform and D are all fit on the train passages only, then applied to both
  splits.
- Every metric block goes through `run_resubmit_evaluate.evaluate_from_similarity`,
  the evaluator's single definition of the reported metrics. Task A AUROC is
  sklearn `roc_auc_score` over all test pairs, same- against
  different-directory; tau is learned on train.
- ABTT is `EmbeddingCleaner`. Train-selected D is the first argmax of train
  DirAcc@1 over {1,2,3,5,7,10} (`find_optimal_D_phase11`). Train-selected layer
  is the first argmax of train AUROC.
- "Collapsed" follows the paper's definition (Sec. 4, "We call a layer collapsed
  when its baseline AUROC is below 0.70"): 26 layers, all T5. The top-PC share
  cutoff of 0.6 is the paper's label-free flag for those layers; it flags the
  same 26 plus mT5-base layer 4 (baseline AUROC 0.799). The rank-1 test is
  reported under both definitions, and the conclusion is the same.
- Two things would silently change numbers. `EmbeddingCleaner(num_components=0)`
  returns the input uncentered, so D=0 centers explicitly. sklearn
  `PCA(n_components=k)` with the default solver goes randomized for small k,
  so whitening pins `svd_solver="full"`.

### Reproduction check

The H1 and WHITEN runs recompute every published mean-pooled cell of
`runs/active/resubmit/results/phase_resubmit_results.csv` (100 model-layers per
method):

| Method | Cells | Max \|diff\| | Equal to 3 dp | D matches |
|---|---|---|---|---|
| baseline | 100 | 7.1e-08 | 100/100 | |
| abtt_fixed (D=10) | 100 | 3.0e-08 | 100/100 | 100% |
| abtt_optimal (train-selected D) | 100 | 3.0e-08 | 100/100 | 100% |
| E3 retained subspace vs abtt_optimal | 100 | 3.0e-08 | 100/100 | 100% |
| whitening (full rank) | 100 | 1.5e-03 | 81/100 | |
| fine-tuned baseline and abtt_optimal (`finetune_*_layer_results.csv`) | 128 | 4.6e-08 | | 100% |

Sample cells (published / ours): LaTa L7 baseline 0.498 / 0.498; LaTa L7 ABTT
0.962 / 0.962; LaTa L12 baseline 0.938 / 0.938; LaTa L12 ABTT 0.971 / 0.971;
PhilTa L9 ABTT 0.983 / 0.983; mT5-base L6 baseline 0.657 / 0.657; LaBSE L11
ABTT 0.987 / 0.987; Qwen3-0.6B L26 baseline 0.966 / 0.966.

The full-rank whitening cells differ in the fourth decimal (max 1.5e-3) only for
Qwen3-0.6B and KaLM-mini, where d > 847 and the last kept component has variance
between 1e-16 and 1e-12. Dividing by the square root of that amplifies float
noise, so the cells depend on BLAS threading. For the four 768-dimensional
models the difference is at most 4e-5. This is the ill-conditioning the paper
describes, seen directly.

## H1: how many directions

Test AUROC against D (`tab:d_ablation`, `fig:d_ablation`); medians over
model-layers.

| Layers | n | raw | D=0 | 1 | 2 | 3 | 5 | 7 | 10 | 15 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Collapsed T5 | 26 | 0.541 | 0.503 | 0.713 | 0.902 | 0.964 | 0.973 | 0.977 | 0.977 | 0.974 | 0.975 | 0.974 | 0.971 |
| Other T5 | 10 | 0.874 | 0.936 | 0.952 | 0.967 | 0.973 | 0.979 | 0.976 | 0.973 | 0.972 | 0.972 | 0.973 | 0.971 |
| Embedding-trained | 64 | 0.903 | 0.944 | 0.966 | 0.971 | 0.976 | 0.979 | 0.980 | 0.978 | 0.977 | 0.975 | 0.974 | 0.973 |
| All | 100 | 0.885 | 0.936 | 0.958 | 0.968 | 0.973 | 0.978 | 0.979 | 0.977 | 0.975 | 0.974 | 0.974 | 0.972 |

Rank-1 test at the 26 collapsed T5 layers (share of the D=10 gain over raw):

| D | Median | Min | Max | Layers >= 80% |
|---|---|---|---|---|
| 0 (centering) | -4% | -10% | 7% | 0/26 |
| 1 | 45% | 13% | 89% | 1/26 (LaTa layer 2) |
| 2 | 78% | 60% | | 12/26 |
| 3 | 96% | 80% | | 26/26 |
| 5 | 100% | 92% | | 26/26 |

Per model, D=1 recovers a median 45% at LaTa (37 to 89%, layers 2-11), 14% at
PhilTa (13 to 66%, layers 3-11) and 64% at mT5-base (56 to 69%, layers 5-11).
With the top-PC-share flag (27 layers, adding mT5-base layer 4) the numbers are
the same to a point (D=1 median 47%, 1/27 at 80%; D=3 at 26/27).

Reading. The prediction was that D=1 recovers at least 80 percent of the gain.
It does not. At the collapsed layers PC1 holds a median 93.8 percent of the
centered training variance (min 76.4), and PCs 2 to 10 together hold only 4.5
percent. Yet removing PC1 alone leaves a median AUROC of 0.713, and removing
PCs 2 and 3 as well gives 0.964. The variance is dominated by one direction,
but the nuisance that breaks cosine ranking spans two to three. Centering alone
does not help at collapsed layers (median 0.503; it slightly hurts LaTa and
PhilTa); at healthy layers it recovers a good share (other T5 0.874 to 0.936,
embedding-trained 0.903 to 0.944).

The grid boundary. Over the paper's grid, train DirAcc@1 picks D=10 at 87 of
100 model-layers (D=7: 10, D=5: 2, D=3: 1), matching the published counts. Over
{1, ..., 50} it would pick D >= 15 at 99 of 100 (D=15: 11, 20: 20, 30: 43,
50: 25). The effect on test AUROC is small: median -0.004, range -0.022 to
+0.023 against the paper-grid D. Beyond D=10 the AUROC curve is flat within
about 0.02 at every model-layer (largest rise: LaTa's middle layers, +0.02 at
D=30 to 50).

## E3: removed against retained subspace

`tab:e3_subspace_split`. Cosine ranking within each part of the vector; Var. is
the removed components' share of the centered train variance. PC1 in 1-D cosine
is a sign match; the facts file also gives -|s_i - s_j| for PC1, which does not
change the reading.

| Model | Layer | D | Var. | Raw | Centered | PC1 | PCs 2..D | PCs 1..D (removed) | Retained |
|---|---|---|---|---|---|---|---|---|---|
| LaTa | 12 | 10 | 0.35 | 0.938 | 0.972 | 0.670 | 0.825 | 0.868 | 0.971 |
| PhilTa | 9 | 10 | 0.99 | 0.539 | 0.501 | 0.462 | 0.545 | 0.487 | 0.983 |
| mT5-base | 2 | 10 | 0.57 | 0.822 | 0.910 | 0.610 | 0.735 | 0.751 | 0.975 |
| LaBSE | 11 | 10 | 0.48 | 0.960 | 0.958 | 0.647 | 0.932 | 0.909 | 0.987 |
| Qwen3-0.6B | 2 | 10 | 0.44 | 0.863 | 0.940 | 0.607 | 0.792 | 0.798 | 0.973 |
| KaLM-mini | 1 | 10 | 0.41 | 0.887 | 0.958 | 0.518 | 0.846 | 0.819 | 0.981 |
| Collapsed T5 (26), median | | 10 | 0.99 | 0.541 | 0.503 | 0.463 | 0.601 | 0.489 | 0.977 |
| LaTa, pre-trained | 12 | 10 | 0.35 | 0.938 | 0.972 | 0.670 | 0.825 | 0.868 | 0.971 |
| LaTa, fine-tuned | 12 | 10 | 0.33 | 0.984 | 0.985 | 0.697 | 0.935 | 0.953 | 0.970 |
| Qwen3-0.6B, pre-trained | 27 | 10 | 0.41 | 0.967 | 0.969 | 0.643 | 0.913 | 0.917 | 0.982 |
| Qwen3-0.6B, fine-tuned | 27 | 10 | 0.31 | 0.993 | 0.996 | 0.684 | 0.980 | 0.983 | 0.992 |
| KaLM-mini, pre-trained | 24 | 10 | 0.32 | 0.956 | 0.979 | 0.663 | 0.909 | 0.924 | 0.981 |
| KaLM-mini, fine-tuned | 24 | 10 | 0.23 | 0.997 | 0.996 | 0.699 | 0.977 | 0.978 | 0.994 |

The layer is each model's train-selected ABTT layer (it matches
`selected_layers.tex`). D is the train-selected D, which is 10 at every row of
the first block. In the fine-tuning block both rows sit at the fine-tuned
model's selected layer with D=10. Fine-tuned Qwen3-0.6B selects D=2 at layer 27;
there the removed subspace ranks at 0.855 against 0.994 retained (PCs 2..D,
here PC2 alone, at 0.672). LaTa and KaLM-mini select D=10.

Over all 100 pre-trained model-layers the retained subspace leads the removed
one by at least 0.080 in the T5 encoders and 0.041 elsewhere (Qwen3-0.6B layer
28). The removed subspace never exceeds 0.894 in a T5 encoder.

Reading.

1. The falsifier ("the removed subspace ranks as well as the retained one in
   the pre-trained T5s") is not met anywhere. At the collapsed layers the
   removed subspace is at chance (0.465 to 0.648), as the paper expected.
2. The removed subspace is not pure nuisance where the full vector is healthy.
   It carries topical signal at 0.74 to 0.94 over the 74 non-collapsed
   layers (LaTa layer 12: 0.868; LaBSE layer 11: 0.909), mostly in PCs 2..D.
   ABTT discards some signal along with the nuisance, but the retained part
   ranks better. Even at the collapsed layers PCs 2..D keep a little signal
   (median 0.601, up to 0.781 at LaTa layer 2).
3. Fine-tuning moves signal into the dominant directions. At the same layer and
   D=10 the removed subspace rises from 0.868 to 0.953 (LaTa), 0.917 to 0.983
   (Qwen3-0.6B) and 0.924 to 0.978 (KaLM-mini), within 0.02 of the retained
   subspace in all three. This is consistent with Rajaee and Pilehvar and with
   ABTT lowering fine-tuned LaTa from 0.984 to 0.970. The removed subspace still
   ranks below the retained one after fine-tuning, so ABTT does not throw away
   all the task signal. It removes a subspace that now carries almost as much
   signal as it keeps.

## WHITEN: reduced-rank whitening

`tab:whiten_reduced`. Every cell is read at its own train-selected layer
(highest training AUROC), as the paper does for every setting; layers in
brackets.

| Model | Base | ABTT | k=64 | k=128 | k=256 | full |
|---|---|---|---|---|---|---|
| LaTa | 0.938 [12] | 0.971 [12] | 0.969 [12] | 0.973 [10] | 0.983 [12] | 0.872 [11] |
| PhilTa | 0.939 [1] | 0.983 [9] | 0.974 [1] | 0.977 [2] | 0.979 [9] | 0.854 [8] |
| mT5-base | 0.838 [12] | 0.975 [2] | 0.971 [1] | 0.979 [1] | 0.974 [2] | 0.845 [2] |
| LaBSE | 0.956 [12] | 0.987 [11] | 0.979 [10] | 0.984 [11] | 0.976 [6] | 0.866 [12] |
| Qwen3-0.6B | 0.966 [26] | 0.973 [2] | 0.973 [17] | 0.978 [2] | 0.983 [4] | 0.594 [23] |
| KaLM-mini | 0.972 [23] | 0.981 [1] | 0.976 [2] | 0.981 [5] | 0.981 [4] | 0.562 [20] |

Whitening minus ABTT (train-selected D) at the same layer, all 100 model-layers:

| k | Median | Range | Whitening higher | Collapsed T5 layers, whitening AUROC |
|---|---|---|---|---|
| 64 | -0.006 | -0.020 to +0.005 | 12/100 | 0.960 to 0.979 |
| 128 | +0.000 | -0.009 to +0.013 | 51/100 | 0.966 to 0.979 |
| 256 | +0.000 | -0.009 to +0.021 | 52/100 | 0.969 to 0.984 |
| full | -0.358 | -0.425 to -0.098 | 0/100 | 0.820 to 0.876 |

Conditioning. Full rank keeps min(n, d) components: 768 for the T5 encoders and
LaBSE, 847 for Qwen3-0.6B (d = 1,024) and KaLM-mini (d = 896). Centering leaves
rank at most 846, so for those two the last component has variance 1e-16 to
1e-12, and the ratio of largest to smallest kept variance is 1.6e11 to 1.8e13.
LaBSE is numerically rank-deficient even at d = 768 (ratio 1.6e13 to 1.4e16);
the three T5 encoders reach 3e5 to 5e11. With k <= 256 the ratio is 14 to 4.5e3
at every layer of LaBSE, Qwen3-0.6B and KaLM-mini, and up to 1.8e8 at the
collapsed T5 layers, where PC1 is huge; whitening still works there, because the
dominant direction is scaled down to unit variance rather than amplified.

Reading. Reduced-rank whitening, fit on train, is on par with ABTT on every
model: k=128 and k=256 are within about 0.01 of ABTT at the train-selected
layer and tie it layer for layer (median difference 0.000). k=64 is slightly
below. Whitening also repairs every collapsed T5 layer. The exclusion of
whitening in Section 3 holds only for full rank. The claim that survives is
"a train-only projection or rescaling of the top directions restores every
layer", not "ABTT restores". This fits the handoff's "the repair itself is not
a contribution" row.

## Replacement LaTeX for each `\pending`

Scoped to the "wording that survives" column of the handoff: the repair is not
a contribution, "T5-specific" only as "in our panel", and "consistent with" for
the fine-tuning result. One sentence per line, no em-dashes, American spelling.

### Sec. 3 "Post-processing", line 217 (`WHITEN`)

Replace the line

```latex
Whitening to a reduced dimension, as \citet{su2021whitening} also do, is the fairer comparison: \pending{WHITEN: PCA whitening with $k \in \{64,128,256\}$, Task~A AUROC}.
```

with

```latex
Whitening to a reduced dimension, as \citet{su2021whitening} also do, is the fairer comparison.
Fit on training embeddings with $k \in \{64,128,256\}$ components, it performs on par with ABTT (Table~\ref{tab:whiten_reduced}).
With $k=128$ or $256$ it reaches Task~A AUROC 0.973--0.984 at each model's train-selected layer, against 0.971--0.987 for ABTT, and at every one of the 100 model-layers it is within 0.021 of ABTT at the same layer, collapsed layers included.
The repair therefore does not depend on ABTT in particular; we keep ABTT as the reference because its $D$ is the quantity Section~\ref{sec:localizing} varies.
```

and `\input{tables/whiten_reduced}` in Sec. 3 or the SIF-variants appendix.
Two further sentences change meaning under this result (not `\pending`, for
the integration pass): line 216 "also ranks below ABTT at every layer" and the
appendix paragraph "Why whitening is excluded" (line 839) should say "full-rank
PCA whitening" each time.

### Sec. 5 "Subspace split (E3)", line 489 (`E3`)

```latex
We find that the removed subspace ranks well below the retained one in every pre-trained model (Table~\ref{tab:e3_subspace_split}).
At the 26 collapsed T5 layers it ranks at chance (median AUROC 0.489; PC1 alone 0.463), while the retained subspace reaches 0.977.
Where the full vector is healthy, the removed subspace still carries topical signal, mostly in PCs 2 to $D$: at LaTa's layer 12 it scores 0.868 against 0.971 for the retained subspace, and over all 100 pre-trained model-layers the retained subspace leads by at least 0.08 in the T5 encoders and 0.04 in the other models.
ABTT therefore discards some signal with the nuisance, but less than it keeps, and the third falsifier below does not apply.
Fine-tuning narrows the gap.
At the fine-tuned models' layers and $D=10$, the removed subspace rises from 0.868 to 0.953 for LaTa, from 0.917 to 0.983 for Qwen3-0.6B, and from 0.924 to 0.978 for KaLM-mini, within 0.02 of the retained subspace in all three.
This is consistent with fine-tuning moving task signal into the dominant directions \citep{rajaee2021finetuning}, and with ABTT lowering fine-tuned LaTa's AUROC from 0.984 to 0.970.
```

### Sec. 5 "How many directions (H1)", line 501 (`H1`)

```latex
We find that the nuisance is not rank-1.
At the 26 collapsed T5 layers, $D=1$ recovers a median 45 percent of the $D=10$ AUROC gain (13 to 89 percent) and reaches the 80 percent threshold at only one layer, LaTa layer~2.
Centering alone recovers nothing (median $-4$ percent).
Two components recover a median 78 percent, and three recover at least 80 percent at every collapsed layer (median 96 percent, AUROC 0.964 against 0.977 at $D=10$).
This happens although the first component holds a median 94 percent of the centered training variance at these layers: one direction dominates the variance, but the nuisance that breaks cosine ranking spans two to three.
PhilTa needs the most components ($D=1$ recovers a median 14 percent) and mT5-base the fewest (64 percent).
Beyond $D=10$ the curve is flat to within about 0.02, and extending the selection grid to $D=50$ would move training selection to $D \ge 15$ at 99 of 100 model-layers while changing test AUROC by at most 0.023.
We therefore drop the one-direction wording for what breaks ranking.
```

Add `\input{tables/d_ablation}` in an appendix if there is room; the figure
already carries the worst-layer curves.

### Figure `fig:d_ablation`, lines 504-507

Replace the placeholder box and the caption with

```latex
\includegraphics[width=\columnwidth]{fig_d_ablation.pdf}
\caption{Task~A test AUROC (top) and training DirAcc@1 (bottom) against the number $D$ of removed principal components, one line per model at its lowest-AUROC baseline layer (in parentheses in the legend). Raw: no correction; $D=0$: centering on the training mean alone. ABTT is fit on training embeddings only; the dotted line marks $D=10$, the top of the selection grid. Filled markers and solid lines: T5 encoders. At the collapsed T5 layers one component restores part of the ranking and three restore nearly all of it.}
```

and delete the `% PLACEHOLDER FIGURE: filled by H1` comment.

### Discussion, line 668 (`H1`)

Replace

```latex
In the middle layers of the T5 encoders in our panel, one dominant direction \pending{H1: or a low-dimensional subspace} takes over the pooled vectors, and cosine ranking falls to or near chance.
```

with

```latex
In the middle layers of the T5 encoders in our panel, a low-dimensional subspace takes over the pooled vectors, and cosine ranking falls to or near chance.
One direction holds most of its variance, but removing it alone restores only part of the ranking; removing three restores nearly all of it.
```

## Where the current paper text disagrees with these results

For the integration pass. None of these lines is edited here.

1. **Sec. 5 H1 paragraph and Discussion**: the rank-1 prediction failed (above).
   Sentences built on "one direction" as the cause of the ranking failure need
   "a low-dimensional subspace" or "two to three directions": line 380 ("one
   direction takes most of the variance, and cosine ranking falls ...") is
   still true as a variance statement, but it should not imply that direction
   alone does the damage. Lines 393-395 and 430 ("reduces to one dominant
   direction") should be read against this.
2. **E1 reference arm (James)**: line 423 says "ABTT with $D=1$ serves as a
   reference, since it repairs a rank-one nuisance under either account". At
   the collapsed T5 layers D=1 does not repair them (median AUROC 0.713, range
   0.598 to 0.925). The E1 prediction "zeroing k <= 5 coordinates restores
   AUROC >= 0.90" should be compared with D=2 to D=3 of ABTT, not D=1.
3. **Sec. 3 whitening**: "ranks below ABTT at every layer" and the appendix
   "Why whitening is excluded" hold only for full rank. Reduced-rank whitening
   ties ABTT (median difference 0.000 at k=128 and 256).
4. **"ABTT restores every layer"** stays true, but reduced-rank whitening does
   too, so the claim should be about a train-only projection of the top
   directions, not about ABTT.
5. **ABTT removes signal at healthy layers**: the removed subspace ranks at
   0.74 to 0.94 where the full vector is healthy. "The signal was present but
   masked" is right for collapsed layers; at healthy layers ABTT also discards
   some signal, and still gains because what it removes ranks worse than what
   it keeps.
6. **Selected D sits on the grid boundary**: training DirAcc@1 would pick
   D >= 15 almost everywhere with a wider grid. The paper already notes the
   boundary; the new number is that it changes test AUROC by at most 0.023.
