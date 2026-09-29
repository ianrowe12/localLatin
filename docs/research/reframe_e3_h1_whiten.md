# Reframe experiments E3, H1 and WHITEN (issue #232)

Results for Ian's three CPU experiments in the analysis reframe
(`reframe_handoff_20260927.md`), with the replacement LaTeX for every
`\pending{E3 ...}`, `\pending{H1 ...}` and `\pending{WHITEN ...}` in
`overleaf_drafts/acl_latex.tex`. That file is not edited here; the text below is
for the single integration pass. Revised after the independent review of PR #237
(routing for WHITEN, per-model H1, dimension-matched E3 control).

Short version:

- **H1 fails its own test.** The nuisance at the collapsed T5 layers is not
  rank-1. Removing one component recovers a median 45 percent of the D=10 AUROC
  gain and passes the 80 percent bar at 1 of 26 collapsed layers. This holds
  even though PC1 carries a median 94 percent of the centered training
  variance there, and it is not an artefact of estimating PC1 on train. How
  many components the repair needs depends on the model: two in LaTa (median 98
  percent of the gain), three in PhilTa (95 percent), and more in mT5-base,
  which still lacks 15 percent at D=3 and keeps rising until D=10. Centering
  alone recovers nothing. The "one direction" wording has to go wherever it
  describes what breaks ranking.
- **E3 does not trigger the third falsifier.** In every pre-trained model the
  removed subspace ranks below the retained one, and at the six train-selected
  layers it also ranks below the next D components, a dimension-matched
  control (LaTa layer 12: 0.868 against 0.907). At the collapsed layers it is
  at chance (median 0.489). After fine-tuning the top D directions rank above
  the next D (LaTa layer 12: 0.953 against 0.900), which fits the Rajaee and
  Pilehvar account of the ABTT drop after fine-tuning. At LaTa layer 12 the
  pre-trained ABTT gain is centering alone (0.972).
- **WHITEN: reduced-rank whitening ranks on par with ABTT but routes below
  it.** With k=128 or 256 components, fit on train, Task A AUROC is 0.973 to
  0.984 at each model's train-selected layer against 0.971 to 0.987 for ABTT,
  and within 0.021 of ABTT layer for layer. On routing it falls short: at each
  model's train-selected Task B layer k=128 reaches DirAcc@1 1.9 to 4.5 points
  below ABTT (assignment 1.7 to 4.1 below), and k=256 4.6 to 15.0 below. The
  threshold does not degenerate as it does at full rank. So the ranking repair
  does not depend on ABTT in particular, and the routing result does.

## Provenance

| Item | Value |
|---|---|
| Script | `scripts/paper/reframe/abtt_subspace_whiten.py` (subcommands `h1`, `e3`, `whiten`, `pc1`, `render`) |
| Batch file | `slurm/paper/reframe_e3_h1_whiten.sbatch` (CPU partition, `beto-delta-cpu`, 16 cores, `--time=00:12:00`) |
| Jobs | 22534981 (first run: h1, e3, whiten): COMPLETED, elapsed 00:05:27 of 00:25:00 reserved (16 x 25 min = 6.7 core-hours reserved; TotalCPU 1:11:10). 22535798 (review round, full rerun with the E3 controls and the `pc1` step): COMPLETED, elapsed 00:06:41 of 00:12:00 reserved (16 x 12 min = 3.2 core-hours reserved; TotalCPU 1:18:13) |
| Pilot | one model-layer per subcommand on the login node, under a minute in all (H1 23 s, E3 7-9 s, WHITEN 8 s, pc1 2 s per model-layer at 4 BLAS threads), used to size `--time` |
| Inputs | `runs/active/resubmit/data/phase_resubmit_split.csv` (tracked since #231); cached mean-pooled hidden states `runs/active/resubmit_bases/phase9_bases/<slug>/hidden_mean_tokempty/` and `runs/active/resubmit_finetune_bases/phase9_bases/<slug>-ft/hidden_mean_tokempty/` from the HPC working copy |
| Alignment | both jobs' alignment gate (`verify_embedding_alignment.py`) passed: 19 caches manifest-verified, 38 byte-identical spot checks. Every matrix is loaded through `AlignmentResolver` (17 rows moved per cache, as in benchmark v1) |
| Outputs | `runs/active/reframe/h1/h1_d_ablation.csv` (1,200 rows), `runs/active/reframe/h1/h1_pc1_robustness.csv` (36 rows), `runs/active/reframe/e3/e3_subspace_split.csv` (198 rows), `runs/active/reframe/whiten/whiten_reduced.csv` (400 rows), `runs/active/reframe/facts_e3_h1_whiten.md`, `runs/active/reframe/reproduction_check.csv`; `overleaf_drafts/tables/{d_ablation,e3_subspace_split,whiten_reduced}.tex`; `overleaf_drafts/figures/fig_d_ablation.pdf` |
| Determinism | the rerun reproduced the H1 and WHITEN CSVs of the first run byte for byte |
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
  splits. The one exception is the `pc1` oracle below, which fits PCs on test
  on purpose and feeds no reported cell.
- Every metric block goes through `run_resubmit_evaluate.evaluate_from_similarity`,
  the evaluator's single definition of the reported metrics. Task A AUROC is
  sklearn `roc_auc_score` over all test pairs, same- against
  different-directory; tau is learned on train. Task B numbers in this memo
  are the evaluator's single-split DirAcc@1 and assignment accuracy, not the
  five-seed Task B protocol of the headline table.
- ABTT is `EmbeddingCleaner`. Train-selected D is the first argmax of train
  DirAcc@1 over {1,2,3,5,7,10} (`find_optimal_D_phase11`). The train-selected
  Task A layer is the first argmax of train AUROC, the Task B layer the first
  argmax of train DirAcc@1.
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
method). The independent reviewer of PR #237 also rebuilt the computation in
plain numpy with its own filename alignment and matched these cells to 1e-7.

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
| LaTa collapsed | 10 | 0.502 | 0.488 | 0.713 | 0.956 | 0.973 | 0.972 | 0.976 | 0.970 | 0.971 | 0.969 | 0.980 | 0.983 |
| PhilTa collapsed | 9 | 0.542 | 0.502 | 0.602 | 0.872 | 0.963 | 0.980 | 0.981 | 0.981 | 0.975 | 0.978 | 0.973 | 0.970 |
| mT5-base collapsed | 7 | 0.657 | 0.678 | 0.865 | 0.894 | 0.934 | 0.955 | 0.974 | 0.978 | 0.975 | 0.974 | 0.965 | 0.962 |
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

Per model (median share of the D=10 gain at the collapsed layers, minimum in
brackets):

| Model | Layers | D=1 | D=2 | D=3 | D=5 | D=7 |
|---|---|---|---|---|---|---|
| LaTa | 2-11 (10) | 45% (37) | 98% (92) | 101% (97) | 101% (99) | 102% (98) |
| PhilTa | 3-11 (9) | 14% (13) | 75% (60) | 95% (92) | 100% (98) | 100% (100) |
| mT5-base | 5-11 (7) | 64% (56) | 74% (73) | 85% (80) | 93% (92) | 98% (97) |

mT5-base keeps rising to D=10: AUROC 0.951 to 0.965 at D=5, 0.969 to 0.978 at
D=7 and 0.974 to 0.982 at D=10. With the top-PC-share flag (27 layers, adding
mT5-base layer 4) the pooled numbers barely move (D=1 median 47%, 1/27 at 80%;
D=3 at 26/27).

Robustness of the D=1 failure (`h1_pc1_robustness.csv`, subcommand `pc1`). It is
not a PC1 estimation artefact. At all 26 collapsed layers the train and test
PC1 agree to |cos| of at least 0.994 (min 0.9947). At LaTa layer 6, PhilTa layer 10 and
mT5-base layer 8 the agreement is 0.9993, 0.9958 and 1.0000, and removing a
test-fitted PC1 (an oracle) also fails, at AUROC 0.684, 0.666 and 0.869 against
0.671, 0.602 and 0.865 for the train-fitted PC1. Over the 26 layers the oracle
gains at most 0.069 over the train fit. After the train-fitted D=1 removal, the
next direction holds 66 percent (LaTa L6) and 74 percent (PhilTa L10) of the
remaining test variance, so a second strong direction is left behind; in
mT5-base L8 it holds 33 percent. Over all collapsed layers that share is 0.22
to 0.66 in LaTa, 0.20 to 0.75 in PhilTa and 0.24 to 0.54 in mT5-base.

Reading. The prediction was that D=1 recovers at least 80 percent of the gain.
It does not. At the collapsed layers PC1 holds a median 93.8 percent of the
centered training variance (min 76.4), and PCs 2 to 10 together hold only 4.5
percent. Yet removing PC1 alone leaves a median AUROC of 0.713. The variance is
dominated by one direction, but the nuisance that breaks cosine ranking spans
a few directions, two to three in LaTa and PhilTa and more in mT5-base.
Centering alone does not help at collapsed layers (median 0.503; it slightly
hurts LaTa and PhilTa); at healthy layers it recovers a good share (other T5
0.874 to 0.936, embedding-trained 0.903 to 0.944).

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
change the reading. Two dimension-matched controls sit inside the retained
subspace: Next D (PCs D+1 to 2D) and Rand. D (random D-dimensional projections
of the retained vectors, mean over five seeds).

| Model | Layer | D | Var. | Raw | Cent. | PC1 | PCs 2..D | PCs 1..D (removed) | Next D | Rand. D | Retained |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LaTa | 12 | 10 | 0.35 | 0.938 | 0.972 | 0.670 | 0.825 | 0.868 | 0.907 | 0.861 | 0.971 |
| PhilTa | 9 | 10 | 0.99 | 0.539 | 0.501 | 0.462 | 0.545 | 0.487 | 0.922 | 0.873 | 0.983 |
| mT5-base | 2 | 10 | 0.57 | 0.822 | 0.910 | 0.610 | 0.735 | 0.751 | 0.896 | 0.866 | 0.975 |
| LaBSE | 11 | 10 | 0.48 | 0.960 | 0.958 | 0.647 | 0.932 | 0.909 | 0.958 | 0.902 | 0.987 |
| Qwen3-0.6B | 2 | 10 | 0.44 | 0.863 | 0.940 | 0.607 | 0.792 | 0.798 | 0.906 | 0.882 | 0.973 |
| KaLM-mini | 1 | 10 | 0.41 | 0.887 | 0.958 | 0.518 | 0.846 | 0.819 | 0.910 | 0.880 | 0.981 |
| Collapsed T5 (26), median | | 10 | 0.99 | 0.541 | 0.503 | 0.463 | 0.601 | 0.489 | 0.919 | 0.874 | 0.977 |
| LaTa, pre-trained | 12 | 10 | 0.35 | 0.938 | 0.972 | 0.670 | 0.825 | 0.868 | 0.907 | 0.861 | 0.971 |
| LaTa, fine-tuned | 12 | 10 | 0.33 | 0.984 | 0.985 | 0.697 | 0.935 | 0.953 | 0.900 | 0.874 | 0.970 |
| Qwen3-0.6B, pre-trained | 27 | 10 | 0.41 | 0.967 | 0.969 | 0.643 | 0.913 | 0.917 | 0.909 | 0.884 | 0.982 |
| Qwen3-0.6B, fine-tuned | 27 | 10 | 0.31 | 0.993 | 0.996 | 0.684 | 0.980 | 0.983 | 0.969 | 0.927 | 0.992 |
| KaLM-mini, pre-trained | 24 | 10 | 0.32 | 0.956 | 0.979 | 0.663 | 0.909 | 0.924 | 0.909 | 0.855 | 0.981 |
| KaLM-mini, fine-tuned | 24 | 10 | 0.23 | 0.997 | 0.996 | 0.699 | 0.977 | 0.978 | 0.967 | 0.918 | 0.994 |

The layer is each model's train-selected ABTT layer (it matches
`selected_layers.tex`). D is the train-selected D, which is 10 at every row of
the first block. In the fine-tuning block both rows sit at the fine-tuned
model's selected layer with D=10. Fine-tuned Qwen3-0.6B selects D=2 at layer 27;
there the removed subspace ranks at 0.855 against 0.994 retained (PCs 2..D,
here PC2 alone, at 0.672). LaTa and KaLM-mini select D=10.

Over all 100 pre-trained model-layers the retained subspace leads the removed
one by at least 0.080 in the T5 encoders and 0.041 elsewhere (Qwen3-0.6B layer
28). The removed subspace never exceeds 0.894 in a T5 encoder. It ranks below
the next D components at 89 of 100 pre-trained model-layers (63 of the 74
non-collapsed ones), including all six train-selected layers.

Reading.

1. The falsifier ("the removed subspace ranks as well as the retained one in
   the pre-trained T5s") is not met anywhere. At the collapsed layers the
   removed subspace is at chance (0.465 to 0.648), as the paper expected.
2. The dimension-matched control sharpens this. At the selected layers of the
   pre-trained models the top D directions rank below the next D (LaTa layer
   12: 0.868 against 0.907), so per dimension they carry less retrieval signal
   than what follows them. They still carry some where the full vector is
   healthy (0.74 to 0.94 over the 74 non-collapsed layers), so ABTT does
   discard some retrieval signal there. At the collapsed layers PCs 2..D keep a
   little (median 0.601, up to 0.781 at LaTa layer 2).
3. At LaTa layer 12 the pre-trained ABTT gain comes from centering alone
   (0.972 centered, 0.971 after ABTT), so removing the top components costs
   0.001 before fine-tuning and 0.015 after it (0.985 centered, 0.970 after
   ABTT).
4. After fine-tuning the top D directions rank above the next D: 0.953 against
   0.900 for LaTa, 0.983 against 0.969 for Qwen3-0.6B and 0.978 against 0.967
   for KaLM-mini, within 0.02 of the whole retained subspace in all three. For
   LaTa this reverses the pre-trained order. For Qwen3-0.6B and KaLM-mini the
   pre-trained order at the same layers was already reversed (0.917 against
   0.909 and 0.924 against 0.909), so fine-tuning widens the lead rather than
   creating it; these two layers are among the 11 pre-trained exceptions. This
   is consistent with Rajaee and Pilehvar and with ABTT lowering fine-tuned
   LaTa from 0.984 to 0.970.

## WHITEN: reduced-rank whitening

`tab:whiten_reduced` (now a `table*` with Task B columns). Every cell is read at
its own train-selected layer for its task; layers in brackets. Task B values
are single-split evaluator numbers, not the five-seed Task B.

Task A AUROC:

| Model | Base | ABTT | k=64 | k=128 | k=256 | full |
|---|---|---|---|---|---|---|
| LaTa | 0.938 [12] | 0.971 [12] | 0.969 [12] | 0.973 [10] | 0.983 [12] | 0.872 [11] |
| PhilTa | 0.939 [1] | 0.983 [9] | 0.974 [1] | 0.977 [2] | 0.979 [9] | 0.854 [8] |
| mT5-base | 0.838 [12] | 0.975 [2] | 0.971 [1] | 0.979 [1] | 0.974 [2] | 0.845 [2] |
| LaBSE | 0.956 [12] | 0.987 [11] | 0.979 [10] | 0.984 [11] | 0.976 [6] | 0.866 [12] |
| Qwen3-0.6B | 0.966 [26] | 0.973 [2] | 0.973 [17] | 0.978 [2] | 0.983 [4] | 0.594 [23] |
| KaLM-mini | 0.972 [23] | 0.981 [1] | 0.976 [2] | 0.981 [5] | 0.981 [4] | 0.562 [20] |

Task B, DirAcc@1 / assignment accuracy in percent:

| Model | ABTT | k=64 | k=128 | k=256 | full |
|---|---|---|---|---|---|
| LaTa | 86.1 / 88.5 [8] | 81.2 / 84.7 [1] | 83.1 / 86.2 [1] | 77.0 / 80.7 [10] | 41.1 / 62.5 [2] |
| PhilTa | 88.3 / 91.1 [1] | 83.0 / 86.5 [1] | 86.4 / 88.8 [1] | 83.7 / 86.8 [9] | 41.0 / 62.5 [2] |
| mT5-base | 88.5 / 90.3 [1] | 78.9 / 82.6 [1] | 85.8 / 88.6 [1] | 83.8 / 87.2 [1] | 41.0 / 62.5 [1] |
| LaBSE | 88.5 / 90.8 [11] | 86.1 / 88.8 [11] | 84.0 / 86.7 [11] | 73.5 / 78.0 [12] | 40.3 / 62.5 [12] |
| Qwen3-0.6B | 89.4 / 91.5 [5] | 84.8 / 87.2 [26] | 85.0 / 88.0 [6] | 75.9 / 79.3 [20] | 20.5 / 62.4 [6] |
| KaLM-mini | 89.4 / 91.7 [3] | 82.6 / 85.4 [23] | 86.7 / 89.2 [2] | 84.5 / 87.8 [3] | 16.8 / 62.4 [3] |

Whitening minus ABTT (train-selected D) at the same layer, all 100 model-layers:

| k | AUROC median | AUROC range | DirAcc@1 median (>= ABTT) | Assignment median (>= ABTT) | tau range | Collapsed T5, whitening AUROC |
|---|---|---|---|---|---|---|
| 64 | -0.006 | -0.020 to +0.005 | -9.0 pts (0/100) | -6.9 pts (0/100) | 0.508-0.613 | 0.960 to 0.979 |
| 128 | +0.000 | -0.009 to +0.013 | -1.9 pts (21/100) | -1.4 pts (22/100) | 0.377-0.513 | 0.966 to 0.979 |
| 256 | +0.000 | -0.009 to +0.021 | -4.1 pts (15/100) | -3.3 pts (16/100) | 0.251-0.357 | 0.969 to 0.984 |
| full | -0.358 | -0.425 to -0.098 | -64.0 pts (0/100) | -25.6 pts (0/100) | 0.000-0.065 | 0.820 to 0.876 |

Conditioning. Full rank keeps min(n, d) components: 768 for the T5 encoders and
LaBSE, 847 for Qwen3-0.6B (d = 1,024) and KaLM-mini (d = 896). Centering leaves
rank at most 846, so for those two the last component has variance 1e-16 to
1e-12, and the ratio of largest to smallest kept variance is 1.6e11 to 1.8e13.
LaBSE is numerically rank-deficient even at d = 768 (ratio 1.6e13 to 1.4e16);
the three T5 encoders reach 3e5 to 5e11. With k <= 256 the ratio is 14 to 4.5e3
at every layer of LaBSE, Qwen3-0.6B and KaLM-mini, and up to 1.8e8 at the
collapsed T5 layers, where PC1 is huge; whitening still works there, because the
dominant direction is scaled down to unit variance rather than amplified.

Reading. Reduced-rank whitening, fit on train, ranks on par with ABTT: k=128
and k=256 are within about 0.01 of ABTT at the train-selected layer and tie it
layer for layer (median AUROC difference 0.000), and they repair every
collapsed T5 layer; k=64 is slightly below. On routing they do not match ABTT.
The learned threshold stays in a normal range (tau 0.25 to 0.61, against 0.000
to 0.065 at full rank), so reduced-rank whitening does not degenerate it, but
DirAcc@1 and assignment accuracy fall below ABTT at most layers and at every
model's train-selected Task B layer. The exclusion of whitening in Section 3
holds as written only for full rank. What survives is "on pairwise ranking, a
train-only projection or rescaling of the top directions restores every layer";
on routing ABTT remains the better correction, which justifies keeping it as
the reference.

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
Fit on training embeddings with $k=128$ or $256$ components, it ranks on par with ABTT (Table~\ref{tab:whiten_reduced}): Task~A AUROC 0.973--0.984 at each model's train-selected layer against 0.971--0.987, and within 0.021 of ABTT at every one of the 100 model-layers, collapsed layers included; $k=64$ falls slightly below.
On pairwise ranking the repair therefore does not depend on ABTT in particular.
On routing it does: reduced-rank whitening does not degenerate the threshold as full rank does, but at each model's train-selected Task~B layer $k=128$ reaches DirAcc@1 1.9 to 4.5 points below ABTT, and $k=256$ up to 15 points below.
We therefore keep ABTT as the reference.
```

and `\input{tables/whiten_reduced}` (a `table*`) in Sec. 3 or the SIF-variants
appendix. Line 216 needs no change: it is about full-rank whitening, which the
next sentence now contrasts with reduced rank.

### Sec. 5 "Subspace split (E3)", line 489 (`E3`)

```latex
We find that the removed subspace ranks below the retained one in every pre-trained model (Table~\ref{tab:e3_subspace_split}).
At the 26 collapsed T5 layers it ranks at chance (median AUROC 0.489; PC1 alone 0.463), while the retained subspace reaches 0.977.
Where the full vector is healthy the removed subspace still carries retrieval signal, mostly in PCs 2 to $D$, but the retained subspace leads by at least 0.08 in the T5 encoders and 0.04 in the other models at every pre-trained layer.
At the selected layers of the pre-trained models the top $D$ directions also rank below the next $D$ (LaTa layer 12: 0.868 against 0.907), so per dimension they carry less signal than what follows them; after fine-tuning they rank above it (0.953 against 0.900 for LaTa), and the third falsifier does not apply.
At LaTa's layer 12 the pre-trained gain from ABTT comes from centering alone (0.972), so removing the top components costs 0.001 before fine-tuning and 0.015 after it.
In fine-tuned Qwen3-0.6B and KaLM-mini the removed subspace also comes within 0.02 of the retained one (0.983 against 0.992, and 0.978 against 0.994, at $D=10$).
This is consistent with fine-tuning moving task signal into the dominant directions \citep{rajaee2021finetuning}, and with ABTT lowering fine-tuned LaTa's AUROC from 0.984 to 0.970.
```

### Sec. 5 "How many directions (H1)", line 501 (`H1`)

```latex
We find that the nuisance is not rank-1.
At the 26 collapsed T5 layers, $D=1$ recovers a median 45 percent of the $D=10$ AUROC gain (13 to 89 percent) and reaches the 80 percent threshold at only one layer, LaTa layer~2.
Centering alone recovers nothing (median $-4$ percent).
Two components recover a median 78 percent, and three recover at least 80 percent at every collapsed layer (median 96 percent, AUROC 0.964 against 0.977 at $D=10$).
This happens although the first component holds a median 94 percent of the centered training variance at these layers: one direction dominates the variance, but the nuisance that breaks cosine ranking spans a few directions, two to three in LaTa and PhilTa and more in mT5-base.
$D=1$ recovers the most in mT5-base (median 64 percent) and the least in PhilTa (14 percent), but mT5-base needs the most components overall: LaTa reaches the full gain with two (median 98 percent), PhilTa with three (95 percent), while mT5-base still lacks 15 percent at $D=3$ and keeps rising until $D=10$.
The failure of $D=1$ is not an estimation artifact: at LaTa layer~6, PhilTa layer~10 and mT5-base layer~8 the first component fit on training and on test passages agree to $|\cos| \ge 0.996$, removing a test-fitted first component also leaves AUROC at 0.684, 0.666 and 0.869, and after the training-fitted removal the next direction still holds 66 to 74 percent of the remaining test variance in LaTa and PhilTa.
Beyond $D=10$ the curve is flat to within about 0.02, and extending the selection grid to $D=50$ would move training selection to $D \ge 15$ at 99 of 100 model-layers while changing test AUROC by at most 0.023.
We therefore drop the one-direction wording for what breaks ranking.
```

`\input{tables/d_ablation}` (a `table*`, now with per-model medians over the
collapsed layers) can go in an appendix; the figure carries the worst-layer
curves.

### Figure `fig:d_ablation`, lines 504-507

Replace the placeholder box and the caption with

```latex
\includegraphics[width=\columnwidth]{fig_d_ablation.pdf}
\caption{Task~A test AUROC (top) and training DirAcc@1 (bottom) against the number $D$ of removed principal components, one line per model at its lowest-AUROC baseline layer (in parentheses in the legend). The horizontal axis is categorical: raw is the uncorrected vector, $D=0$ is centering on the training mean alone, and the ticks are not evenly spaced in $D$. ABTT is fit on training embeddings only; the dotted vertical line marks $D=10$, the top of the selection grid, and the gray horizontal line chance (AUROC 0.5). Filled markers and solid lines: T5 encoders; hollow markers and dashed lines: embedding-trained models. At the collapsed T5 layers one component restores part of the ranking, and two to five restore nearly all of it.}
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
One direction holds most of its variance, but removing it alone restores only part of the ranking; removing two to five restores nearly all of it.
```

## Integration list: other paper sentences these results change

For the later integration pass. None of these lines is edited here. Line
numbers are those of `acl_latex.tex` on main at `1b43b17` (unchanged from this
branch's base).

1. **Line 671 (Discussion)**: "A projection fit on training vectors removes the
   direction and restores ranking at every depth." Change "removes the
   direction" to "removes these few directions".
2. **Line 430 (Sec. 5)**: "The account then reduces to ``one dominant
   direction''" becomes "The account then reduces to ``a few dominant
   directions''".
3. **Line 654 (Sec. 7 "Where the repair costs")**: the sentence that poses the
   subspace question ("if fine-tuning moves task signal into the top
   components, then removing them removes signal (E3)") can now be answered.
   Append: "E3 finds that it does: at layer 12 the removed subspace of
   fine-tuned LaTa ranks at 0.953, against 0.868 before fine-tuning and 0.900
   for the next ten components."
4. **Line 839 (appendix "Why whitening is excluded")**: say what the mechanism
   is and restrict it to full rank. Replace the whitening half of the
   "different reasons" sentence with: "full-rank whitening also rescales the
   near-null directions that 847 training vectors cannot estimate (condition
   numbers $3\times10^{5}$ to $10^{16}$), which amplifies noise and degenerates
   the routing threshold; reduced-rank whitening drops them and does not
   degenerate it". Also "SIF and whitening fall short" becomes "SIF and
   full-rank whitening fall short", and "PCA whitening" at the start of the
   paragraph becomes "Full-rank PCA whitening".
5. **Line 837 (appendix intro)**: "the PCA whitening variant that
   Section~\ref{sec:postproc} excludes" becomes "the full-rank PCA whitening
   variant that Section~\ref{sec:postproc} excludes".
6. **Line 380**: "one direction takes most of the variance, and cosine ranking
   falls ..." is still true as a variance statement; do not let the following
   sentences imply that this one direction alone does the damage (lines
   393-395 are read against H1).
7. **Line 216**: no change needed (it concerns full-rank whitening).
8. **For James (E1 region; line 423 and the caption at line 454)**: E1 needs an
   ABTT $D=3$ reference column, because $D=1$ does not repair the collapsed
   layers (median AUROC 0.713, range 0.598 to 0.925). Suggested wording for
   line 423: "ABTT with $D=1$ and with $D=3$ serve as references: $D=1$ removes
   the dominant direction, and $D=3$ recovers at least 80 percent of the $D=10$
   gain at every collapsed T5 layer." The caption at line 454 ("ABTT with one
   component (ABTT$_{D=1}$), a reference that repairs a rank-one nuisance under
   either account") needs the same change. The E1 prediction "zeroing $k \le 5$
   coordinates restores AUROC $\ge 0.90$" should be compared with ABTT at
   $D=2$ to 3, not $D=1$.
9. **Selected D sits on the grid boundary**: training DirAcc@1 would pick
   D >= 15 almost everywhere with a wider grid. The paper already notes the
   boundary; the new number is that it changes test AUROC by at most 0.023.
