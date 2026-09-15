# Whole-paper review, 2026-09-15

Target: `overleaf_drafts/` at `origin/main` head `149db5f` (after #198, #199, #200, #202, #204,
#205). Second whole-paper review; the first is `whole_paper_review_20260914.md`. This one had a
shell: a fresh worktree at `149db5f`, `runs/` from the main checkout, Python for every number,
a fresh-aux LaTeX build, and every generator re-run against the committed tables.

Verdict up front. The numbers are right: every numeric claim in the main text, captions and
appendix prose recomputes from the tracked sources, and every generated table regenerates byte
for byte from its generator. Two sentences are wrong against the paper's own tables and block
the push; the rest is should-fix or nits.

Summary of findings (exact corrections in sections B and C):

Blocking
1. `figures/fig_attribution_rho_loo_main.tex` (generated) says ABTT improves rho_LOO "in all six
   cells"; the prose, the abstract, Table 4 and the figure itself say five of six with PhilTa
   MaRC a tie (0.469 to 0.445). The wording is hardcoded in
   `scripts/ig/build_main_attribution_artifacts.py` line 622. See B.6.
2. Section 6.3, fine-tuning paragraph: "gives the best uncorrected Task A row (AUROC 0.984)" is
   false since #205 put Qwen3-0.6B (fine-tuned) at 0.996 in the same table. See B.3.

Should-fix
3. Appendix H calls Qwen3-0.6B "the strongest encoder in the table that never saw Latin as a
   pre-training target"; KaLM-mini beats it on three of the four headline columns, and the
   pre-training claim is not verifiable. See B.3.
4. Appendix A, Labels: "pending confirmation with the project director" is a draft note inside
   the submission. See B.8.
5. Appendix A, Labels: 690 / 150 is not reproducible from the directory names under any rule
   tried here (646 / 194 strict, 694 / 146 loose). See A.1.
6. `build_headline_tables.py` with default arguments, and `slurm/resubmit/benchmark_v1_taska.sbatch`,
   drop the Qwen3-0.6B fine-tuned row from Tables 2 and 3 and revert the caption to one
   reference system. See A.10.
7. Section 5, MaRC hyperparameters: "200 Adam steps" is "up to 200"; learning rate, early
   stopping, lambda, gamma and the IG step count are unstated. See B.7.
8. Section 6.3 heading scopes the fine-tuning match to LaTa but not to routing; on Task A the
   fine-tuned LaTa row (0.984) beats zero-shot ABTT (0.971). See B.3.
9. "v2" split jargon in four captions. See C.5.

Nits: B.4 (1.7 vs 1.6), B.2 ("catastrophic"), C.4 (leftover files that travel to Overleaf),
C.5 (caption wording), C.6 (no data statement).

---

## A. Number audit

Method. `scratch/audit_20260915.py` (not committed; the recipe is in each row) reads the
tracked sources and applies the paper's selection rules: Task A layer = argmax of
`train_aucroc`, Task B layer = argmax of `train_dir_acc_at_1` (ties to the lowest layer, as
`build_headline_tables.py` does with `idxmax`), attribution layer = earliest layer within 0.5
points of the best train `dir_acc_at_1` under `abtt_optimal`. Sources:
`runs/active/resubmit/results/phase_resubmit_results.csv`,
`runs/active/resubmit/taskb_mseed/aggregated_results.csv`,
`runs/active/resubmit/results/finetune/*` and `runs/active/resubmit/finetune/{,qwen3_0.6b/}run_info.json`,
`runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv` (test split),
`runs/active/ig_examples_200pos_v1/attribution_metrics/summary_v2.csv` plus the paired statistics
from `v2_hidden/` via `build_main_attribution_artifacts.paired_cell_stats`,
`runs/active/resubmit/data/phase_resubmit_split.csv`,
`runs/active/resubmit/cluster_viz/cluster_silhouette_{main,appendix}.csv`, and
`data/canon_labelled/` for the label taxonomy.

Verdict key: OK = matches to the printed precision; NOTE = matches but needs a wording or
provenance change; MISMATCH = does not match.

### A.1 Dataset, split, protocol, provenance

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 3.3, Table 1 | 847 train / 858 test | 847 / 858 | OK |
| Abstract, Intro, 3.2 | 840 directories, 1,705 witnesses, sizes 1 to 10 | 840 / 1,705 / max 10 | OK |
| Table 1 | singletons 545: 272/273; doubletons 108: 108/108; multi 187: 467/477 | 545: 272/273; 108: 108/108; 187: 467/477 | OK |
| 3.3 | doubletons as whole folders, 54 train / 54 test | 0 mixed folders; 54 / 54 | OK |
| 3.3 | 565 positive train pairs, 596 positive test pairs | 565 / 596 | OK |
| Table 1 caption | 535 existing, 323 new test files | 535 / 323 | OK |
| 5 (Task B metrics), App G | 62.4 percent class prior | 535/858 = 62.35 | OK |
| 6.2, App K captions | 1,160 winnable witnesses | 1,160 | OK |
| App A, Transcription | 142 of the 1,705 witnesses from Paris, BnF, lat. 1454 | 142 files with siglum `BN1454` | OK |
| App A, Labels | 690 source keys, 150 other labels | strict key regex `^[A-Za-z]+\.\d+(\.\d+)*$`: 646 / 194; loose CCL-prefix rule (`CARL.501?.N`, `CTOU.567.16 (15)`, `DSIR.384.255 cap. 11`, `Capit.Martini.N` counted as keys): 694 / 146. Neither gives 690 / 150 | NOTE, see B.8 |
| App A, Labels | Apostolic Canons, biblical references, edition citations, unidentified sources; each unidentified source a singleton | `Can.apost.` 50 dirs; 7 labels containing "unidentified"/"unknown", all size 1 | OK |
| App A, Derivation | 279 units of BN2123 reproduced, 100 percent after whitespace normalisation | `data_derivation.md`: 279/279 normalised, 158/279 byte-identical, directory agreement 279/279 | OK (not recomputable without the uncommitted TEI export) |
| 4.1 | 12/12/12/12/28/24 layers | max layer per model in results CSV: 12, 12, 12, 12, 28, 24 | OK |
| Discussion | one SVD per layer, fit on 847 training files | `n_train` = 847 | OK |
| 5 | D grid {1,2,3,5,7,10} | values selected across the 6-model grid: {3,5,7,10}; fine-tuned Qwen selects 2 | OK |

### A.2 Task A headline (Table 2) from `phase_resubmit_results.csv`

All 24 (model, setting) cells of `tables/taskA_headline.tex` match layer, AUROC and gap at the
train-selected layer (also confirmed by the byte-identical regeneration in A.10).

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 6.1, Table 2 caption | baseline AUROC 0.838 (mT5-base) to 0.972 (KaLM-mini) | 0.8375 to 0.9715 | OK |
| Abstract, 6.1, Table 2 caption | ABTT band 0.971 to 0.987 | 0.9714 (LaTa) to 0.9866 (LaBSE) | OK |
| Abstract, 6.1 | gap 0.024 to 0.237 baseline; 0.525 to 0.611 ABTT | 0.0241 / 0.2375; 0.5252 / 0.6107 | OK |
| 6.1 | gains 0.137 mT5-base, 0.034 LaTa, 0.009 KaLM-mini | 0.1372, 0.0338, 0.0091 | OK |
| 6.1 | KaLM-mini best baseline AUROC 0.972 with gap 0.056; Qwen3-0.6B 0.966 with the smallest gap 0.024 | 0.972 / 0.056; 0.966 / 0.024, smallest | OK |
| 6.1 | ABTT raises every model's gap by at least 0.28 | smallest rise 0.2878 (LaTa) | OK |
| Intro, 6.1 | SIF-only never matches ABTT-only on either task for any model | true in all 12 Task A cells and all 12 Task B cells | OK |
| Intro | at each T5 encoder's worst baseline layer ABTT is within 1.0 point of that model's best ABTT layer | LaTa 0.92, PhilTa 0.11, mT5-base 0.44 points | OK |

### A.3 Task B headline (Table 3)

All 24 cells of `tables/taskB_headline.tex` match.

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, contribution 3, 6.3 | dir@1 spread 39.3 (46.6 mT5-base to 85.9 KaLM-mini) to 3.3 (86.1 to 89.4) | 39.28 (46.6 to 85.9) to 3.26 (86.1 to 89.4) | OK |
| 6.3, Table 3 caption | assignment spread 40.3 to the range 88.5 to 91.7; every ABTT cell above every baseline cell | 40.33; 88.46 to 91.72; min ABTT 88.46 > max baseline 87.53 | OK |
| 6.1 SIF paragraph | SIF-only adds 9.4 / 8.2 / 7.3 (LaTa / PhilTa / mT5-base), +0.1 LaBSE, -1.9 KaLM-mini, -6.3 Qwen3-0.6B | 9.44 / 8.16 / 7.34 / +0.12 / -1.86 / -6.29 | OK |
| 6.1 SIF paragraph | SIF before ABTT moves assignment by at most 1.8 either way | largest +1.75 (LaTa), -1.17 (LaBSE) | OK |
| Discussion | ABTT adds 3.5 to 9.1 dir@1 on non-T5 | LaBSE 6.99, Qwen 9.09, KaLM 3.50 | OK |
| Discussion | existing-versus-new wrong for roughly one file in ten | ABTT assignment error 8.3 to 11.5 percent | OK |

### A.4 Layerwise profile (6.2, Figures 1 and 2, Appendix B)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, Intro, 6.2 | baseline minima 0.496 LaTa L6, 0.538 PhilTa L10, 0.654 mT5-base L5 | 0.4957 (L6), 0.5380 (L10), 0.6537 (L5) | OK |
| 6.2 | ABTT at those layers 0.964 / 0.982 / 0.978 | 0.9637 / 0.9824 / 0.9779 | OK |
| 6.2 | non-T5 final-layer AUROC 0.956 / 0.958 / 0.956; minima 0.806 / 0.858 / 0.861 | L12 0.9562, L28 0.9580, L24 0.9558; L1 0.8057, L1 0.8578, L5 0.8614 | OK |
| Fig 2 caption | non-T5 never below 0.80 | min 0.8057 | OK |
| Intro, 6.2, Fig 2 | non-T5 improve close to monotonically | largest adjacent drop LaBSE -0.008 (L8), Qwen -0.009 (L28), KaLM -0.025 (L3); KaLM peaks L23 0.972 then 0.956 | OK (judgement) |
| Intro, Fig 2 caption | mT5-base loses more than half of its above-chance separation | (0.838 - 0.654) / (0.838 - 0.5) = 54 percent | OK |
| Abstract, Intro, Fig 1 caption | baseline gap under 0.29 everywhere, under 0.11 for Qwen3-0.6B | max 0.2864 (LaTa L1); Qwen max 0.0981 | OK |
| Fig 1 caption | ABTT lifts all six near 0.5 at every depth | per-model minimum ABTT gap 0.442 (mT5-base) to 0.525 | OK (judgement) |
| Fig 1 caption, App B | mT5-base gap rises mid-depth while AUROC falls | gap L1 0.027, L5 0.219, L8 0.249, L12 0.081; AUROC L5 0.654 | OK |
| Fig 1 caption | LaTa and PhilTa gap collapses toward zero mid-depth | LaTa min -0.074 (L8); PhilTa min 0.040 (L10) | OK |
| Fig 1 caption | SIF-only helps LaTa and PhilTa, leaves the other four near baseline | mean gap change +0.186, +0.074; mT5 -0.034, LaBSE 0.000, Qwen +0.007, KaLM +0.015 | OK |
| Intro, 5, 6.4, App B | train-only rule gives LaTa 7, PhilTa 1, mT5-base 1 | earliest within 0.005 of best train dir@1 under ABTT: 7 (argmax 8, 0.8607 vs 0.8595), 1, 1 | OK |
| 5, 6.4 | attribution layers use D = 10 | `abtt_optimal` selects D = 10 at LaTa 7, PhilTa 1, mT5-base 1 | OK |
| 6.4 | LaTa L7 AUROC 0.498 to 0.962, dir 0.302 to 0.868; PhilTa L1 0.939 to 0.977, 0.693 to 0.883; mT5-base L1 0.822 to 0.979, 0.451 to 0.885 | all six pairs match to three decimals | OK |
| Fig 3 caption (App B) | PhilTa collapsed retrieval layer L8 = train-AUROC argmax within 30 to 70 percent depth (L4 to L8); test AUROC 0.542; lowest test 0.538 at L10 | band argmax L8 (train 0.5803); test 0.5421; min L10 0.5380; figure panel prints gap 0.05 (CSV 0.046) | OK |

### A.5 Geometry (6.2, Table 5) from `geometry_per_layer.csv`, test split

Most anisotropic layer = argmax of `pc1_variance_ratio` over raw rows.

| Model | Layer | mean cosine | PC1 raw / ABTT | eff. rank raw / ABTT | cosine after ABTT | Verdict |
|---|---|---|---|---|---|---|
| LaTa | 8 | 0.230 | 0.956 / 0.036 | 1.34 / 168.40 | 0.0014 | OK (L4 is 0.9550 against L8 0.9557; near tie, see NOTE) |
| PhilTa | 6 | 0.584 | 0.862 / 0.039 | 1.83 / 155.02 | 0.0015 | OK |
| mT5-base | 5 | 0.314 | 1.000 / 0.044 | 1.00 / 151.74 | 0.0019 | OK |
| LaBSE | 8 | 0.878 | 0.506 / 0.044 | 17.88 / 147.50 | 0.0012 | OK |
| Qwen3-0.6B | 16 | 0.953 | 0.167 / 0.033 | 75.05 / 172.57 | 0.0011 | OK |
| KaLM-mini | 5 | 0.887 | 0.396 / 0.029 | 26.63 / 176.36 | 0.0011 | OK |

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, Intro, 6.2 | mean cosine 0.23 to 0.95 at the most anisotropic layer | 0.230 to 0.953 | OK |
| 6.2 | ABTT drives all six to within 0.002 of zero | max 0.0019 | OK |
| 6.2 | effective rank from as low as 1.00 to between 147.50 and 176.36 | 1.00; 147.50 to 176.36 | OK |
| 6.2 | top-PC shares 0.956, 0.862, 1.000 (T5) and 0.506, 0.167, 0.396 (non-T5) | match | OK |
| 6.2 | t-SNE silhouette over 1,160 winnable witnesses rises by 0.07 to 1.16 on the three T5 encoders | LaTa +1.158, PhilTa +0.074, mT5-base +0.724; n = 1,160; UMAP +1.066, +0.099, +0.672 | OK |
| App K | panels at the train-selected ABTT layers | figure panels print L=8, 1, 1 (main) and title "train-selected layer"; silhouette CSVs dated 2026-09-14 15:50 (PR #188) | OK |

NOTE on LaTa layer 8: the top-PC share at L8 (0.9557) beats L4 (0.9550) by 0.0007. Table 5 is
correct, but the "most anisotropic layer" for LaTa is a near tie and any re-extraction could
flip it. Not a paper change; worth knowing before someone re-runs the diagnostics.

### A.6 Five-seed Task B (6.3, Table 13, Table 16)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 6.3 | baseline spread 36.6, 50.7 mT5-base to 87.3 KaLM-mini, at the Base subscripts (L1, L1, L12, L11, L28, L23) | 73.1, 70.5, 50.7, 82.0, 82.4, 87.3; spread 36.60 | OK |
| 6.3 | SIF+ABTT spread 1.7, 88.9 LaTa to 90.6 LaBSE, std at most 1.0 | 88.92, 89.77, 89.34, 90.55, 89.29, 89.30; spread 1.63; std 0.51 / 0.77 / 0.49 / 0.97 / 0.70 / 0.45 | NOTE, see B.4 |
| Table 13 rows | Top-1 to Top-5 means and stds | byte-identical regeneration (A.10) | OK |
| 6.3, Discussion | every model exceeds 95 percent by Top-2 and gains little after Top-3 | Top-2 min 95.1; Top-3 to Top-5 gain at most 1.4 | OK |
| Table 13 / Table 16 captions | the reported layer is not always the layer with the highest five-seed mean | LaTa L7 89.6 > L12 88.9; PhilTa L7 90.0 > L1 89.8; mT5-base L1 89.9 > L2 89.3; Qwen L8 90.3 > L5 89.3; KaLM L2 89.3 = L1 | OK |

### A.7 Whitening (Appendix G)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| App G | 100 whitening cells; every file routed as existing (existing accuracy 1.000); new accuracy never above one file in 323; assignment at the 62.4 prior; different-directory mean at or below 0.015; prior beats baseline in 60 of 100 cells; dir@1 never above 0.415 | 100 cells; existing min 1.000; new max 0.0031 = 1/323; assignment 0.6235 to 0.6247; `diff_avg` max 0.0144; 60 of 100; dir@1 max 0.415 | OK |

### A.8 Fine-tuning ceilings (6.3, Tables 2, 3, 17, Appendix H)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 6.3, Tables 2/3/17 | LaTa fine-tuned: AUROC 0.984, assignment 83.4; +ABTT 87.8 / 85.2; ABTT lowers AUROC 0.984 to 0.970; gap 0.387 to 0.548 | 0.9839, 83.45; 87.76, 85.20; 0.9697; 0.3872 to 0.5478 | OK |
| 6.3 | below every zero-shot ABTT cell (88.5 to 91.7; 86.1 to 89.4) | 87.8 < 88.46; 85.2 < 86.13 | OK |
| 6.3 | "gives the best uncorrected Task A row (AUROC 0.984)" | Qwen3-0.6B (fine-tuned), uncorrected, is 0.996 in the same table | MISMATCH, see B.3 |
| 6.3, Tables 2/3 captions | Qwen3-0.6B fine-tuned + ABTT 92.1 / 91.4, above every zero-shot ABTT cell; AUROC 0.996 to 0.994; gap 0.684 to 0.716; fine-tuned alone 91.4 / 90.8 | 92.07 / 91.38 vs max zero-shot 91.72 / 89.39; 0.9961 to 0.9941; 0.684 to 0.716; 91.4 / 90.8 | OK |
| 6.3 | five seeds: 0.923 against 0.905 | 0.9227 +/- 0.0018 vs 0.9051 +/- 0.0036 (1.8 points, 4.9 SD of the zero-shot estimate) | OK |
| Table 17 notes | LaTa five seeds 0.877 vs 0.877 | 0.8766 vs 0.8767 | OK |
| 6.3, App H, Tables 2/3 captions | 499 of 565 pairs; 28 of 190 directories; 71 dev files; 206 of 535 = 38.5 percent | both `run_info.json`: 499 / 565; 28 dev + 162 fit = 190; 71; 206 / 535 = 38.50 | OK |
| App H, Table 17 caption | LaTa selects epoch 7 (last run, budget 8); D = 10 at every layer | selected 7, epochs run 7, budget 8; `D` unique {10} over 12 layers; optimal == fixed | OK |
| App H, Table 17 caption | Qwen selects epoch 3 of 6 run; 12 of 28 layers at the top of the grid; D = 2 at layer 27 | selected 3, run 6; D counts {10: 12, 7: 11, 3: 2, 2: 2, 5: 1}; L27 D = 2; gradient checkpointing on | OK |
| App H | recipe: temperature 0.05, AdamW 2e-5, weight decay 0.01, warmup 10 percent, clip 1.0, 16 pairs per batch, up to 8 epochs, patience 3, seed 42 | both `run_info.json` configs match | OK |
| App H | train loss 4e-05 by epoch 4; the flipped file at loss 6e-04; one file = 1.4 points | Qwen dev curve: epoch 3 5.8e-04, epoch 4 4.0e-05; 100/71 = 1.41 | OK |
| App H | Qwen pre-trained already routes all 71 dev files | epoch-0 dev dir@1 = 1.000 | OK |
| App H | "the strongest encoder in the table that never saw Latin" | KaLM-mini: baseline AUROC 0.972 vs 0.966, baseline dir@1 85.9 vs 80.3, ABTT assignment 91.7 vs 91.5, ABTT dir@1 89.4 = 89.4 | NOTE, see B.3 |

### A.9 Attribution (6.4, Tables 6 and 7, Figure 4) from the v1 run

All 24 cells of Table 4 match `summary_v2.csv`; the LaTa IG row of Table 18 was spot-checked
(tau 0.007 / 0.262, InsAUC gap 0.265 / 0.060, Suff@25 0.990 / 0.804, Comp@25 1.242 / 0.177).

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, contribution 4, 6.4, Table 4/18 captions | rho_LOO 5/6, DelAUC gap 4/6, tau_b agreeing in the same five, InsAUC gap 5/6 | 5/6, 4/6, tau 5/6 in the same cells, InsAUC 5/6 | OK |
| 6.4 | rho wins by 10.3 to 22.5 paired SE, the sixth a tie at 1.5 | paired z: LaTa IG 16.0, LaTa MaRC 17.0, PhilTa IG 17.5, mT5 IG 22.5, mT5 MaRC 10.3; PhilTa MaRC -1.47 | OK |
| 6.4, Table 4 caption | DelAUC gap: four wins, one further tie (LaTa MaRC) | paired z: LaTa MaRC -1.8 (tie), LaTa IG -8.1 (loss), PhilTa IG +27.7, PhilTa MaRC +7.7, mT5 IG +23.3, mT5 MaRC +18.3 | OK |
| Table 4 caption | random-order reference 0.692 to 0.961 | `del_auc_random_mean` 0.692 to 0.961 | OK |
| Table 4/18 captions | 195 to 199 ABTT pairs against 199 to 200 baseline | `del_auc_gap_n`: ABTT 195 to 199; baseline 199 to 200 | OK |
| 5, 6.4, Table 18 caption | sufficiency side fails the shuffled-attribution control in one baseline cell; rho and DelAUC gap pass every cell | `rand_ins_auc_gap_gap` < 0 only for mT5-base IG baseline (-0.024); `rand_del_auc_gap_gap` and `rand_loo_rho_gap` > 0 in all 12 | OK |
| Table 4 caption, 5 | DelAUC gap = random-order minus attribution-order area, positive when attribution beats chance | `src/attribution_metrics.py` line 593: `"del_auc_gap": rand_mean - attr` | OK |
| Figure 4 caption | "ABTT improves the leave-one-out rank-correlation signal in all six cells" | 5/6; PhilTa MaRC 0.469 to 0.445 | MISMATCH, see B.6 |
| 5 | MaRC runs 200 Adam steps per pair from zeta_0 = log 9 | `slurm/ig/run_attribution_200pos_v1.sbatch`: `--steps 200 --lr 0.1 --lambda_sparsity 0.01 --gamma_tv 0.001 --early_stop_thresh 0.01 --early_stop_min_steps 50`; `MaskOptimConfig.init_mask_logit = 2.197`; IG uses 40 steps | NOTE, see B.7 |

### A.10 Generator regeneration (default arguments, diff against the committed files)

Run from the worktree root with `runs/` pointing at the main checkout.

| Generator | Command | Files | Result |
|---|---|---|---|
| `scripts/resubmit/build_headline_tables.py` | bare | `taskA_headline.tex`, `taskB_headline.tex` | **DIFFERS**: the Qwen3-0.6B (fine-tuned) row is dropped and the reference caption reverts to "a reference system ... LaTa". Byte-identical only with `--finetune_csv` and `--finetune_run_info` passed for both models. `slurm/resubmit/benchmark_v1_taska.sbatch` line 75 passes neither. |
| `scripts/resubmit/build_per_layer_tables.py` | bare (audit CSVs to scratch) | nine per-layer tables incl. `taskB_ranking_appendix_mseed.tex` | identical |
| `scripts/resubmit/build_lasttok_comparison_table.py` | bare | `appendix_lasttok_comparison.tex` | identical |
| `scripts/resubmit/visualize_taskb_mseed.py` | `--agg_csv`, `--layer_select_csv`, `--out_dir` as in `benchmark_v1_taskb_mseed.sbatch` | `taskB_topk.tex` | identical |
| `scripts/ig/build_main_attribution_artifacts.py` | bare | `attribution_metrics_main.tex`, `attribution_metrics_secondary.tex`, `figures/fig_attribution_rho_loo_main.tex` | identical (the stamp resolves to `ig_examples_200pos_v1`; the figure caption defect is in the generator, so identity here does not clear it) |
| `scripts/ig/package_attribution_sweep_appendix.py` | `--strict` | the two sweep tables | identical |
| `scripts/ig/build_delauc_sensitivity_table.py` | bare | `attribution_delauc_sensitivity.tex` (not input) | identical |
| `scripts/resubmit/finetune_ceiling.py` | `--stages report --tex_extra_run LaTa:finetune_lata:...` as in `finetune_qwen_ceiling_eval.sbatch`, `--tex_out` to scratch | `finetune_ceiling.tex` | identical |

Fix for the headline generator: make `--finetune_csv` / `--finetune_run_info` default to both
runs (LaTa then Qwen3-0.6B, the order of the committed rows) instead of the single
`DEFAULT_FINETUNE_CSV`, pass both explicitly in `benchmark_v1_taska.sbatch`, and add the
headline tables to the byte-identity check in `tests/test_paper_table_generators.py`, which
currently pins only the attribution generators.

---

## B. Claims audit (sceptical ARR reviewer)

### B.1 Anisotropy universal versus the T5-only collapse

Consistent and scoped everywhere it appears. Abstract: "universal across the six models";
Intro and Discussion: "universal on this corpus"; Related Work cites Machina and Mercer for
"neither universal nor always harmful". Contribution 2 says "the mid-depth retrieval collapse
that only the three T5 encoders show"; Limitations says "holds for the three T5 encoders in
our set, and we cannot separate architecture, pre-training objective, and checkpoint". The
evidence supports the split (A.4): the three T5 minima are 0.496, 0.538, 0.654 and the three
non-T5 minima are 0.806, 0.858, 0.861. The "first report" sentence in the Discussion carries
"to our knowledge". The Related Work sentences about Razzhigaev et al. and Godey et al. (peak in
decoder-only models; T5 entered with encoder and decoder states concatenated) were not
re-verified against the papers here; they match `t5_anisotropy_lit.md` and the previous
review's reading.

### B.2 mT5-base wording

"LaTa and PhilTa fall to chance in their middle layers and mT5-base loses more than half of its
above-chance separation" (Intro, Figure 2 caption) is exact: 54 percent. Nit: the abstract's
"Catastrophic mid-depth collapse is not universal" leans on "catastrophic" for a model whose
minimum is 0.654; "A mid-depth collapse of retrieval is not universal" is what the numbers say
and matches the Intro sentence.

### B.3 Fine-tuning sentences

Evidence (A.8) supports every number. Three wording problems.

1. **Blocking.** 6.3: "Fine-tuning LaTa ... gives the best uncorrected Task A row (AUROC
   0.984)". Since #205, Qwen3-0.6B (fine-tuned) sits in Table 2 at 0.996 uncorrected. Replace
   with: "gives an uncorrected Task A AUROC of 0.984, above every zero-shot baseline row
   (0.838 to 0.972), and 83.4 assignment accuracy".
2. **Should-fix.** The paragraph heading "The parameter-free repair matches contrastive
   fine-tuning on LaTa, but not on Qwen3-0.6B" is scoped to LaTa, as required, but not to
   routing. On Task A the fine-tuned LaTa row beats zero-shot ABTT (0.984 against 0.971) and
   ABTT then costs it 1.4 points. "Matches" is true for routing (86.1 vs 85.2 single seed,
   0.877 vs 0.877 over five seeds). Heading: "The parameter-free repair matches contrastive
   fine-tuning on LaTa routing, but not on Qwen3-0.6B". Body: "it matches what the available
   supervision buys on routing at this training budget".
3. **Should-fix.** Appendix H: "the second model is the strongest encoder in the table that
   never saw Latin as a pre-training target". KaLM-mini has the higher baseline AUROC (0.972 vs
   0.966), baseline dir@1 (85.9 vs 80.3) and ABTT assignment (91.7 vs 91.5), and ties on ABTT
   dir@1 (89.4). "Never saw Latin as a pre-training target" is not verifiable from the Qwen3
   release notes (119 languages and dialects). Replace with: "so the second model is a
   multilingual decoder-only encoder with no Latin-specific pre-training, Qwen3-0.6B, whose ABTT
   directory accuracy at rank 1 (89.4) ties KaLM-mini for the best zero-shot cell".

"At this training budget" appears in 6.3, both headline captions, Table 17's caption and
Appendix H, and the 499 pairs are memorised by epoch 4 (loss 4e-05), so the scoping is
earned. The 38.5 percent statement (206 of 535 test queries whose directory supplied a
training pair) recomputes from both `run_info.json` files and is attached to both ceilings in
6.3 ("both ceilings are also flattered"), which is the right place.

### B.4 Five-seed spread 36.6 to 1.7

Both true at the Base and SIF+ABTT subscripts of Table 3, and the sentence now names those
subscripts. Every value quoted (73.1 ... 87.3; 88.9 ... 90.6) is a row of Table 16, so the
reader can find them. Nit: 1.7 is the difference of the printed 90.6 and 88.9; at full precision
it is 1.63, which prints as 1.6 under the convention the paper uses for 39.3 (39.28) and 3.3
(3.26). Say "1.6 points (88.9 for LaTa to 90.6 for LaBSE as printed)" or leave it and accept
the rounding-path inconsistency.

### B.5 Attribution claims

Stated at the right strength and consistent across the abstract, contribution 4, 6.4, the
Discussion, Limitations and the two table captions: five of six with the sixth a tie (paired
z = -1.47), DelAUC gap four of six with one further tie (LaTa MaRC, z = -1.8) and one loss
(LaTa IG), sufficiency-side metrics failing the shuffled control in one baseline cell (mT5-base
IG, -0.024) and kept in the appendix, cross-variant comparisons descriptive, masking scoped to
the pooled read-out at layer L. The caption sign (random-order minus attribution-order,
positive when attribution beats chance) matches `attribution_metrics.py`. The
sensitivity table (#202) is generated but not input, as decided.

### B.6 Blocking: the generated figure caption contradicts the prose

`overleaf_drafts/figures/fig_attribution_rho_loo_main.tex`: "ABTT improves the leave-one-out
rank-correlation signal in all six cells". The figure itself shows PhilTa MaRC moving left
(0.469 to 0.445); Table 4 marks the baseline in bold there; every prose sentence says five of
six. The string is hardcoded at `scripts/ig/build_main_attribution_artifacts.py` line 622 and
survived the #187 re-sample because the figure caption, unlike the table caption, is not
computed from the wins. Fix in the generator (the file is stamped and must not be hand-edited):
"ABTT raises rho_LOO in five of the six cells; the sixth, PhilTa MaRC, moves from 0.469 to
0.445, a tie within two standard errors (Table 4)." Derive the count from `_wins` and the tie
cell from `_ties_for`, as the table caption already does. While there, drop the `\textbf{}`
lead-in and "candidate-attribution result" (jargon nowhere else in the paper).

### B.7 Experimental Setup section

Section 5 carries protocol and metrics only: train-only fitting, the four settings, Task A and
Task B metrics, attribution metrics with the plain statement that the ERASER-style metrics were
dropped from the main table and why, then hyperparameters and the layer rule. Section 4 carries
methods only; the one result-flavoured sentence there ("on this corpus it degenerates the
learned routing threshold") is the exclusion rationale for whitening and is fine.

Should-fix, reproducibility: "MaRC runs 200 Adam steps per pair from zeta_0 = log 9" understates
the configuration the artifacts were built with (`run_attribution_200pos_v1.sbatch`): up to 200
Adam steps at learning rate 0.1, stopping after step 50 once the masked cosine is within 0.01
of the unmasked one, with lambda = 0.01 and gamma = 0.001 in Equation 1; integrated gradients
uses 40 integration steps. Equation 1 introduces lambda and gamma and the paper never gives
their values. Replace the clause with: "MaRC runs up to 200 Adam steps per pair (learning rate
0.1, early stop after 50 steps once the masked cosine is within 0.01 of the full one) from
zeta_0 = log 9 with lambda = 0.01 and gamma = 0.001, and integrated gradients uses 40 steps".

### B.8 Dataset section and Appendix A

Section 3.2 is two paragraphs (about 210 words), readable in under a minute. It says "We adapt
that material into a benchmark" and "we derive the plain-text files"; "construct" does not
occur anywhere in the paper. Glosses are present for TEI-P5, witness, fragment and source key;
"siglum" is confined to Appendix A and glossed there. Proofreading is scoped: "That proofreading
stage belongs to the post-2019 workflow." HTR on one manuscript is stated with its 142 witnesses.
The domain-expert paragraph Abigail is to supply (meeting 2026-09-14) is not in yet; the Intro
reads fine without it.

Two should-fix items in Appendix A, Labels.

1. "The assignment of witnesses to keys was carried out by the project; a description of the
   annotators, instructions and verification is pending confirmation with the project
   director." A reviewer reads "pending confirmation" as an unfinished paper. Until #193
   lands, write it as a stated limitation: "The assignment of witnesses to keys was carried
   out by the project; we report the keys as supplied and do not have a description of the
   annotators, their instructions or the verification applied."
2. "Of the 840 directory labels, 690 are such source keys. The other 150 are ..." Neither
   number reproduces from `data/canon_labelled/` (A.1: 646 / 194 under a strict key pattern,
   694 / 146 once annotated keys such as `CARL.501?.18` and `CTOU.567.16 (15)` count as keys).
   Either state the rule that gives 690 / 150 in the appendix, or write "about 690 ... the
   remaining 150 or so". The figures come from `story_memo_arr_oct.md`, which does not record a
   rule either.

### B.9 Lexical baselines and retrieval-quality overclaims

No residual mention: "lexical" occurs once, about tokenizer artifacts with no lexical content;
BM25, TF-IDF and Levenshtein occur nowhere. No sentence states or implies that the embeddings
beat string matching; the strongest retrieval sentences are numeric ("every model puts the
correct directory in a two-item shortlist for more than 95 percent of test queries", "the
existing-versus-new call is still wrong for roughly one file in ten") and both recompute.
"ABTT lifts every model into a narrow high-accuracy band" (Intro) is the one adjective; the
band is 86.1 to 89.4 and it is stated two sentences later, so it stands. `tables/lexical_baselines.tex`
remains on disk for the rebuttal, un-input.

### B.10 Density-figure rule

The Figure 3 caption states the rule (highest training-set AUROC within 30 to 70 percent of
encoder depth, layers 4 to 8), names the layer it picks (8), its test AUROC (0.542), and
distinguishes it from the most anisotropic layer (6) and the lowest-test-AUROC layer (10,
0.538). All four numbers recompute (A.4), the generator's band matches
(`run_resubmit_distributions.py` line 99), and the legend words "Same" and "Different" are
tied to "equivalent" and "non-equivalent" in the caption.

### B.11 Generated captions against prose

Checked every generated caption against the prose that cites it. One contradiction (B.6). The
rest agree: the headline captions repeat the 0.838 to 0.972, 0.971 to 0.987, 62.4, 499 of 565,
28-directory, 88.5 to 91.7 and 86.1 to 89.4 figures exactly; the Table 13 and Table 16 captions
state the same layer rule as 6.3; the last-token caption states the ABTT-subscript rule and
says mT5-base was not run; the Table 17 caption says "the 565 positive pairs available in the
train split" where the headline captions say "499 of the 565" (nit: harmonise to the latter).

### B.12 Overall

Soundness 4, Significance 3, Clarity 4, Novelty 3, unchanged from the first review. The
selection protocol is train-only everywhere a number is reported (the first review's three
test-selected artefacts were fixed in #188 and the regenerated figures print the train-selected
layers). The two blocking items are one-sentence fixes plus one generator line. Recommend:
fix 1 and 2, then push to the mirror; items 3 to 9 can ride the next paper PR.

---

## C. Presentation

### C.1 Fresh-aux build

`pdflatex; bibtex; pdflatex x3` on a copy of `overleaf_drafts/` with all aux files removed:
exit 0, 39 pages, 0 undefined references, 0 undefined citations, 0 overfull boxes, 0 LaTeX
warnings of any kind, no "Rerun" request, bibtex 0 warnings. Discussion ends on page 8
("one SVD per layer" is on page 8); Limitations and References start on page 9; Appendix A
starts on page 11. The committed `overleaf_drafts/acl_latex.pdf` (tracked on purpose) is also
39 pages and contains the post-#205 strings, so it is current.

### C.2 Cross-references, citations, floats, graphics

- Every `\ref` resolves. Labels defined and never referenced: `sec:related`, `sec:setup`,
  `sec:tasks` (harmless; `sec:layers` on the same section is the one used) and the two
  un-input tables `tab:attribution_delauc_sensitivity`, `tab:lexical_baselines`.
- All 50 cite keys used in the two `.tex` files exist in `custom.bib`. The bibliography
  renders: `CCL (2026a)` / `CCL (2026b)`, `Godey, de la Clergerie, and Sagot` (the first
  review's "de la" omission is fixed), `Liu et al. (2025)` for `liu2025medieval`.
- Every float is referenced: Figures 1 to 12, Tables 1 to 8 and the appendix tables.
- All 11 graphics referenced from `acl_latex.tex` and `fig_attribution_rho_loo_main.tex` exist
  under `overleaf_drafts/figures/` and are tracked in git (the `*.pdf` gitignore pattern is
  overridden for them), so a fresh clone builds. No absolute paths in any `.tex`; the only
  repository path is a `% Source: runs/...` comment in the un-input sensitivity table.
- `\input{tables/taskB_topk.tex}` is the one input with an explicit extension (cosmetic).

### C.3 Hard rules

- Em-dashes: none in `acl_latex.tex`, `related_work.tex` or any table file.
- Body bullets: none; contributions are inline (1) to (4).
- "family"/"families", "Latin department", "construct", "labeled": none.
- Proofreading scoped to post-2019: yes (App A). HTR on one manuscript: yes, with 142 witnesses.
- ARR anonymity: author block is the template placeholder; acknowledgments are commented out;
  the only personal names are inside citations of the CCL project (`firey2009ccl`,
  `eichbauer2014ccl`); the footnote says "project director, personal communication, 2026".
- Fine-tuning claim: scoped to LaTa in the heading, 6.3, Table 3 caption and Appendix H, with
  the Qwen result stated alongside (B.3 asks for the routing scope as well).
- Train-only selection: every table caption states its train rule; the per-layer tables bold
  the train-selected row; the t-SNE/UMAP panels print the train-selected layer.

### C.4 Leftovers that would travel to Overleaf

`scripts/paper/sync_paper_repo.sh push` tars the whole of `overleaf_drafts/`, so all of the
following reach the co-authors. None breaks the build.

- `overleaf_drafts/figures/`: 18 PDFs not referenced by any `.tex`
  (`fig_appendix_{aucroc,gap,taskb}_per_model`, `fig_attribution_methods_abtt_compare`,
  `fig_lexical_vs_embedding`, `fig_release_{aucroc,gap,taskb}_per_model`,
  `fig_retrieval_mark_labse_ex{21,22,24}`, `fig_{tsne,umap}_per_model`, `taskb_mseed_breakdown`,
  `taskb_mseed_table`, `taskb_mseed_topk_bar`, `taskb_rank_distribution_table`,
  `taskb_topk_table`) and their PNG twins; plus `taskb_mseed_table.tex`, `taskb_topk_table.tex`,
  `taskb_mseed_selected_configs.csv` (stale: `sif_abtt_fixed` rows at LaTa L1 and PhilTa L7,
  a selection the paper no longer uses), `taskb_rank_distribution.csv`,
  `taskb_top5_predictions.csv`, `taskb_topk_table.csv`.
- `overleaf_drafts/tables/`: `attribution_delauc_sensitivity.tex` (generated, not input, header
  says "Not \input anywhere yet"; a decision item from #202, not a defect) and
  `lexical_baselines.tex` (kept for the rebuttal, known).
- `overleaf_drafts/formatting.md` (the ACL template's instructions) and `acl_latex.pdf`.
- No TODO/FIXME markers and no stale decision comments remain in `acl_latex.tex`; the only
  comment blocks are the float-packing note, the `\clearpage` note and the commented
  acknowledgments. `tables/finetune_ceiling.tex` ends with a generated "Notes for whoever
  moves these rows into the paper" comment block that is now historical.

### C.5 Captions, terminology, wording nits

- "v2" appears in the Table 1 caption ("under the v2 varied-allocation protocol"), the Table 13
  caption ("at a fixed v2 split"), the Table 16 caption ("of the v2 train/test split") and the
  Table 8 caption ("single-seed v2 split"). It is an internal version tag the reader cannot
  resolve. Drop "v2" and "varied-allocation" (Section 3.3 already describes the protocol), or
  define it once in 3.3.
- Terminology is consistent: "labelled", "directory", "witness", "directory accuracy at rank
  1", "ABTT-only", "SIF-only", "retrieval-adapted MaRC", "mT5-base", "Qwen3-0.6B", "KaLM-mini".
  "Top-$K$" in the metric definition against "top-$k$" elsewhere is the one mixed case.
- Captions are self-contained: each headline caption defines its columns and its layer rule;
  the density caption explains its legend; the t-SNE captions state the layer rule and the
  silhouette population. Figure 4's caption is the one that needs rewriting (B.6).
- Appendix letters match every `\ref`: A provenance, B layer diagnostics, C score
  distributions, D per-layer T5, E last-token, F per-layer non-T5, G SIF and whitening, H
  reference systems, I attribution sweeps, J qualitative, K cluster geometry. (Section numbers
  in this review use the built PDF: Table 5 is the layer-diagnostics table, Table 13 the top-K
  table, Table 4 the attribution main table, Table 18 the secondary table, Table 17 the
  fine-tuning table, Table 16 the five-seed per-layer table, Figure 3 the density figure,
  Figure 4 the rho_LOO figure.)

### C.6 Not present

No data-availability or ethics statement; CCL licensing is still unconfirmed (CLAUDE.md), so
do not add a release promise yet. The ARR checklist will ask; a one-sentence data statement
can go in once licensing is settled.
