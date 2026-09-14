# Whole-paper review, 2026-09-14

Target: `overleaf_drafts/acl_latex.tex` plus `related_work.tex`, `custom.bib`, `tables/*.tex`,
`figures/*` in the main working copy at `1e001b9`. Reviewer persona for section B: the
Methodologist (train-only selection, overclaiming, caption/prose consistency).

Scope limits. This session had no shell, so no worktree, no LaTeX build, no Python. Every
number in section A was recomputed by reading the tracked CSVs directly and applying the
paper's stated selection rule by hand (argmax of `train_aucroc` for Task A, argmax of
`train_dir_acc_at_1` for Task B, ties to the lowest layer; `build_headline_tables.py`
lines 111, 444, 470 confirm `idxmax` on those two columns). Anything that needs a compiled
PDF is listed in section C as not run, with the command that settles it.

Sources read: `runs/active/resubmit/results/phase_resubmit_results.csv` (all 700 rows),
`runs/active/resubmit/taskb_mseed/aggregated_results.csv` (KaLM and Qwen blocks, plus the
generated `tables/taskB_ranking_appendix_mseed.tex` for the rest, spot-checked against the
CSV), `runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv` (all 200 test-split
rows), `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_v2.csv`
(all 54 rows), `runs/active/resubmit/results/lexical_baselines*.csv`,
`runs/active/resubmit/results/finetune/finetune_lata_ceiling_comparison.csv`,
`runs/active/resubmit/cluster_viz/cluster_silhouette_{main,appendix}.csv`,
`runs/active/resubmit/data/phase_resubmit_split.csv` (row counts by grep),
`docs/research/{lexical_vs_embedding,finetune_ceiling,attribution_metrics_decision,benchmark_v1,t5_anisotropy_lit,plan_20260906,story_memo_arr_oct,paper_writing_guidelines}.md`,
the generators `build_headline_tables.py`, `build_per_layer_tables.py`,
`run_resubmit_distributions.py`, `visualize_clusters_2d.py`,
`build_lasttok_comparison_table.py`, the sbatch files that drive them, and the PNG
renders of every main-text and appendix figure.

---

## A. Number audit

Verdict key: OK = matches the source to the printed precision; NOTE = matches but the
wording or provenance needs attention; MISMATCH = does not match.

### A.1 Dataset, split, protocol

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, Intro, 2.2 | 840 labelled fragments, 1,705 witnesses | benchmark_v1.md: 840 dirs, 1,705 files; split CSV 847+858 = 1,705 | OK |
| 2.3, Table 1 | 847 train / 858 test | grep split CSV: 847 `"train"`, 535+323 = 858 test | OK |
| 2.3, Table 1 | 545 singletons 272/273; 108 doubletons 108/108; 187 multi 467/477 | benchmark_v1 histogram: 1:545, 2:108, 3..10: 32+25+79+22+16+9+3+1 = 187; column sums 847 / 858 | OK |
| 2.3 | 54 doubleton dirs to train, 54 to test | 108 dirs split as whole folders, 108 files each side | OK |
| 2.3 | 565 positive train pairs, 596 positive test pairs | benchmark_v1.md corrected split: 565 / 596 | OK |
| Table 1 caption | 535 existing, 323 new test files | grep: 535 rows `"test",*,True,`; 323 rows `"test",*,False,`; CSV `n_existing=535,n_new=323` | OK |
| 2.2 | directory sizes 1 to 10 | histogram max 10 | OK |
| 4.5, Table 2 caption, App. F | 62.4 percent class prior | 535/858 = 0.6235 | OK |
| 2.3 | dev slice carved out of train by directory | finetune_ceiling.md: 28 of 190 train dirs, 71 files | OK |
| 5.3 fine-tune para | fine-tuned on 565 positive train pairs | 565 available; 499 used after the dev carve (finetune_ceiling.md) | NOTE, see B.7 |
| Discussion | one SVD per layer, fit on 847 training files | n_train = 847 | OK |

### A.2 Task A headline (Table `tab:taskA_headline`) recomputed from `phase_resubmit_results.csv`

Layer = argmax `train_aucroc`; value = test `aucroc` / `gap` at that layer.

| Model | Base | SIF | ABTT | SIF+ABTT | Verdict |
|---|---|---|---|---|---|
| LaTa | L12 0.9376 / 0.237 | L1 0.9475 / 0.366 | L12 0.9714 / 0.525 | L7 0.9656 / 0.580 | OK |
| PhilTa | L1 0.9393 / 0.231 | L1 0.9546 / 0.319 | L9 0.9826 / 0.540 | L8 0.9818 / 0.573 | OK |
| mT5-base | L12 0.8375 / 0.081 | L1 0.8454 / 0.036 | L2 0.9747 / 0.536 | L1 0.9741 / 0.519 | OK |
| LaBSE | L12 0.9562 / 0.183 | L12 0.9557 / 0.184 | L11 0.9866 / 0.611 | L10 0.9759 / 0.580 | OK |
| Qwen3-0.6B | L26 0.9664 / 0.024 | L22 0.9419 / 0.053 | L2 0.9726 / 0.553 | L2 0.9723 / 0.540 | OK |
| KaLM-mini | L23 0.9715 / 0.056 | L23 0.9640 / 0.056 | L1 0.9806 / 0.568 | L2 0.9811 / 0.535 | OK |

Prose derived from it:

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 5.1 | baseline AUROC 0.838 (mT5) to 0.972 (KaLM) | 0.8375, 0.9715 | OK |
| Abstract, 5.1, Table 3 caption | ABTT band 0.971 to 0.987 | 0.9714 (LaTa) to 0.9866 (LaBSE) | OK |
| Abstract, 5.1 | gap 0.024 to 0.237 baseline; 0.525 to 0.611 ABTT | Qwen 0.0241 / LaTa 0.2375; LaTa 0.5252 / LaBSE 0.6107 | OK |
| 5.1 | gains 0.137 (mT5), 0.034 (LaTa), 0.009 (KaLM) | 0.1372; 0.9714-0.9376 = 0.0338; 0.0091 | OK (LaTa rounds to 0.034 on full precision, 0.033 on the printed values) |
| 5.1 | ABTT raises every gap by at least 0.28 | smallest is LaTa 0.5252-0.2375 = 0.288 | OK |
| Intro | worst baseline layer, ABTT within 1.0 pt of that model's best ABTT layer | LaTa L6 0.9637 vs best 0.9729 (L11): 0.92; PhilTa L10 0.9824 vs 0.9835 (L8): 0.11; mT5 L5 0.9779 vs 0.9823 (L9): 0.44 | OK |
| 5.1 SIF para | SIF-only never matches ABTT-only on either task | true in all 12 cells of both tables | OK |

### A.3 Task B headline (Table `tab:taskB_headline`)

Layer = argmax `train_dir_acc_at_1`; value = test `overall_assignment_acc` / `dir_acc_at_1`.

| Model | Base | SIF | ABTT | SIF+ABTT | Verdict |
|---|---|---|---|---|---|
| LaTa | L1 73.8 / 72.1 | L1 83.2 / 81.8 | L8 88.5 / 86.1 | L12 90.2 / 88.2 | OK |
| PhilTa | L1 70.9 / 69.3 | L1 79.0 / 77.7 | L1 91.1 / 88.3 | L1 91.4 / 88.9 | OK |
| mT5-base | L12 47.2 / 46.6 | L1 54.5 / 54.1 | L1 90.3 / 88.5 | L2 90.3 / 87.5 | OK |
| LaBSE | L11 83.3 / 81.5 | L12 83.4 / 81.9 | L11 90.8 / 88.5 | L11 89.6 / 86.9 | OK |
| Qwen3-0.6B | L28 82.3 / 80.3 | L27 76.0 / 74.2 | L5 91.5 / 89.4 | L5 90.8 / 88.6 | OK |
| KaLM-mini | L23 87.5 / 85.9 | L22 85.7 / 84.0 | L3 91.7 / 89.4 | L1 90.8 / 88.0 | OK |

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, contrib. 3, 5.3 | dir@1 spread 39.3 -> 3.3 | 85.9-46.6 = 39.3; 89.4-86.1 = 3.3 | OK |
| 5.3 | assignment spread 40.3 -> 88.5 to 91.7; every ABTT cell above every baseline cell | 87.5-47.2 = 40.3; min ABTT 88.46 > max base 87.53 | OK |
| 5.1 SIF para | SIF adds 9.4 / 8.2 / 7.3 (LaTa/PhilTa/mT5), +0.1 LaBSE, -1.9 KaLM, -6.3 Qwen | 9.44 / 8.16 / 7.34 / +0.12 / -1.87 / -6.29 | OK |
| 5.1 SIF para | SIF before ABTT moves assignment by at most 1.8 either way | max is LaTa +1.75 | OK |
| Discussion | ABTT adds 3.5 to 9.1 dir@1 on non-T5 | LaBSE 7.0, Qwen 9.1, KaLM 3.5 | OK |
| Discussion | existing-vs-new wrong for roughly one in ten | assignment 88.5 to 91.7 | OK |

### A.4 Layerwise profile (5.2, Figs 1 and 2)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, 5.2 | baseline AUROC minima 0.496 LaTa L6, 0.538 PhilTa L10, 0.654 mT5 L5 | 0.4957 (L6; L5 0.4965), 0.5380 (L10), 0.6537 (L5) | OK |
| 5.2 | ABTT at those layers 0.964 / 0.982 / 0.978 | 0.9637 / 0.9824 / 0.9779 | OK |
| 5.2 | non-T5 baseline rises to 0.956 / 0.958 / 0.956 (last layer); minima 0.806 / 0.858 / 0.861 | L12 0.9562; L28 0.9580; L24 0.9558; minima L1 0.8057, L1 0.8578, L5 0.8614 | OK, but see B.4: KaLM peaks at L23 0.972 and Qwen at L27 0.967 before the last-layer drop |
| Fig 2 caption | non-T5 never below 0.80 | min 0.806 | OK |
| Fig 1 caption, abstract, intro | baseline gap under 0.29 everywhere; under 0.11 for Qwen | max baseline gap LaTa L1 0.2864; Qwen max L28 0.0981 | OK (0.286 is close to the stated bound; say "under 0.29" only) |
| Fig 1 caption | ABTT lifts all six near 0.5 at every depth | min ABTT gap mT5 L12 0.442, KaLM L13 0.467 | OK ("near 0.5" is fair) |
| Fig 1 caption, App. A | mT5 baseline gap rises in the middle while AUROC falls | gap L1 0.027, L5 0.219, L8 0.249, L12 0.081; AUROC L5 0.654 | OK |
| 3.2, 5.4 | operational rule (earliest layer within 0.5 pp of best train dir@1 under ABTT) gives LaTa 7, PhilTa 1, mT5 1 | LaTa: best L8 0.8607, L7 0.8595 is earliest within 0.005; PhilTa L1 0.8867 best; mT5 L1 0.8784 best | OK |
| 5.4 | LaTa L7 AUROC 0.498 -> 0.962, dir 0.302 -> 0.868; PhilTa L1 0.939 -> 0.977, 0.693 -> 0.883; mT5 L1 0.822 -> 0.979, 0.451 -> 0.885 | 0.4975/0.9618, 0.3019/0.8683; 0.9393/0.9768, 0.6935/0.8834; 0.8216/0.9794, 0.4510/0.8846 | OK |
| Fig 3 caption (App. B) | PhilTa L8 = mid-depth layer with highest train AUROC; test AUROC 0.542; lowest test AUROC 0.538 at L10 | generator uses 30 to 70 percent depth = L4..L8; train AUROC there peaks at L8 0.5803; test 0.5421; L10 0.5380 | OK, but the caption must say "30 to 70 percent depth", because L3 (train 0.6077) beats L8 if "mid-depth" is read loosely |

### A.5 Geometry (5.2, Table 4 in App. A) from `geometry_per_layer.csv`, test split

Most anisotropic layer = argmax `pc1_variance_ratio` over raw test rows.

| Model | Layer (argmax PC1) | mean cosine raw | PC1 raw | PC1 ABTT | eff. rank raw -> ABTT | cosine ABTT | Paper | Verdict |
|---|---|---|---|---|---|---|---|---|
| LaTa | 8 (0.9557 vs L4 0.9550) | 0.2300 | 0.956 | 0.036 | 1.34 -> 168.40 | 0.0014 | 8; 0.230; 0.956; 0.036; 1.34->168.40 | OK |
| PhilTa | 6 | 0.5836 | 0.862 | 0.039 | 1.83 -> 155.02 | 0.0015 | matches | OK |
| mT5-base | 5 | 0.3144 | 0.99986 | 0.044 | 1.00 -> 151.74 | 0.0019 | 5; 0.314; 1.000; 0.044; 1.00->151.74 | OK |
| LaBSE | 8 | 0.8785 | 0.506 | 0.044 | 17.88 -> 147.50 | 0.0012 | 0.878; 0.506; 147.50 | OK |
| Qwen3-0.6B | 16 | 0.9529 | 0.167 | 0.033 | 75.05 -> 172.57 | 0.0011 | 0.953; 0.167 | OK |
| KaLM-mini | 5 | 0.8872 | 0.396 | 0.029 | 26.63 -> 176.36 | 0.0011 | 0.887; 0.396; 176.36 | OK |

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, intro | mean cosine 0.23 to 0.95 at the most anisotropic layer | 0.230 to 0.953 | OK |
| 5.2 | ABTT drives all six to within 0.002 of zero | max 0.0019 (mT5) | OK |
| 5.2 | effective rank from as low as 1.00 to between 147.50 and 176.36 | 1.00 (mT5); 147.50 (LaBSE) to 176.36 (KaLM) | OK |
| 5.2 | silhouette rises by 0.07 to 0.72 on the three T5 encoders (baseline -> ABTT D=10) | t-SNE: LaTa +0.155, PhilTa +0.074, mT5 +0.724; UMAP: +0.152, +0.099, +0.672 | OK (min 0.074, max 0.724) |

### A.6 Five-seed Task B (5.3, Table `tab:taskb`, `tab:taskB_ranking_appendix_mseed`)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 5.3 | baseline spread 36.6, 50.7 mT5 to 87.3 KaLM, at train-selected layers | baseline at Base subscripts (L1, L1, L12, L11, L28, L23): 73.1, 70.5, 50.7, 82.0, 82.4, 87.3 -> 36.6 | OK, see B.6 on which rows carry it |
| 5.3, story memo | SIF+ABTT spread 1.7, 88.9 LaTa to 90.6 LaBSE, std at most 1.0 | at SIF+ABTT subscripts (12, 1, 2, 11, 5, 1): 88.9, 89.8, 89.3, 90.6, 89.3, 89.3; std 0.5/0.8/0.5/1.0/0.7/0.5 | OK |
| 5.3, Discussion | every model exceeds 95 percent by Top-2 | 96.1, 95.8, 95.1, 95.5, 95.7, 95.1 | OK |
| Table 5 caption | topk rows are the bold rows of the mseed appendix table | spot-checked KaLM L1 (0.596/0.893 +- 0.012/0.005), Qwen L5 (0.487/0.893) against `aggregated_results.csv` | OK |
| Table 5 caption | not always the layer with the highest five-seed mean | LaTa L7 0.896 > L12 0.889; Qwen L8 0.903 > L5 0.893 | OK |

### A.7 Lexical and fine-tuning (5.3 paragraphs, Tables 2 and 3 reference rows)

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| 5.3, Table 2 caption | TF-IDF AUROC 0.987 vs best ABTT 0.987 | lexical CSV 0.9869; LaBSE ABTT 0.9866 | OK |
| 5.3, Table 3 caption | 91.8 vs 91.7 assignment; 89.9 vs 89.4 dir@1 | 0.9184 vs KaLM 0.9172; 0.8986 vs 0.8939 | OK |
| 5.3 | recall@1 0.878 vs 0.854, McNemar p = 0.001; low-overlap p = 0.023, 0.033; routing edge p = 0.33 | lexical_vs_embedding.md: 0.878/0.854, p = 0.0011; 0.023; 0.033; 0.33 | OK (p-values not recomputable without Python; taken from the memo, which prints the discordant counts) |
| 5.3 | BM25 AUROC 0.974, routes no better than the 62.4 prior | 0.9738; assignment 62.5 | OK (62.5 is 0.1 above the prior; "no better than" is fair) |
| Table 2/3 rows | Levenshtein 0.952 / 0.362 / 87.8 / 86.2 | 0.9524 / 0.3622 / 87.76 / 86.25 | OK |
| 5.3 fine-tune | 0.984 AUROC, 83.4; with ABTT 87.8 and 85.2; ABTT 0.984 -> 0.970, gap 0.387 -> 0.548 | comparison CSV: 0.9839, 83.45; 87.76, 85.20; 0.9697; 0.3872 -> 0.5478 | OK |
| 5.3 fine-tune | below every zero-shot ABTT cell (88.5 to 91.7; 86.1 to 89.4) | min ABTT 88.5 > 87.8; 86.1 > 85.2 | OK |
| Tables 2/3 caption | "trained contrastively on the 565 positive train pairs" | 499 pairs trained on after the 28-directory dev carve | NOTE, see B.7 |

### A.8 Attribution (5.4, Tables 6 and 7) from `summary_v2.csv`

| Cell | rho_LOO base -> ABTT | DelAUC gap base -> ABTT | tau base -> ABTT | Table | Verdict |
|---|---|---|---|---|---|
| LaTa IG | 0.0134 -> 0.2984 | 0.5040 -> 0.1777 | 0.0067 -> 0.2122 | 0.013/0.298; 0.504/0.178; 0.007/0.212 | OK |
| LaTa MaRC | 0.0700 -> 0.3969 | 0.2444 -> 0.2868 | 0.0589 -> 0.2921 | matches | OK |
| PhilTa IG | 0.1445 -> 0.5775 | 0.8056 -> 0.4204 | 0.1061 -> 0.4323 | matches | OK |
| PhilTa MaRC | 0.1790 -> 0.3669 | 0.7576 -> 0.3528 | 0.1326 -> 0.2576 | matches | OK |
| mT5 IG | 0.1377 -> 0.6263 | 0.3942 -> 0.5476 | 0.1022 -> 0.4648 | matches | OK |
| mT5 MaRC | 0.2718 -> 0.4150 | 0.0750 -> 0.4309 | 0.1858 -> 0.2912 | matches | OK |

| Location | Claim | Recomputed | Verdict |
|---|---|---|---|
| Abstract, 5.4, Table 6 caption | rho_LOO 6/6, DelAUC gap 3/6 | 6/6; wins LaTa MaRC, mT5 IG, mT5 MaRC = 3/6 | OK |
| 5.4 | one DelAUC win within 1.2 SE | memo A7: LaTa MaRC +0.047 +- 0.039 | OK (not recomputable without the per-pair JSONs) |
| 5.4, Table 7 caption | sufficiency-side metrics fail the shuffle control in two baseline cells | `rand_ins_auc_gap_gap_mean`: LaTa MaRC baseline -0.189, mT5 IG baseline -0.030; all other 10 cells positive | OK |
| Table 6 caption | random-order reference 0.671 to 0.928 | `del_auc_random_mean`: mT5 ABTT 0.6715, mT5 baseline 0.9283 | OK |
| Table 6/7 captions | 191 to 195 ABTT pairs vs 200 | `_n` columns: LaTa 195, PhilTa 191, mT5 191 | OK |
| Table 7 | Suff@25, Comp@25, MinFrac@0.80, InsAUC gap columns | LaTa IG: 0.9645/0.7289; 0.6580/0.1944; 0.0415/0.3299; 0.1530/0.0324, all match | OK |
| 5.4, Table 7 caption | tau_b agrees with rho in 6/6 | all six tau cells rise | OK |

### A.9 Items not recomputable here

- McNemar p-values and the 1.2 SE statement (need the per-pair JSONs and Python):
  `python scripts/resubmit/lexical_vs_embedding.py` and the A7 block of
  `attribution_metrics_decision.md` are the sources; both were regenerated after #139.
- Anything that needs a PDF (section C).

---

## B. Claims audit (Methodologist)

### B.1 "Anisotropy is universal" vs Machina and Mercer

Abstract and Intro open with "Anisotropy is universal:" as a bare statement. Related Work
then cites machina2024anisotropy for "Anisotropy is neither universal". Both are correct in
context (the colon clause scopes the first to the six models), but a reviewer skimming the
abstract will read the two as a contradiction. Discussion already says "universal on this
corpus". Recommend the same scoping in the abstract and intro: "Anisotropy is universal
across the six models:" and "Anisotropy is universal on this corpus:".

### B.2 Collapse is T5-only; "first report" hedge

The evidence supports the split: the three T5 encoders bottom at 0.496 / 0.538 / 0.654,
the three non-T5 models never drop below 0.806 and end at 0.956 to 0.958. Contribution (2)
says "the T5-specific mid-depth retrieval collapse"; Limitations correctly narrows to "the
three T5 encoders in our set". "T5-specific" in the contribution list generalises past
three checkpoints of one family. Recommend: "the mid-depth retrieval collapse that only the
three T5 encoders show".

Discussion: "so we read this as a first report for T5 encoders, not a property of the
architecture." The literature memo (t5_anisotropy_lit.md, section 5) ends with "to our
knowledge, not previously reported". The paper drops the hedge. Recommend: "so, to our
knowledge, this is the first report of the collapse for T5 encoders, and we do not read
it as a property of the architecture." Also Related Work says T5 "enters that work with
encoder and decoder states pooled"; Godey et al. concatenate encoder and decoder results.
"concatenated" is the accurate word.

### B.3 mT5-base and "lose almost all pairwise separation"

Intro: "LaTa, PhilTa, and mT5-base lose almost all pairwise separation in their middle
layers." Fig 2 caption: "The three T5 encoders in the top row lose almost all pairwise
separation in their middle layers, with LaTa reaching chance". For mT5-base the minimum is
AUROC 0.654 against a best baseline of 0.838: it loses (0.838-0.654)/(0.838-0.5) = 54
percent of its above-chance separation, not "almost all", and its cosine gap rises over
the same layers. The abstract's own numbers (0.496, 0.538, 0.654) contradict "almost all"
for the third model. Recommend in both places: "LaTa and PhilTa fall to chance in their
middle layers and mT5-base loses more than half of its above-chance separation".
Section 5.2's wording ("bottoms out at ... 0.654 for mT5-base") is fine.

### B.4 "Improve close to monotonically with depth"

Supported for LaBSE (dips of at most 0.01 between adjacent layers) and Qwen (L27 0.967
then L28 0.958). KaLM peaks at L23 0.972 and drops to 0.956 at L24; the prose "rises with
depth to 0.956 for KaLM-mini" quotes the last layer, while the headline table selects L23
at 0.972. Not a mismatch, but say "rises with depth to a final-layer AUROC of 0.956 for
LaBSE, 0.958 for Qwen3-0.6B, and 0.956 for KaLM-mini" so the reader does not look for
0.956 in Table 2 and find 0.972.

### B.5 Train-only selection: three violations visible in the paper

The paper states (2.3) "every choice above is made on train and evaluated once on test",
and the parent's hard rule is all selection train-only. Three artefacts break it.

1. Per-layer appendix tables. The captions of `tab:taskA_main`, `tab:taskB_routing_main`,
   `tab:taskB_ranking_main`, `tab:taskA_appendix`, `tab:taskB_routing_appendix`,
   `tab:taskB_ranking_appendix`, `tab:taskA_appendix_sif`, `tab:taskB_routing_appendix_sif`
   say "Rows in bold mark the best layer per model, selected by ABTT overall assignment
   accuracy" (or AUROC / Acc@1). `build_per_layer_tables.py` line 283 takes `idxmax` on
   the test column. Result: `tab:taskB_routing_main` bolds LaTa L1 (test 0.909) while
   Table 3 reports L8 (88.5, train-selected). A reviewer who cross-reads the two sees a
   3-point gap and a test-set argmax. Only the mseed table (issue #175) uses the train rule.
   Fix: extend `train_selected_layers` from `taskb_mseed_selection.py` to the other seven
   generated tables, bold the train-selected row (which is then the headline subscript), and
   change the caption sentence to "Rows in bold mark the layer chosen by the train-only rule
   of Section 3.2, the same layer as the subscript in Table 2/3." If the test maximum must
   stay visible, print it in italics with a caption sentence that says it is test-selected
   and not used anywhere in the paper.

2. `tab:lasttok_comparison` (App. D). `build_lasttok_comparison_table.py` defaults
   `--select_on overall_assignment_acc`, the test column, and the values confirm it (LaTa
   Mean L1 0.909 is the test argmax; the train rule gives L8 0.885). The caption also says
   "Best layer per (model, pooling) selected by Task A assignment accuracy" (assignment
   accuracy is a Task B metric) and "on the balanced split" (obsolete term), and the table
   includes Qwen3-8B, which is not one of the paper's six models. Fix: regenerate with
   `--select_on train_dir_acc_at_1` (the column exists in the lasttok results CSV since it
   comes from `run_resubmit_evaluate.py`), drop the Qwen3-8B rows or add one sentence saying
   why a seventh model appears here only, and rewrite the caption: "Best layer per (model,
   pooling) chosen by training-set directory accuracy at rank 1; test assignment accuracy,
   directory accuracy at rank 1 and cosine gap reported at that layer."

3. t-SNE / UMAP figures (App. H). `benchmark_v1_cluster_viz.sbatch` passes
   `taskA_per_model_summary.csv` as `--best_layer_csv`; that file holds LaTa L1
   (`sif_abtt_fixed`, test assignment 0.9196), PhilTa L1, mT5 L1, LaBSE L11, Qwen L5,
   KaLM L1, i.e. the test argmax across methods, and the panels are then drawn at those
   layers for `baseline` vs `abtt_fixed`. The captions say "at each model's selected layer"
   without a rule, and the figure title reads "at each model's best layer". Fix: rerun with
   the train-selected ABTT layers (LaTa 8, PhilTa 1, mT5 1, LaBSE 11, Qwen 5, KaLM 3), and
   state the rule in the caption. This also fixes the silhouette sentence in 5.2, whose
   numbers would change.

### B.6 Five-seed spread sentence

"The baseline spread of 36.6 points, from 50.7 for mT5-base to 87.3 for KaLM-mini" is read
at the Base subscripts of Table 3 (L1, L1, L12, L11, L28, L23). The bold rows of the mseed
appendix table are the SIF+ABTT layers, where the baseline reads 70.7 / 70.5 / 47.6 / 82.0 /
48.7 / 59.6. The sentence is true, but the table the reader is sent to does not show 50.7
or 87.3 in bold. Recommend: "at each method's own train-selected layer (the Base and
SIF+ABTT subscripts of Table 3)".

### B.7 Fine-tuning ceiling

The text and both reference-row captions say "fine-tuned contrastively on the 565 positive
train pairs". The run trains on 499 pairs after a 28-directory dev carve (finetune_ceiling.md,
"The dev carve"); 565 is the number available. The paper gives no fine-tuning setup at all
(objective, batch size, learning rate, epochs, early stopping, dev metric), so the reference
rows are not reproducible from the paper. Recommend: in the captions, "fine-tuned
contrastively on the positive train pairs (499 after a directory-level dev carve)", and add
a short appendix paragraph (or `\input{tables/finetune_ceiling}`, which already exists,
is generated, and carries the epoch-7 / D=10-boundary caveats) with: symmetric InfoNCE,
in-batch negatives with no two pairs from one directory, temperature 0.05, AdamW 2e-5,
batch 16 pairs, up to 8 epochs, patience 3, selected epoch 7 (terminal), seed 42. Also
say the ceiling is "at this training budget", which the finetune memo insists on and the
paper omits. The finding itself (parameter-free repair matches supervision on routing,
ABTT costs 1.4 AUROC points on the fine-tuned row) is stated at the right strength.

### B.8 Lexical disclosure

Abstract, intro, and the 5.3 paragraph all say TF-IDF matches the best embedding
configuration and that the paper does not claim embeddings beat surface overlap. This is
the memo's verdict and the numbers hold. Two gaps. (a) The lexical baselines are never
described in the method section: no BM25 parameters, no character n-gram range beyond the
row label, no min-max rescaling, no Levenshtein normalisation, and `tables/lexical_baselines.tex`
(which has all of this in its caption) is not `\input`. Add three sentences to Section 4
or an appendix, or input that table. (b) "leads only marginally on low-overlap pairs
(p = 0.023 and 0.033, uncorrected)" is honest, but the twelve-comparison family should be
named: "uncorrected over twelve comparisons per stratum".

### B.9 Attribution claims

The 5.4 sentence is B5 of the decision memo almost verbatim, and Tables 6 and 7 match
`summary_v2.csv`. The three caveats the memo requires are all present: unequal pair counts,
descriptive cross-variant comparison, and representation-level masking scoped to the
layer-L read-out. The 1.2 SE tie is stated. The InsAUC gap 5/6 is kept out of the main
table with the shuffle-control reason. Nothing to change on strength. One consistency
point: contribution (4) says "on present evidence ABTT improves leave-one-out rank
faithfulness while ERASER-style rationale metrics stay mixed", and the abstract says the
same; fine.

### B.10 Density figure selection

Caption says PhilTa L8 is "the mid-depth layer with the highest training-set AUROC". The
generator restricts to 30 to 70 percent depth (L4 to L8 for a 12-layer encoder), and L8 is
the train argmax there. L3 has train AUROC 0.608 and would win under any looser reading.
State the window in the caption. The figure itself labels the panels "Collapsed layer (L8)"
with legend "Different / Same"; the caption speaks of "collapsed retrieval layer" and
"Equivalent / non-equivalent pairs". Say "(Same and Different in the legend)" once so the
caption is self-contained.

### B.11 Generated captions vs prose

- `tab:taskB_ranking_appendix_mseed`: "the pure abtt_optimal variant reported in the main
  paper". The main paper now reports four settings; ABTT-only is one of them. Reword to
  "the ABTT-only variant of Tables 2 and 3".
- `fig:tsne`, `fig:tsne_appendix`: "Colors mark the six most-populated directories". The
  rendered legend has twelve directories (CNEO.315.13 ... CANC.314.22). Fix the number or
  the figure.
- `fig:tsne`: "projections of the 1,705 manuscript witnesses" while the silhouette is on
  the 1,160 winnable files; say "all 1,705 witnesses, with the 1,160 winnable ones scored".
- No other generated caption contradicts the prose; the headline captions repeat the
  prose numbers exactly.

### B.12 Overall

Soundness 4, Significance 3, Clarity 4, Novelty 3. The headline numbers are right, the
selection protocol is right where the paper reports numbers, and the honest framing of the
lexical baseline and the fine-tuning ceiling is the strongest part of the draft. The three
test-selected artefacts in B.5 are the only soundness problem, and they are appendix
material that can be regenerated without new GPU time. Recommendation for the co-author
sync: proceed after B.5 and B.3 are fixed.

---

## C. Presentation

### C.1 Not run (no shell)

- Fresh-aux build, undefined references and citations, overfull boxes, page of the last
  numbered section, page of Limitations. Command:
  `cd overleaf_drafts && rm -f *.aux *.bbl *.blg *.log && pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex && pdflatex acl_latex && grep -E "undefined|Overfull|LaTeX Warning" acl_latex.log`
  then `python -c "import fitz; d=fitz.open('acl_latex.pdf'); print([i+1 for i,p in enumerate(d) if 'Limitations' in p.get_text()])"`.
- Page budget risk, from reading: the main text is dense, with a `figure*` (Fig 1), two
  `table*` (Tables 2, 3), a single-column table (Table 1), a single-column table (Table 6)
  and one equation before the `\clearpage`. If the build shows the Discussion ending on
  page 9, the shortest cuts that lose nothing audited above are the second half of the
  Task B definition paragraph in 2.1 (duplicated in 4.5) and the "PCA whitening (excluded)"
  paragraph in 3.3 (duplicated in App. F).

### C.2 Checked by reading

- Labels: every `\ref` in `acl_latex.tex` has a matching `\label` in the input files
  (tab:taskA_main, taskB_routing_main, taskB_ranking_main, taskA_appendix,
  taskB_routing_appendix, taskB_ranking_appendix, taskA_appendix_sif,
  taskB_routing_appendix_sif, taskb, taskB_ranking_appendix_mseed, lasttok_comparison,
  attribution_metrics_main, attribution_metrics_secondary, attribution_sweep_main_methods,
  attribution_sweep_supplemental_methods, all fig: and app: labels, eq:retrieval_mark).
  `sec:related` is defined and never referenced (harmless).
- Citations: all 44 keys used in the two .tex files exist in `custom.bib`. `acl.sty` and
  `acl_natbib.bst` are present. Unused entries (li2020isotropy, rajaee2021cluster,
  huang2021whiteningbert, the template ones) are harmless.
- Every float is referenced in the text: Figs 1 to 11, Tables 1 to 7 plus the sixteen
  appendix tables. Orphan table files that are not input and not referenced:
  `finetune_ceiling.tex`, `lexical_baselines.tex`, `attribution_metrics_appendix.tex`,
  `attribution_metrics_200pos_sweep_appendix.tex`, `retrieval_summary_main.tex`,
  `taskA_per_model.tex`. They do not break the build; the first two should be input (B.7,
  B.8), the rest can be deleted from the Overleaf mirror to avoid co-author confusion.
- All figure files referenced exist as PDF in `overleaf_drafts/figures/`. Memory says figure
  PDFs are gitignored; confirm `scripts/paper/sync_paper_repo.sh push` carries them, or the
  Overleaf build will fail on eleven missing graphics.
- Figure legends match captions: Fig 1 and Fig 2 legends read "Baseline / SIF-only /
  ABTT-only", matching the caption colour key; no "Dip Layer" text remains anywhere outside
  `scripts/_archive/`. Fig 4 legend "baseline / ABTT" matches.
- Terminology: "labelled" is used consistently (no "labeled"); "directory" is the unit
  everywhere; "family" and "Latin department" do not occur; "witness" and "siglum" are
  introduced in 2.2; "most anisotropic layer" is used for the geometry rule and "collapsed
  retrieval layer" for the train-AUROC rule, and the density caption explains that they are
  different PhilTa layers (6 vs 8) and that the test minimum is a third (10). One residual:
  App. B is titled "Score Distributions at the Collapsed Retrieval Layer" while the figure
  panels say "Collapsed layer"; harmonise to "collapsed retrieval layer" in the figure title
  or drop "retrieval" in the section title.
- Em-dashes: none in either .tex file (the grep hit in `acl_lualatex.tex` is a template
  comment; `acl_lualatex.tex` should be removed from the mirror so nobody compiles it).
- Firey annotations (plan_20260906.md table): every item is resolved. "families" gone;
  "consolidate" -> "identifying duplicates"; "link related" -> "analyzing witnesses";
  "strict" gone (the only "strict" hits are inside "restricted"); "meaning-equivalent
  families" -> "840 labelled canon-law fragments with 1,705 manuscript witnesses";
  "legal reasoning" -> "legislation and judicial opinions"; "Manuscript copies" ->
  "manuscripts"; "editorial history" -> "transmission history"; "everyday" gone;
  "arbitrary" -> "randomly selected"; "one legal text" -> "one labelled fragment".
- Provenance: founding 2009, manual transcription from images or eighteenth-century
  printings, pre-2019 microfilm to TEI-P5 by contributors, post-2019 Transcription Desk with
  proofreading scoped to that route, per-unit credit, plain text derived from the TEI
  export. All match CLAUDE.md and the story memo. One caveat CLAUDE.md records that the
  paper does not: HTR was attempted on one manuscript (BnF lat. 1454) and is still in
  proofreading. If any of its units are in the 1,705, "Transcription is manual" needs
  "with one manuscript transcribed by HTR and still in proofreading"; if none are, nothing
  to do. `python - <<'EOF'` over `data/canon_labelled/*/` filenames for the BnF 1454 siglum
  settles it.
- James Wong 2026-09-12 comments (#175 to #178). GitHub was not reachable from this
  session. From the commit log at 1e001b9: #175 (one selection rule for the five-seed
  tables) landed as F10 / PR #181 and is documented in benchmark_v1.md; #176 (lexical
  disclosure up front) landed as F11 / PR #182 and is in the abstract, intro and 5.3;
  the two remaining comments are inferred to be F12 (provenance claims cite project
  documentation or the director, PR #180) and F13 (state what the gap figure shows for each
  T5 encoder, PR #179), both merged and both visible in the draft. The four 2026-09-03
  Overleaf comments are also resolved: figure/caption term match, whitening explanation
  paragraph, "non-T5 models", "T5 encoders", no "Retrieval-MarK", "SIF-only"/"ABTT-only"
  used throughout.
- Anonymity: author block is the template placeholder; acknowledgments are commented out;
  the only named person is in the `eichbauer2014ccl` bibliography title and the
  `firey2009ccl` author field, which are citations of the project. The footnote
  "project director, personal communication, 2026" is fine under review mode.
- Ethics and data statement: there is no ethics section and no data-availability sentence.
  The ARR checklist will ask about licence and release; CLAUDE.md records that CCL licensing
  is unconfirmed. Do not add a release promise until that is settled; add a one-sentence
  data statement after it is.
- Leftovers to remove before the sync: `% FINAL WORDING GATED ON #124` (line 85; #124 is
  resolved), `% SIDDIQUE TO REVIEW WORDING` (line 162), the 20-line "OPEN DECISIONS"
  comment block after Limitations (lines 271 to 290; move to an issue), and the stale
  `figures/taskb_selected_configs.csv` (787/918 split, pre-v1) which should not travel to
  Overleaf.
- Bibliography rendering: `ccl2026about` / `ccl2026desks` with author `{{CCL}}` render as
  CCL (2026a, b); `liu2024medieval` key says 2024 but the entry is 2025 (renders correctly,
  cosmetic); `godey2024anisotropy` drops "de la" from "de la Clergerie"; otherwise entries
  are complete and venue-consistent.
- Title still carries "Geometric Repair", which the story memo flagged as inviting a
  method-paper review; open decision for the co-authors, not a defect.
