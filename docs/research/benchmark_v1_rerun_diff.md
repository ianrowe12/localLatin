# Benchmark v1 re-run: what changed in the paper

Issue #113, epic #107. Every number the ARR October 2026 submission reports is
re-derived here from the corrected labels frozen as benchmark v1
(`benchmark_v1.md`). Prof. Firey's two corrections move `BN2123.89r.5` from
`Can.apost.48` to `Can.apost.49` and `BN2123.89r.6` from `Can.apost.49` to
`Can.apost.50`. The texts are byte-identical, so nothing was re-embedded and
every job in this re-run was CPU only.

## 1. The alignment hazard, and how it is handled

`file_id` is the row index into every cached `.npy`. It is assigned by sorting on
`(folder_id, filename)`, so moving a file between directories permutes the split
rows while the cached matrices keep their extraction order. Seventeen rows move,
in the window `file_id` 1554-1570. Every count stays the same, and the consumers
guarded alignment with a row-count check, which cannot see a permutation. A naive
re-run would have scored seventeen vectors against seventeen wrong labels and
said nothing.

### The choice: align by filename, do not rewrite the caches

`benchmark_v1.md` offered two routes: permute every cached array with the
documented old-to-new mapping, or teach the consumers to align by filename. This
re-run takes the second, implemented as `src/embedding_alignment.py` and wired
into every script that indexes a labelled embedding matrix by split row.

Why not permute the arrays:

* It rewrites 734 matrices to express a change that is purely about labels. The
  whole premise of the correction is that the vectors did not change; rewriting
  them makes that claim unverifiable afterwards.
* It is not idempotent. Running the permutation twice silently double-permutes,
  and nothing on disk records whether it has been applied. A half-finished pass
  leaves a cache that is partly one order and partly the other, with no way to
  tell which files are which.
* It fixes exactly this correction. The next label correction needs the same
  dangerous pass again.

Aligning by filename costs microseconds, leaves the caches untouched, is
idempotent, and generalises: any future relabelling is handled with no
intervention at all. Filenames are unique across the corpus, so the mapping is a
bijection, and the code checks the bijection rather than assuming it.

### Where the row order comes from

The extractor already writes a `meta.csv` into each run directory recording the
corpus order it walked. All 19 caches under
`runs/active/resubmit_bases/phase9_bases/` carry one, and all 19 are identical.
That file is the manifest. A `row_order.csv` at a bases root is the fallback for
caches written before per-run manifests existed; this re-run wrote one there as a
frozen copy of the extraction-time order.

A cache with no manifest still runs, positionally, but says so loudly on stderr,
so the unverifiable case is visible rather than silent.

### Evidence

```bash
python scripts/resubmit/verify_embedding_alignment.py
```

checks three things over all 19 caches and exits non-zero on any failure:

1. Every cache resolves to a row-order manifest.
2. The permutation each manifest implies equals the permutation implied by
   diffing the pre-correction split against the corrected one, row for row: the
   same 17 moves, in the same order, as the table in `benchmark_v1.md`.
3. The vector a named file receives after alignment is bit-identical to the one
   it received before the relabelling.

The named-file spot check, over all 19 caches and 38 checks:

```
ok   bowphs_LaTa/hidden_mean_tokempty/hidden_layer10_embeddings.npy
     BN2123.89r.6.txt: split row 1560 -> 1570,
     sha256 67a32e48a9b5314b (before 67a32e48a9b5314b)
ok   bowphs_LaTa/hidden_sif_tokempty/hidden_layer10_embeddings_sif.npy
     BN2123.89r.6.txt: split row 1560 -> 1570,
     sha256 aa7332085923b734 (before aa7332085923b734)
...
38 spot checks over 19 caches
PASS: every cache is manifest-verified and every spot check is byte-identical
```

Every job log also carries a per-cache line, for example:

```
[alignment] .../bowphs_LaTa/hidden_mean_tokempty:
verified-permuted (1705 rows, 17 moved, manifest .../meta.csv)
```

## 2. What the corrected split changes

Regenerating the split with the carry-over rule reproduces
`benchmark_v1.md` exactly: 17 rows change `file_id`, 12 rows change something
else (two files change `folder_id`, ten change `folder_size`), and no file
changes `split`, `taskb_role`, `is_test_query`, `has_test_partner` or
`has_reference_dir`. Train and test counts hold at 847 and 858. The only
substantive change is one extra positive test pair, 595 to 596, from
`BN2123.89r.6` joining the test files of `Can.apost.50`.

## 3. Numbers in the paper

Method: `scripts/paper/numeric_token_diff.py` pairs the numeric tokens of each
file before and after and reports only the ones whose value moved.

### Tables

| Table | numeric cells | changed | max abs delta |
|---|---|---|---|
| `taskA_headline.tex` | 100 | 39 | 0.074 (plus four layer subscripts) |
| `taskB_headline.tex` | 100 | **0** | - |
| `attribution_metrics_main.tex` | 71 | **0** | - |
| `taskA_main.tex` | 180 | 82 | 0.005 |
| `taskB_routing_main.tex` | 253 | 3 | 0.040 |
| `taskB_ranking_main.tex` | 256 | 3 | 0.040 |
| `appendix_lasttok_comparison.tex` | 62 | **0** | - |
| `taskA_appendix.tex` | 320 | 123 | 0.014 |
| `taskB_routing_appendix.tex` | 449 | 3 | 0.008 |
| `taskB_ranking_appendix.tex` | 454 | 3 | 0.008 |
| `taskA_appendix_sif.tex` | 503 | 259 | 0.006 |
| `taskB_routing_appendix_sif.tex` | 502 | 6 | 0.019 |
| `taskB_ranking_appendix_mseed.tex` | 1308 | 811 | 0.061 |
| `attribution_metrics_sweep_*.tex` | 598 | **0** | - |
| `taskB_topk.tex` (was inline) | 65 | 47 | 0.6 |
| `lexical_baselines.tex` | 35 | 8 | 0.003 |
| `finetune_ceiling.tex` | 44 | 4 | 1.0 (a Top-1 percentage, 83.6 to 82.6) |

Reading the table:

* **Task B routing is unmoved.** `taskB_headline.tex` is bit-identical cell for
  cell: the same layers are selected, and every assignment and directory accuracy
  rounds to the same figure. The routing story in the paper does not depend on
  the correction at all.
* **Task A moves in the third decimal, plus four layer flips.** AUROC changes in
  all 700 result rows, because AUROC is computed over the label-partitioned test
  pair set and that partition gained a pair. The mean absolute change is 0.0010
  and the largest is 0.0063. Four of the 48 (model, setting) headline cells now
  select a different layer, because the training-set argmax flipped between
  near-tied layers: PhilTa ABTT 1 to 9, mT5-base ABTT 1 to 2, Qwen3-0.6B baseline
  28 to 26, KaLM-mini SIF+ABTT 4 to 2. Those flips, not the label change itself,
  are what move the headline band from 0.971-0.984 to 0.971-0.987.

  They are also the whole of the 0.074 max delta in that table, which is not a
  number moving but a different cell being shown. Qwen3-0.6B's baseline cosine
  gap reads 0.024 at layer 26 where it read 0.098 at layer 28, and the layer
  behind it is a coin toss: train AUROC is 0.957202 at L26 against 0.956524 at
  L28, a margin of 0.0007, and before the correction it ran the other way by
  0.0002. The largest delta from a cell that kept its layer is 0.006. The
  paper's new sentence "Qwen3-0.6B ranks nearly as well at 0.966 with the
  smallest gap of all, 0.024" and the lower end of the reported baseline gap
  range both rest on that tie-break. The selection rule is train-only and was
  not changed, so this is protocol-consistent, but it is a tie and is worth
  knowing about: on the pre-correction split the same rule reported 0.098.
* **The five-seed appendix moves most.** `run_taskb_mseed.py` redraws the
  query/reference partition per seed by walking directories in `folder_id` order
  and drawing from one RNG stream. `Can.apost.49` loses a test file and
  `Can.apost.50` gains one, so the number of draws changes there and every
  directory sorting after it gets different draws. This is a property of the
  reseeding protocol, not of the correction, and it is why 811 of 1308 cells move
  while the headline single-split table does not move at all. The five-seed
  summary numbers still round the same: baseline spread 36.6 points, SIF+ABTT
  spread 1.0 point, standard deviations at most 1.1.

  `taskB_topk.tex` is the five-seed table in the paper, so it moves with them:
  47 of its 65 numeric tokens change, by up to 0.6 points (KaLM-mini Top-3, 96.4
  to 97.0). It is a new file in this pass because the table moved out of
  `acl_latex.tex` into a generator, but the numbers are *not* the ones the
  inline table carried, and the section 4.3 prose was edited to match: LaTa
  Top-1 90.1 to 90.0, LaBSE 90.4 to 90.6, KaLM-mini 89.4 to 89.5, so the printed
  SIF+ABTT band goes 89.4-90.4 to 89.5-90.6. The prose reports the spread from
  the unrounded values, 90.5503 minus 89.5250, which is 1.0253 and rounds to
  1.0; subtracting the rounded endpoints instead gives 1.1. The endpoints in the
  prose are therefore printed to two decimals so the arithmetic a reader does in
  their head agrees with the number next to it.
* **Attribution is untouched**, deliberately. See section 6.
* **Whitening is the noisiest row set** and is excluded from the main tables
  anyway. Its PCA is fit on the train block, whose membership is unchanged, but
  sklearn's randomized SVD solver depends on row order, and whitening amplifies
  the near-degenerate directions by `1/sqrt(eigenvalue)`. So its numbers move by
  up to 0.05 in cosine terms. Every claim the paper makes about whitening
  survives: existing accuracy 1.000 in all 100 cells, new accuracy never above
  one file in 323, assignment accuracy pinned at the 62.4 percent class prior,
  beats baseline in 60 of 100 cells, directory accuracy at rank 1 never above
  0.415. Only the different-directory cosine bound moved, 0.014 to 0.015.

### Prose

39 numeric tokens in `acl_latex.tex` were updated to match. The full token-level
report is reproducible with:

```bash
python scripts/paper/numeric_token_diff.py overleaf_drafts/acl_latex.tex
```

| Section | Before | After |
|---|---|---|
| Abstract | AUROC 0.497, 0.539, 0.654 | 0.496, 0.538, 0.654 |
| Abstract | 0.971 to 0.984 band | 0.971 to 0.987 band |
| Abstract | gap 0.525 to 0.610 | 0.525 to 0.611 |
| Split (3.2) | 595 positive test pairs | 596 |
| Whitening (3.x) | different-directory cosine at or below 0.014 | 0.015 |
| Results 4.1 | baseline AUROC 0.839 to 0.970 | 0.838 to 0.972 |
| Results 4.1 | ABTT band 0.971 to 0.984 | 0.971 to 0.987 |
| Results 4.1 | baseline gap range 0.056 to 0.238 | 0.024 to 0.237 |
| Results 4.1 | ABTT gap range 0.525 to 0.610 | 0.525 to 0.611 |
| Results 4.1 | gains 0.140 / 0.032 / 0.011 | 0.137 / 0.034 / 0.009 |
| Results 4.1 | KaLM-mini best baseline 0.970, smallest gap 0.056 | KaLM-mini 0.972 with a 0.056 gap; Qwen3-0.6B 0.966 with the smallest gap, 0.024 |
| Results 4.2 | dip AUROC 0.497 / 0.539 / 0.654 | 0.496 / 0.538 / 0.654 |
| Results 4.2 | ABTT at those layers 0.963 / 0.982 / 0.977 | 0.964 / 0.982 / 0.978 |
| Results 4.2 | non-T5 depth trends 0.807-0.957, 0.859-0.958, 0.888-0.954 | 0.806-0.956, 0.858-0.958, 0.887-0.956 |
| Results 4.2 | layer minima 0.807, 0.859, 0.862 | 0.806, 0.858, 0.861 |
| Results 4.2 | silhouette rise 0.09 to 0.75 | 0.07 to 0.72 |
| Results 4.3 | five-seed baseline low 50.6 | 50.7 |
| Results 4.3 | SIF+ABTT band 89.4 to 90.4 | 89.53 to 90.55 (2 dp, so the stated 1.0-point spread and the printed endpoints agree) |
| Discussion | 0.971 to 0.984 AUROC band | 0.971 to 0.987 |

Claims that were checked and did **not** need changing: the 0.23 to 0.95 mean
cosine range and every geometry figure in Section 4.2 and Table 6 (geometry is
computed on train and test blocks whose membership did not change, and it is
invariant to row order); the 0.29 baseline cosine margin ceiling; every SIF
delta in Section 4.1; the whole of Section 4.3's single-split Task B paragraph;
the 36.6 to 1.0 point five-seed compression; "by Top-2 every model exceeds 95"
(the minimum is now 95.3); the 3.5 to 9.1 point non-T5 gain; the 3.3-point
routing band; the split statistics table; and the operational attribution layers
LaTa 7, PhilTa 1, mT5-base 1, which the train-only rule still selects.

## 4. Regenerated artifacts

Committed generators now cover everything the paper shows. New in this pass:

| Artifact | Producer |
|---|---|
| `tables/taskA_headline.tex` | `scripts/resubmit/build_headline_tables.py` |
| `tables/taskB_headline.tex` | `scripts/resubmit/build_headline_tables.py` |
| `figures/fig_release_aucroc_6model.pdf` | `plot_metric_grid_6model` in `scripts/resubmit/visualize_resubmit.py` |
| `figures/fig_release_gap_6model.pdf` | same |
| `tables/taskB_topk.tex` (was inline) | `scripts/resubmit/visualize_taskb_mseed.py` |
| `tables/finetune_ceiling.tex` | `slurm/resubmit/benchmark_v1_finetune_eval.sbatch` (rescore only) |
| `figures/fig_{tsne,umap}_{main,appendix}.pdf` | `slurm/resubmit/benchmark_v1_cluster_viz.sbatch` |

The two headline table generators were validated by regenerating them from the
pre-correction results CSV and diffing against the committed files: both
reproduce byte for byte. All three generated tables are committed exactly as the
generators emit them, header line included, so re-running
`slurm/resubmit/benchmark_v1_taska.sbatch` leaves the tree clean rather than
dirtying line 1 of three files. The figure `save` helper now passes
`CreationDate: None` to the PDF writer for the same reason: matplotlib otherwise
stamps the wall clock into every PDF, so a re-run rewrote five bytes in each and
made a real figure change indistinguishable from a re-run. PNGs carried no such
stamp and were already byte-reproducible; the one-time PDF diff in this pass is
that stamp being dropped.

Still hand-maintained in `acl_latex.tex`, and unchanged by this correction: the
split statistics table (`tab:split`) and the layer diagnostics table
(`tab:layer_diagnostics_main`).

### The fine-tuning ceiling

`tables/finetune_ceiling.tex` (#123) landed on main while this re-run was in
flight, and it is split-dependent. Its scoring half is CPU only and the
fine-tuned checkpoint and embeddings are unchanged, so it is rescored here on the
corrected split rather than left stale;
`slurm/resubmit/benchmark_v1_finetune_eval.sbatch` does that in about four
minutes. Its embedding cache had no per-run manifest, so this pass wrote a
`row_order.csv` at `runs/active/resubmit_finetune_bases/phase9_bases/` from the
frozen extraction-time order and wired the resolver into
`scripts/resubmit/finetune_lata_ceiling.py`. Without that, a re-run there would
have hit exactly the misalignment this issue exists to prevent.

The rescore is an interim, not a promotion. The checkpoint itself was trained on
the pre-correction labels, and the correction changes the pair set it learned
from: the train positive pairs stay at 565, but two it trained on as positives
are now wrong (`BN2123.89r.5` with `KoeD213.10r.1` and with `Vat5845.10r.7`) and
two it never saw are now positives (`BN2123.89r.5` with `C1525.7v.6` and with
`Hat42.149r.3`). Four pairs in 565, about 0.35 percent, so rescoring the frozen
checkpoint is a reasonable stand-in for the table, but the checkpoint is not a
benchmark v1 artifact and #138 stays open until it is retrained on v1 labels.

## 5. The reviewer pilot's predictions, and why their layer is now sticky

The four `unlabelled_predictions_{variant}.csv` files and the six `qq_sim_*.npz`
matrices under `runs/active/resubmit/unlabelled/` are not a paper artifact. They
are what the reviewer pilot serves: a scholar opens a query and sees the top-1
directory one of these files predicted. Re-running them on the corrected split is
not the neutral act it is for a table.

### What the plain re-run did

Regenerating all four variants with the unmodified selector and diffing
`rank1_dir` against the currently deployed CSVs:

| variant | rank1_dir changed | layer flips |
|---|---|---|
| `raw` | 8 / 13,428 | none |
| `sif` | 6 / 13,428 | none |
| `abtt` | 11 / 13,428 | none |
| `sif_abtt` | **1,286 / 13,428** | **Qwen3-0.6B 7 to 1** |

Three variants move by the handful of queries the two relabelled files actually
touch. `sif_abtt`, which is the webapp default, moves 1,286, and 1,276 of those
are Qwen3-0.6B alone: 57 percent of that model's 2,238 reviewer-facing answers.

The cause is not the label change. `select_best` picks the layer by
`overall_assignment_acc`, layer 1 rose from below 0.911422 to 0.913753 while
layers 7 and 8 stayed at 0.911422, and a 0.23 point move flipped the deployed
layer from 7 to 1. Different layer, different embedding space, different
shortlist for more than half the corpus, from a correction that touches two
files. `qq_sim_Qwen_Qwen3-Embedding-0.6B.npz` had been rebuilt at layer 1 to
match, so the whole Qwen pipeline had moved together and silently.

### The rule

`run_resubmit_unlabelled_retrieval.py` now keeps the deployed layer unless a new
best layer beats it, on the same selection metric, by more than a tolerance:

* `scripts/resubmit/deployed_unlabelled_layers.json` records the layer each
  (variant, model) currently serves. It was written by
  `--record_deployed_layers`, which reads the `layer` column of the live CSVs, so
  the record is what is deployed rather than what anyone remembers deciding.
* The tolerance is **0.005**, half a point of assignment accuracy. It sits an
  order of magnitude above the 0.0023 that this correction moved Qwen3-0.6B by,
  and an order of magnitude below the 0.02 to 0.09 that separates a model's good
  layers from its bad ones. So it absorbs relabelling noise and does not defend a
  layer that is genuinely worse.
* Every decision is printed and written to
  `sticky_layer_decisions_{variant}.json` next to the predictions.
* `build_qq_matrices.py` calls the same selector rather than its own copy. The
  query-query matrices score reviewer-created directories, so building them at a
  different layer from the predictions would mix two embedding spaces in one
  shortlist.
* `--layer_overrides` still wins over the rule; an explicit pin is an explicit
  pin.

The rule is not a claim that layer 7 is better than layer 1. It is a claim that
layer 1 is not better *enough* to justify invalidating a live shortlist over it.

### The paper does not use it

`build_headline_tables.py`, `visualize_taskb_mseed.py` and the per-layer table
builders take the pure argmax, with no stickiness anywhere. Every layer subscript
in `overleaf_drafts/` is the unmodified train-set argmax, which is why
`taskA_headline.tex` still reports Qwen3-0.6B baseline at layer 26 and not at 28.
`--no_sticky_layers` reproduces that behaviour here.

### What shipped

Under the sticky rule the deployed layers are unchanged in all 24 (variant,
model) cells, and the predictions differ from the deployed ones only where the
two corrected directories reach:

| variant | rank1_dir changed vs deployed | layer flips |
|---|---|---|
| `raw` | 8 / 13,428 | none |
| `sif` | 6 / 13,428 | none |
| `abtt` | 11 / 13,428 | none |
| `sif_abtt` | **14** / 13,428 | none |

Per model, changed top-1 answers out of 2,238 queries each:

| model | raw | sif | abtt | sif_abtt |
|---|---|---|---|---|
| LaTa | 0 | 0 | 2 | 1 |
| PhilTa | 0 | 0 | 0 | 1 |
| mT5-base | 5 | 2 | 2 | 3 |
| LaBSE | 0 | 1 | 4 | 4 |
| Qwen3-0.6B | 2 | 2 | 1 | 4 |
| KaLM-mini | 1 | 1 | 2 | 1 |

`sif_abtt` goes from 1,286 changed answers to 14, and the remaining 14 are the
ones the two relabelled directories actually reach. The six `qq_sim_*.npz`
matrices were rebuilt at the same sticky layers, so the predictions and the
reviewer-directory scores stay in one embedding space.

Two things this pass does *not* settle, both pre-existing and neither in scope
for #113:

1. `select_best` chooses the layer by `overall_assignment_acc` on the **test**
   split. That is a test-set selection, unlike the train-only rule the paper
   uses, and it predates this branch. The sticky rule reduces its blast radius
   but does not fix it.
2. Reviewer feedback rows already recorded against a shortlist are keyed by
   query and directory, not by layer. Nothing in this pass invalidates them,
   because nothing they were recorded against has moved, but a future deliberate
   layer change would need that question answered before it ships.

## 6. Deliberately not re-run

**The 200-pair attribution metrics.** `scripts/ig/sample_positive_test_pairs.py`
draws its pairs from `runs/phase9/phase9_split.csv`, the legacy phase 9 split over
`data/canon/`, and the sbatch does not override that default. `data/canon/` was
not touched by the benchmark v1 corrections, which apply to `data/canon_labelled/`
only. So re-running the sampler reproduces the same 600 rows byte for byte, and
`attribution_metrics_main.tex` and both sweep tables come back numerically
identical, which the diff above confirms. The operational attribution layers are
also unchanged, so the locked choices in `scripts/ig/attribution_model_config.py`
remain the ones the train-only rule selects.

Two follow-ups this uncovered, neither in scope for #113 and both needing GPU
time and Ian's approval:

1. Five of the 600 sampled pairs use `BN2123.89r.5` as query or candidate and are
   recorded as positives against `Can.apost.48`. Under benchmark v1 that file
   belongs to `Can.apost.49` while both its partners stay in `Can.apost.48`, so
   those five rows are now negatives labelled positive: LaTa 2 of 200, PhilTa 1,
   mT5-base 2. `BN2123.89r.6` is not in the sample.
2. More broadly, the attribution sample is drawn from a different corpus and a
   different split than every retrieval number in the paper. Re-sampling against
   `phase_resubmit_split.csv` would fix both at once, but the shared RNG stream
   means all 600 pairs change, so every IG NPZ, MaRC mask, summary CSV, sweep
   table and the rho-LOO figure would have to be regenerated.

## 7. How to reproduce

```bash
# 1. Install the corrected split (backs itself up first).
mkdir -p runs/active/resubmit/data/benchmark_v1
cp runs/active/resubmit/data/phase_resubmit_split.csv \
   runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv
python scripts/resubmit/run_resubmit_data_prep.py \
    --carry_over_from runs/active/resubmit/data/benchmark_v1/phase_resubmit_split.pre_correction_backup.csv \
    --out_dir runs/active/resubmit/data

# 2. Gate on alignment.
python scripts/resubmit/verify_embedding_alignment.py

# 3. Re-run. All CPU; Task A took 35 min, the five seeds 23 min,
#    the lexical baselines 2 min, the downstream chain 6 min.
sbatch slurm/resubmit/benchmark_v1_taska.sbatch        # -> job A
sbatch slurm/resubmit/benchmark_v1_taskb_mseed.sbatch  # -> job B
sbatch slurm/resubmit/benchmark_v1_lexical.sbatch
sbatch --dependency=afterok:A slurm/resubmit/benchmark_v1_cluster_viz.sbatch
sbatch --dependency=afterok:A:B slurm/resubmit/benchmark_v1_downstream.sbatch

# 4. Diff the paper.
python scripts/paper/numeric_token_diff.py overleaf_drafts/acl_latex.tex
python scripts/paper/numeric_token_diff.py overleaf_drafts/tables/*.tex
```

## 8. Compile

Fresh-aux `latexmk -pdf` on a clean tree: **0 undefined references, 0 overfull
boxes, 0 LaTeX warnings**. Thirteen underfull hboxes remain, all justification
badness in paragraphs.

Every one of those figures is identical to what `origin/main` compiles to on its
own, checked by compiling the `origin/main` tree side by side: 36 pages, the same
13 underfull hboxes, the last body page before the bibliography is page 8, and
`tab:taskb` sits on page 9 in both. The body still runs past the ACL eight-page
limit; this pass does not worsen it by a single line.
