# Supervised fine-tuning reference ceiling (LaTa)

Issues #123 and #138, epic #109. **This is a reference ceiling, not a proposed
method.** The paper's pipeline is zero-shot: it never sees a labelled pair. This
experiment asks the complementary question a reviewer will ask anyway, namely
how much of the gap to a perfect system is left once a model is allowed to
train on the task's own supervision. The answer bounds what post-processing on
frozen representations can be expected to achieve, and it is reported as a
bound, never as a system we advocate.

Every number below comes from **benchmark v1** (`benchmark_v1.md`), re-trained
and re-scored end to end on the corrected labels under #138. What that changed
is in *Benchmark v1 re-run* below; the verdict did not move.

Everything here is fine-tuned and selected on the TRAIN split only. The test
split is untouched until the final evaluation, which uses the paper's own
evaluator with no changes.

## Setup

| Item | Value |
|---|---|
| Model | `bowphs/LaTa`, T5 encoder only, 12 blocks, mean pooling |
| Objective | Symmetric InfoNCE over positive pairs with in-batch negatives (the objective behind sentence-transformers' MultipleNegativesRankingLoss, implemented here because the environment has no `sentence_transformers`) |
| Temperature | 0.05 |
| Training pairs | all within-directory pairs from train directories with >= 2 files, minus the dev carve |
| Batching | 16 pairs (32 sequences) per step, with **no two pairs from the same directory in a batch**, so in-batch negatives are always true negatives |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, linear warmup 10% then linear decay, grad clip 1.0 |
| Precision | bf16 autocast, fp32 master weights |
| Epochs | up to 8, early stop after 3 epochs without dev improvement |
| Seed | 42 (Python, NumPy, Torch; also the dev carve and the batch order) |
| Tokenisation | max_length 512, `tokenizer_empty` token filter, identical to the paper's extraction |

### The dev carve

Model selection needs held-out data that the contrastive objective has never
seen, and it has to be held out by **directory**, not by file: two files from
the same directory are a positive pair, so splitting a directory across
train and dev would leak the exact supervision being measured.

- 190 train directories have >= 2 files, giving 565 positive pairs in total.
- 15% of those directories (28 directories, 71 files) are held out as DEV.
- The remaining 162 directories yield 499 training pairs, batched 32 to an epoch.
- Dev metric: directory accuracy@1 inside the dev pool, i.e. the fraction of
  dev files whose nearest other dev file is from the same directory. AUROC over
  dev pairs breaks ties. The checkpoint with the best dev accuracy@1 is the one
  extracted from; epoch 0 in the dev curve is the pre-trained encoder, so the
  curve shows what training actually bought.

The run early-stopped after epoch 7 and selected epoch 7. Directory accuracy@1
saturated at epoch 4 on a 71-file pool, where one file is worth 1.4 points, so
AUROC over the 2,485 dev pairs is what separates the last four epochs.

| Epoch | Train loss | Dev dir. acc.@1 | Dev AUROC |
|---|---|---|---|
| 0 (pre-trained) | | 0.930 | 0.947 |
| 1 | 0.801 | 0.930 | 0.963 |
| 2 | 0.581 | 0.944 | 0.974 |
| 3 | 0.404 | 0.972 | 0.979 |
| 4 | 0.352 | 0.986 | 0.981 |
| 5 | 0.305 | 0.986 | 0.983 |
| 6 | 0.243 | 0.986 | 0.983 |
| **7 (selected)** | **0.228** | **0.986** | **0.984** |

**Selected checkpoint: epoch 7, dev directory accuracy@1 0.986, dev AUROC 0.984.**

**Epoch 7 is the terminal epoch of the sweep.** The budget was 8 epochs with
patience 3, and patience fired after epoch 7, so the selected checkpoint is also
the last one trained. Dev accuracy@1 had been flat at 0.986 (70 of 71 files)
since epoch 4 and the AUROC tiebreak was still creeping up by a few times 1e-4
per epoch, which reads as saturation rather than truncation. Still, nothing here
rules out that a longer or larger training run would go further, so every number
below is a ceiling **at this training budget**, not an asymptote.

## Evaluation

Fine-tuned mean-pooled embeddings are extracted for all 1,705 labelled files at
every encoder layer (1-12) and written in the canonical
`phase9_bases/<slug>/hidden_mean_tokempty/` layout, so the paper's evaluators
read them unchanged:

- **Task A** (`run_resubmit_evaluate.evaluate_single`): test AUROC and cosine
  gap over the test n x n cosine matrix.
- **Task B** (same function): assignment accuracy and directory accuracy@1,
  with tau learned on train by best F1.
- **Task B, 5 seeds** (`run_taskb_mseed.evaluate_model_for_seed`, seeds 42-46):
  the paper's query/reference protocol, run at the selected layers for both the
  fine-tuned and the pre-trained encoder so the comparison is like for like.

Layer selection follows the paper's headline tables exactly: Task A takes the
layer with the best **train** AUROC, Task B the layer with the best **train**
directory accuracy@1. No test metric is ever used to pick a layer.

**Row alignment.** Every cached matrix is read through an `AlignmentResolver`
(`src/embedding_alignment.py`), which pairs cache rows to split rows by
filename via the `meta.csv` written beside the matrices, rather than by row
position. That matters here because the two caches in play disagree by
construction: the paper's LaTa cache was frozen before the benchmark v1 label
corrections and resolves as **verified-permuted, 17 rows moved**, while the
fine-tuned cache was extracted after them and resolves as **verified-identity,
0 rows moved**. A positional pairing would have scored 17 fine-tuned vectors
against the wrong labels without an error.

**Extraction parity.** The same script re-extracts with the *pre-trained*
weights and diffs against the paper's cached LaTa embeddings, loaded through the
same resolver. Agreement is at float32 rounding noise: max absolute difference
5.7e-05 at layer 1 and 1.4e-06 at layer 12, mean cosine 1.000000. The two
absolute numbers are not comparable as they stand, because layer-1 activations
peak near |x| = 89 and layer-12 near |x| = 0.30; in relative terms both are
about 1e-06. This confirms that text loading, pooling and token filtering are
identical to the paper's pipeline, so any difference in the numbers below is
caused by the fine-tuning, not by the harness.

## Results

Test-set scores. Layer index is the subscript; Task A and Task B select layers
independently, both on train metrics. Task B figures are percentages.

| System | Task A AUROC | Cosine gap | Assignment acc. | Dir. acc.@1 |
|---|---|---|---|---|
| LaTa (pre-trained) | 0.938₁₂ | 0.237₁₂ | 73.8₁ | 72.1₁ |
| LaTa (pre-trained) + ABTT | 0.971₁₂ | 0.525₁₂ | **88.5₈** | **86.1₈** |
| LaTa (fine-tuned) | **0.984₁₂** | 0.387₁₂ | 83.4₁₂ | 81.6₁₂ |
| LaTa (fine-tuned) + ABTT | 0.970₁₂ | **0.548₁₂** | 87.8₁₂ | 85.2₁₂ |

**Where the pre-trained rows come from.** They are *copied* out of the paper's
`phase_resubmit_results.csv` by `build_comparison`, not recomputed: only the
fine-tuned bases are passed through `evaluate_layers`. They therefore match the
published headline tables by construction, benchmark v1 included.

**The ceiling is where the zero-shot pipeline already is.** On directory
routing, ABTT on the frozen encoder scores 86.1 and the fine-tuned encoder with
ABTT scores 85.2. Supervision does not buy a better routing system here; it buys
a better *raw* representation, and post-processing had already recovered that
gain without any labels. On benchmark v1 the supervised system in fact lands
0.9 points **below** the label-free one, which is inside the seed spread (see
the 5-seed table, where the two are 0.877 and 0.877).

**Fine-tuning without ABTT does not reach ABTT without fine-tuning.** Contrastive
training lifts the uncorrected last layer a long way (72.1 to 81.6 dir. acc.@1,
0.938 to 0.984 AUROC), but ABTT on the frozen model still routes better (86.1).
The 565 available pairs are simply not much supervision.

**ABTT still adds after fine-tuning, on Task B only.** Routing improves 81.6 to
85.2 (+3.6 points), so the whitening-style correction is doing something the
contrastive objective did not. Task A moves the other way: AUROC drops 0.984 to
0.970. Once supervision has separated the pairs, removing dominant components
costs a little ranking signal while still helping the thresholded decision,
which is what the larger cosine gap (0.387 to 0.548) reflects.

**Fine-tuning does not repair the mid-depth collapse.** In the fine-tuned model,
layers 2 through 11 still sit at 0.50 to 0.57 AUROC, essentially unchanged from
the pre-trained model, and the cosine gap over that range is near zero or
negative: +0.024 at layer 2, then -0.049 to -0.066 at layers 3 through 11. The
pre-trained model is +0.019 at layer 2 and -0.048 to -0.074 over layers 3 to 11,
so the whole band barely moves. Contrastive training fixes the layer its loss is
attached to and leaves the anisotropy of the middle layers intact; ABTT lifts
every layer into the 0.96 to 0.98 band both before and after fine-tuning (the
fine-tuned ABTT sweep spans 0.9625 to 0.9751 across layers 1-12). This is direct
evidence that the collapse the paper documents is a property of the
representation geometry, not a deficiency that end-task supervision happens to
fix.

### Task B under the 5-seed protocol

The single-seed Task B split is one draw of the query/reference assignment, so
the same four systems were re-scored with the paper's multi-seed protocol,
seeds 42 to 46, at each system's selected layer.

| System | Layer | Dir. acc.@1 | Existing | New |
|---|---|---|---|---|
| LaTa (pre-trained) | 1 | 0.731 ± 0.010 | 0.563 | 0.949 |
| LaTa (pre-trained) + ABTT | 8 | **0.877 ± 0.004** | 0.840 | 0.923 |
| LaTa (fine-tuned) | 12 | 0.834 ± 0.009 | 0.784 | 0.899 |
| LaTa (fine-tuned) + ABTT | 12 | **0.877 ± 0.008** | 0.852 | 0.908 |

Averaging over seeds sharpens the reading. ABTT on the frozen encoder (0.8767)
and the full supervised ceiling (0.8766) are about one ten-thousandth apart, far
inside one standard deviation of either. The gap ABTT closes on the fine-tuned
model (0.834 to 0.877, +4.2 points) is roughly five standard deviations, so ABTT
after fine-tuning is a real effect on routing rather than seed noise.

### Does ABTT still add anything after fine-tuning?

Yes on Task B, no on Task A, and the two answers are consistent. On the
fine-tuned last layer ABTT lifts routing by 3.6 points single-seed and 4.2 points
over five seeds, and it widens the cosine gap from 0.387 to 0.548, which is what
makes a single global threshold $\tau$ work. It costs 1.4 points of Task A AUROC,
because ranking does not need a threshold and the removed components still
carried some ordering signal.

**Read $D=10$ as a boundary hit, not a finding.** The sweep picks $D=10$ at every
fine-tuned layer, the same value it picks before fine-tuning, but 10 is the top of
the paper's grid ($D \in \{1,2,3,5,7,10\}$). The sweep never had the option of
going higher, and `abtt_optimal` is therefore numerically identical to
`abtt_fixed` in all 24 layer x method rows.

The practical consequence for the paper: fine-tuning and ABTT are not additive.
They arrive at the same place, and the correction gets there without labels.

### Why the ceiling is, if anything, overstated

No test file was trained on, and no dev file contributed a training pair; the
carve is directory-disjoint and was verified as such. But witnesses inside one
directory are near-duplicate hand copies of the same source text, and 206 of the
535 test query files (38.5%) sit in a directory that supplied training pairs. The
fine-tuned encoder has therefore seen near-copies of about two fifths of the
routable test items. That is not leakage under the split's own definition, but it
does flatter the fine-tuned rows.

This cuts in favour of the conclusion. The finding is that the ceiling is where
the zero-shot pipeline already is; an overstated ceiling makes that reading
conservative, because the honest ceiling would sit at or below the number
reported here.

## Benchmark v1 re-run

The first run of this experiment (#123, PR #134) trained and scored on the
pre-#131 split. Benchmark v1 moves `BN2123.89r.5.txt` from `Can.apost.48` to
`Can.apost.49` and `BN2123.89r.6.txt` from `Can.apost.49` to `Can.apost.50`, so
the whole pipeline was re-run from scratch on the corrected labels: same seed,
same config, same dev-carve protocol.

### What moved in the pair set

Both corrected files are in the affected window, and both matter to training:

| | Pre-correction | Benchmark v1 |
|---|---|---|
| Train directories with >= 2 files | 190 | 190 |
| Positive pairs available in train | 565 | 565 |
| Dev directories / files | 28 / 72 | 28 / **71** |
| Fit directories / **training pairs** | 162 / **497** | 162 / **499** |
| Batches per epoch | 31 | **32** |

The dev directories are the same 28 either way; the carve is drawn from the
directory list, whose length did not change. What changed is which pool one file
sits in. `BN2123.89r.5.txt` is a train file that was in `Can.apost.48`, a *dev*
directory, so it contributed no training pair; it now sits in `Can.apost.49`, a
*fit* directory with two other train files.

**Exactly two pairs enter the training set and none leave:**

| Pair | Directory |
|---|---|
| `BN2123.89r.5.txt` with `C1525.7v.6.txt` | `Can.apost.49` |
| `BN2123.89r.5.txt` with `Hat42.149r.3.txt` | `Can.apost.49` |

The other 497 pairs are identical, file for file. `BN2123.89r.6.txt` is a test
query and never enters training; its gold answer moves to `Can.apost.50`, which
is what adds the 596th test positive pair.

### What moved in the numbers

Single-seed test scores, PR #134's published run to this one:

| System | Task A AUROC | Cosine gap | Assignment acc. | Dir. acc.@1 |
|---|---|---|---|---|
| LaTa (pre-trained) | 0.938 → 0.938 | 0.238 → 0.237 | 73.8 → 73.8 | 72.1 → 72.1 |
| LaTa (pre-trained) + ABTT | 0.971 → 0.971 | 0.525 → 0.525 | 88.5 → 88.5 | 86.1 → 86.1 |
| LaTa (fine-tuned) | 0.984 → 0.984 | 0.385 → 0.387 | 83.6 → 83.4 | 81.7 → 81.6 |
| LaTa (fine-tuned) + ABTT | 0.970 → 0.970 | 0.548 → 0.548 | 88.3 → 87.8 | 85.9 → 85.2 |

5-seed Task B directory accuracy@1: pre-trained 0.731 → 0.731, pre-trained +
ABTT 0.876 → 0.877, fine-tuned 0.836 → 0.834, fine-tuned + ABTT 0.879 → 0.877.

The largest single move is 0.7 points, on the fine-tuned + ABTT single-seed
routing row. Two extra training pairs out of 499 and one file leaving a 72-file
dev pool are not expected to move a supervised system further than that, and
they did not.

**All three findings survive, and the headline one gets slightly stronger.**
On the previous run the two routing systems were 0.2 points apart single-seed
(label-free ABTT 86.1, supervised ceiling 85.9) and 0.3 points apart over five
seeds, the other way round (0.876 against 0.879). On benchmark v1 the five-seed
figures are 0.8767 and 0.8766, and single-seed the label-free system leads by
0.9 points. The claim is that the two land in the same place; over five seeds
they now land there almost exactly, and whichever way the single-seed draw
falls, no version of this experiment has supervision buying a better routing
system than the correction does.

## Reproducing

Two jobs. Only the first needs a GPU, and it is the only GPU job this work was
approved for. Scoring reads cached `.npy` files, which the repo's budget rule
sends to the CPU partition, so it must not sit inside the GPU reservation.

```bash
# 1. GPU: fine-tune, extract all 12 layers, parity-check against the paper's cache.
sbatch slurm/resubmit/finetune_lata_ceiling.sbatch

# 2. CPU: Task A, Task B, 5-seed Task B, comparison CSV, LaTeX rows.
sbatch slurm/resubmit/finetune_lata_ceiling_eval.sbatch
```

Both accept `CODE_ROOT` (scripts, `src/`, `data/`) and `REPO_ROOT` (the `runs/`
tree) as environment overrides; they are the same path in a normal checkout and
differ only when submitting from a git worktree.

Regenerating only `tables/finetune_ceiling.tex`, after the CSVs exist, needs
neither job and no recompute: `--stages report` rebuilds the comparison from
`finetune_lata_layer_results.csv`, reuses the saved 5-seed aggregate, and
rewrites the table.

The two functions that decide whether the ceiling is honest, the directory-level
dev carve and the directory-disjoint batching, live in `src/finetune_pairs.py`
rather than in the CLI. That module imports nothing heavier than pandas, so
`tests/test_finetune_ceiling_pairs.py` runs on a clean CI checkout instead of
being skipped for want of torch.

### The generated table's caption

`overleaf_drafts/tables/finetune_ceiling.tex` is generated, header
`% generated table`, and it names no repository path because that directory
ships to Overleaf (#117). Its caption and notes used to carry literals ("the 565
positive pairs", "epoch 7, the terminal epoch"), which is how a re-run on a
changed split could ship new numbers under old prose with nothing to flag it.
`CeilingFacts` in the generator now derives all of them from the run that
produced the rows: the pair count, the selected epoch and whether it was
terminal, whether the $D$ sweep hit the top of its grid, and the near-duplicate
overlap statistic. `tests/test_finetune_ceiling_caption.py` changes each input
and asserts the caption follows.

Outputs:

| Path | Contents |
|---|---|
| `runs/active/resubmit/finetune/dev_curve.csv` | per-epoch train loss and dev metrics |
| `runs/active/resubmit/finetune/selection.json` | selected epoch and its dev metric |
| `runs/active/resubmit/finetune/dev_directories.csv`, `train_pairs.csv` | the exact dev carve and training pairs |
| `runs/active/resubmit/finetune/encoder_best.pt` | selected encoder weights |
| `runs/active/resubmit_finetune_bases/phase9_bases/bowphs_LaTa-ft/hidden_mean_tokempty/` | fine-tuned embeddings, layers 1-12, plus the `meta.csv` recording their row order |
| `runs/active/resubmit/results/finetune/finetune_lata_layer_results.csv` | every layer x method row |
| `runs/active/resubmit/results/finetune/finetune_lata_ceiling_comparison.csv` | the comparison table above |
| `runs/active/resubmit/results/finetune/finetune_lata_mseed_*.csv` | 5-seed Task B |
| `overleaf_drafts/tables/finetune_ceiling.tex` | generated table rows |

Nothing under `runs/` is committed.

## Compute

Benchmark v1 re-run of 2026-09-06, from `sacct`:

| Job | Partition | Elapsed | Reserved | State |
|---|---|---|---|---|
| 21847379 `ft_lata_ceiling` | `gpuA100x4`, 1x A100-40GB | **00:01:29** | 00:15:00 | COMPLETED |
| 21847414 `ft_lata_eval` | `cpu`, 8 cores | 00:02:46 | 00:30:00 | COMPLETED |

**GPU cost: 89 seconds of A100 wall time**, against a 900-second reservation,
which is what SLURM actually charges. Training is 7 epochs of 32 steps at 32
sequences of up to 512 tokens; extraction is two passes over 1,705 files (one
for the parity check, one for the fine-tuned embeddings). The watchdog was armed
across both submissions and reported both COMPLETED.

The original run (#123) cost 00:01:24 on the GPU against a 00:45:00 reservation,
and 00:03:18 on the CPU against 04:00:00. The `--time` values were trimmed to 15
and 30 minutes after it, and this run confirms both are right: 10x and 11x
margin over measured elapsed time.

Seeds: 42 throughout (dev carve, batch order, Torch/NumPy/Python RNGs), and
42 to 46 for the multi-seed Task B protocol.
