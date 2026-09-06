# Supervised fine-tuning reference ceiling (LaTa)

Issue #123, epic #109. **This is a reference ceiling, not a proposed method.**
The paper's pipeline is zero-shot: it never sees a labelled pair. This
experiment asks the complementary question a reviewer will ask anyway, namely
how much of the gap to a perfect system is left once a model is allowed to
train on the task's own supervision. The answer bounds what post-processing on
frozen representations can be expected to achieve, and it is reported as a
bound, never as a system we advocate.

Everything below is fine-tuned and selected on the TRAIN split only. The test
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
- 15% of those directories (28 directories, 72 files) are held out as DEV.
- The remaining 162 directories yield 497 training pairs.
- Dev metric: directory accuracy@1 inside the dev pool, i.e. the fraction of
  dev files whose nearest other dev file is from the same directory. AUROC over
  dev pairs breaks ties. The checkpoint with the best dev accuracy@1 is the one
  extracted from; epoch 0 in the dev curve is the pre-trained encoder, so the
  curve shows what training actually bought.

The run early-stopped after epoch 7 and selected epoch 7. Directory accuracy@1
saturated at epoch 4 on a 72-file pool, where one file is worth 1.4 points, so
AUROC over the 2,556 dev pairs is what separates the last four epochs.

| Epoch | Train loss | Dev dir. acc.@1 | Dev AUROC |
|---|---|---|---|
| 0 (pre-trained) | | 0.917 | 0.931 |
| 1 | 0.816 | 0.917 | 0.947 |
| 2 | 0.595 | 0.931 | 0.958 |
| 3 | 0.414 | 0.958 | 0.963 |
| 4 | 0.353 | 0.972 | 0.965 |
| 5 | 0.297 | 0.972 | 0.966 |
| 6 | 0.299 | 0.972 | 0.967 |
| **7 (selected)** | **0.259** | **0.972** | **0.967** |

**Selected checkpoint: epoch 7, dev directory accuracy@1 0.972, dev AUROC 0.967.**

**Epoch 7 is the terminal epoch of the sweep.** The budget was 8 epochs with
patience 3, and patience fired after epoch 7, so the selected checkpoint is also
the last one trained. Dev accuracy@1 had been flat at 0.972 since epoch 4 and the
AUROC tiebreak was still creeping up by +0.0002 per epoch, which reads as
saturation rather than truncation. Still, nothing here rules out that a longer or
larger training run would go further, so every number below is a ceiling **at
this training budget**, not an asymptote.

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

**Extraction parity.** The same script re-extracts with the *pre-trained*
weights and diffs against the paper's cached LaTa embeddings. Agreement is at
float32 rounding noise: max absolute difference 5.7e-05 at layer 1 and 1.4e-06 at
layer 12, mean cosine 1.000000. The two absolute numbers are not comparable as
they stand, because layer-1 activations peak near |x| = 89 and layer-12 near
|x| = 0.30; in relative terms both are about 1e-06. This confirms that row order,
text loading, pooling and token filtering are identical to the paper's pipeline,
so any difference in the numbers below is caused by the fine-tuning, not by the
harness.

## Results

Test-set scores. Layer index is the subscript; Task A and Task B select layers
independently, both on train metrics. Task B figures are percentages.

| System | Task A AUROC | Cosine gap | Assignment acc. | Dir. acc.@1 |
|---|---|---|---|---|
| LaTa (pre-trained) | 0.938₁₂ | 0.238₁₂ | 73.8₁ | 72.1₁ |
| LaTa (pre-trained) + ABTT | 0.971₁₂ | 0.525₁₂ | 88.5₈ | **86.1₈** |
| LaTa (fine-tuned) | **0.984₁₂** | 0.385₁₂ | 83.6₁₂ | 81.7₁₂ |
| LaTa (fine-tuned) + ABTT | 0.970₁₂ | **0.548₁₂** | 88.3₁₂ | 85.9₁₂ |

**Where the pre-trained rows come from.** They are *copied* out of the paper's
`phase_resubmit_results.csv` by `build_comparison`, not recomputed: only the
fine-tuned bases are passed through `evaluate_layers`. They therefore match the
published headline tables by construction. Separately, and as a check on the
harness rather than on the table, cached pre-trained embeddings were re-scored
through the same evaluator (`run_resubmit_evaluate.evaluate_single`) and
reproduced those CSV values, agreeing to 1e-08 on AUROC and exactly elsewhere.
So the comparison is like for like, but the agreement is a verified property of
the code path, not something the comparison table itself demonstrates.

**The ceiling is where the zero-shot pipeline already is.** On directory routing,
ABTT on the frozen encoder scores 86.1 and the fine-tuned encoder with ABTT
scores 85.9. Supervision does not buy a better routing system here; it buys a
better *raw* representation, and post-processing had already recovered that
gain without any labels.

**Fine-tuning without ABTT does not reach ABTT without fine-tuning.** Contrastive
training lifts the uncorrected last layer a long way (72.1 to 81.7 dir. acc.@1,
0.938 to 0.984 AUROC), but ABTT on the frozen model still routes better (86.1).
The 565 available pairs are simply not much supervision.

**ABTT still adds after fine-tuning, on Task B only.** Routing improves 81.7 to
85.9 (+4.2 points), so the whitening-style correction is doing something the
contrastive objective did not. Task A moves the other way: AUROC drops 0.984 to
0.970. Once supervision has separated the pairs, removing dominant components
costs a little ranking signal while still helping the thresholded decision,
which is what the larger cosine gap (0.385 to 0.548) reflects.

**Fine-tuning does not repair the mid-depth collapse.** In the fine-tuned model,
layers 2 through 11 still sit at 0.50 to 0.57 AUROC, essentially unchanged from
the pre-trained model, and the cosine gap over that range is near zero or
negative: +0.024 at layer 2, then -0.049 to -0.064 at layers 3 through 11. The
pre-trained model is +0.021 at layer 2, so layer 2 barely moves either.
Contrastive training fixes the layer its loss is attached to and leaves the
anisotropy of the middle layers intact; ABTT lifts every layer into the 0.96 to 0.98 band both before and
after fine-tuning. This is direct evidence that the collapse the paper documents
is a property of the representation geometry, not a deficiency that end-task
supervision happens to fix.

### Task B under the 5-seed protocol

The single-seed Task B split is one draw of the query/reference assignment, so
the same four systems were re-scored with the paper's multi-seed protocol,
seeds 42 to 46, at each system's selected layer.

| System | Layer | Dir. acc.@1 | Existing | New |
|---|---|---|---|---|
| LaTa (pre-trained) | 1 | 0.731 ± 0.009 | 0.562 | 0.949 |
| LaTa (pre-trained) + ABTT | 8 | 0.876 ± 0.004 | 0.839 | 0.923 |
| LaTa (fine-tuned) | 12 | 0.836 ± 0.012 | 0.777 | 0.912 |
| LaTa (fine-tuned) + ABTT | 12 | **0.879 ± 0.007** | 0.850 | 0.917 |

Averaging over seeds does not change the reading. The gap between ABTT on the
frozen encoder (0.876) and the full supervised ceiling (0.879) is 0.4 points,
inside one standard deviation of either. The gap ABTT closes on the fine-tuned
model (0.836 to 0.879, +4.3 points) is several standard deviations, so ABTT
after fine-tuning is a real effect on routing rather than seed noise.

### Does ABTT still add anything after fine-tuning?

Yes on Task B, no on Task A, and the two answers are consistent. On the
fine-tuned last layer ABTT lifts routing by 4.2 points single-seed and 4.3 points
over five seeds, and it widens the cosine gap from 0.385 to 0.548, which is what
makes a single global threshold $\tau$ work. It costs 1.5 points of Task A AUROC,
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

## Reproducing

Two jobs. Only the first needs a GPU, and it is the only GPU job this issue was
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

The two functions that decide whether the ceiling is honest, the directory-level
dev carve and the directory-disjoint batching, live in `src/finetune_pairs.py`
rather than in the CLI. That module imports nothing heavier than pandas, so
`tests/test_finetune_ceiling_pairs.py` runs on a clean CI checkout instead of
being skipped for want of torch.

Outputs:

| Path | Contents |
|---|---|
| `runs/active/resubmit/finetune/dev_curve.csv` | per-epoch train loss and dev metrics |
| `runs/active/resubmit/finetune/selection.json` | selected epoch and its dev metric |
| `runs/active/resubmit/finetune/dev_directories.csv`, `train_pairs.csv` | the exact dev carve and training pairs |
| `runs/active/resubmit/finetune/encoder_best.pt` | selected encoder weights |
| `runs/active/resubmit_finetune_bases/phase9_bases/bowphs_LaTa-ft/hidden_mean_tokempty/` | fine-tuned embeddings, layers 1-12 |
| `runs/active/resubmit/results/finetune/finetune_lata_layer_results.csv` | every layer x method row |
| `runs/active/resubmit/results/finetune/finetune_lata_ceiling_comparison.csv` | the comparison table above |
| `runs/active/resubmit/results/finetune/finetune_lata_mseed_*.csv` | 5-seed Task B |
| `overleaf_drafts/tables/finetune_ceiling.tex` | generated table rows |

Nothing under `runs/` is committed.

## Compute

Run of 2026-09-06, from `sacct`:

| Job | Partition | Elapsed | Reserved | State |
|---|---|---|---|---|
| 21845735 `ft_lata_ceiling` | `gpuA100x4`, 1x A100-40GB | **00:01:24** | 00:45:00 | COMPLETED |
| 21845746 `ft_lata_eval` | `cpu`, 8 cores | 00:03:18 | 04:00:00 | COMPLETED |

**GPU cost: 1 minute 24 seconds of A100 wall time**, against the roughly two
hours approved for this issue. Training is 7 epochs of 31 steps at 32 sequences
of up to 512 tokens; extraction is two passes over 1,705 files (one for the
parity check, one for the fine-tuned embeddings).

The `--time` values in both sbatch files have since been trimmed to match
(15 minutes for the GPU job, 30 for the CPU one), because SLURM charges the
reserved wall time rather than the elapsed time. The
four-hour CPU reservation was sized from a run on a saturated login node, where
the same scoring took 22 minutes; a dedicated node did it in 3.

Seeds: 42 throughout (dev carve, batch order, Torch/NumPy/Python RNGs), and
42 to 46 for the multi-seed Task B protocol.
