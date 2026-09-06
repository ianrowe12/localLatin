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

<!-- RESULTS:DEV -->

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
float32 rounding noise (max abs difference ~1e-5 at layer 1, ~1e-7 at layer 12;
mean cosine 1.000000), which confirms that row order, text loading, pooling and
token filtering are identical to the paper's pipeline. Any difference in the
numbers below is therefore caused by the fine-tuning, not by the harness.

<!-- RESULTS:MAIN -->

<!-- RESULTS:MSEED -->

<!-- RESULTS:ABTT -->

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

<!-- RESULTS:COMPUTE -->
