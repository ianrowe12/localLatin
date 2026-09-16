# Supervised fine-tuning reference ceiling (LaTa, Qwen3-0.6B and KaLM-mini)

Issues #123, #138, #194 and #210, epic #109. **This is a reference ceiling, not
a proposed method.** The paper's pipeline is zero-shot: it never sees a labelled
pair. This experiment asks the complementary question a reviewer will ask
anyway, namely how much of the gap to a perfect system is left once a model is
allowed to train on the task's own supervision. The answer bounds what
post-processing on frozen representations can be expected to achieve, and it is
reported as a bound, never as a system we advocate.

**Three models, because one was not defensible.** #123 fine-tuned LaTa alone, on
the grounds that it was the strongest model. Siddique's objection on 2026-09-14
was that this is a property of the pick, not an argument: a ceiling measured on
the one Latin-pretrained encoder says nothing about whether the finding is about
supervision or about Latin pre-training. #194 therefore runs the identical
recipe on Qwen3-Embedding-0.6B, which never saw Latin as a pre-training target
and is a decoder rather than a T5 encoder. #210 adds KaLM-mini, which owns the
paper's best zero-shot ABTT cell (91.7 assignment accuracy, and 89.4 directory
accuracy at rank 1 tied with Qwen3-0.6B), so the recipe is measured against the
hardest zero-shot row it has to clear. Same objective, optimiser, schedule,
batch size, seed, dev carve, early stopping and evaluator; the three ceilings
are comparable by construction.

**Every ceiling that was run is reported.** The table carries all three models
whichever side of its own zero-shot row each lands on, and the main-text
sentence is derived per model from the cells. A model is never dropped for its
result.

**The answer is that they do not agree, and the disagreement is the result.**
On LaTa the label-free correction reaches the supervised ceiling; on Qwen3-0.6B
supervision goes 1.8 points past it and on KaLM-mini 2.1 points past it, in both
cases past every zero-shot ABTT cell in the paper. One model of three matches,
two do not. The ceiling is a property of the model, not of the pipeline. Full
reading in *Do the three models agree?* below.

Every LaTa number below comes from **benchmark v1** (`benchmark_v1.md`),
re-trained and re-scored end to end on the corrected labels under #138. What
that changed is in *Benchmark v1 re-run* below; the verdict did not move. The
Qwen3-0.6B and KaLM-mini runs are benchmark v1 from the start.

Everything here is fine-tuned and selected on the TRAIN split only. The test
split is untouched until the final evaluation, which uses the paper's own
evaluator with no changes.

## Setup

The recipe is one recipe. Only the encoder changes.

| Item | Value |
|---|---|
| Models | `bowphs/LaTa` (T5 encoder stack, 12 blocks, 110M), `Qwen/Qwen3-Embedding-0.6B` (decoder stack, 28 blocks, 596M) and `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` (decoder-shaped stack, 24 blocks, 494M), all mean-pooled |
| Objective | Symmetric InfoNCE over positive pairs with in-batch negatives (the objective behind sentence-transformers' MultipleNegativesRankingLoss, implemented here because the environment has no `sentence_transformers`) |
| Temperature | 0.05 |
| Training pairs | all within-directory pairs from train directories with >= 2 files, minus the dev carve |
| Batching | 16 pairs (32 sequences) per step, with **no two pairs from the same directory in a batch**, so in-batch negatives are always true negatives |
| Optimiser | AdamW, lr 2e-5, weight decay 0.01, linear warmup 10% then linear decay, grad clip 1.0 |
| Precision | bf16 autocast, fp32 master weights |
| Epochs | up to 8, early stop after 3 epochs without dev improvement |
| Seed | 42 (Python, NumPy, Torch; also the dev carve and the batch order) |
| Tokenisation | max_length 512, `tokenizer_empty` token filter, identical to the paper's extraction |
| Memory | LaTa trains as is; Qwen3-0.6B and KaLM-mini use gradient checkpointing (see *Recipe delta* below) |

**Pooling parity matters and is not automatic.** The paper reports both decoder
models on hidden-state **mean** pooling with the `tokenizer_empty` filter, not on
the last-token pooling a decoder embedding model is usually used with
(`runs/active/resubmit_bases/phase9_bases/Qwen_Qwen3-Embedding-0.6B/hidden_mean_tokempty/config.json`
and the same file under
`KaLM-Embedding_KaLM-embedding-multilingual-mini-instruct-v2.5/`; both caches
also hold a `hidden_lasttok_tokempty/` sibling, which is not what the headline
tables use). Fine-tuning and extraction here use mean pooling for exactly that
reason: a ceiling pooled differently from the zero-shot row it is compared
against is not a comparison. `scripts/resubmit/finetune_ceiling.py` loads a
non-seq2seq checkpoint with `AutoModel` and trains on `hidden_states[-1]`, which
is what `extract_encoder_cli.py` writes as `hidden_layer28_embeddings.npy` for
Qwen3-0.6B and `hidden_layer24_embeddings.npy` for KaLM-mini.

**KaLM-mini needs `--trust_remote_code`, and that is a correctness requirement
rather than a convenience.** Its config maps `AutoModel` to its own
`modeling.Qwen2Model` with `is_causal: false`, so the stack attends
bidirectionally. Loading it without the remote code would build a causal stack
and a different representation from the one the paper's KaLM-mini rows were
extracted with. `src/extract_encoder_cli.py` is invoked with the same flag for
those rows (`slurm/resubmit/resubmit_extract_kalm.sbatch`), and the parity check
below is what proves the two agree.

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

All three models draw the same 28 dev directories and train on the same 499
pairs: the carve is a function of the split and the seed, not of the model.

### LaTa's dev curve

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

All three models are evaluated the same way. Fine-tuned mean-pooled embeddings
are extracted for all 1,705 labelled files at every encoder layer (1-12 for
LaTa, 1-28 for Qwen3-0.6B, 1-24 for KaLM-mini) and written in the canonical
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
position. That matters here because the caches in play disagree by
construction: the paper's zero-shot caches were frozen before the benchmark v1
label corrections and resolve as **verified-permuted, 17 rows moved**, while a
fine-tuned cache is extracted after them and resolves as **verified-identity,
0 rows moved**. A positional pairing would have scored 17 fine-tuned vectors
against the wrong labels without an error.

**Extraction parity.** The same script re-extracts with the *pre-trained*
weights and diffs against that model's cached embeddings, loaded through the
same resolver. Agreement is at float32 rounding noise in every model.

| Model | Layer | max abs. diff | mean cosine | Source |
|---|---|---|---|---|
| LaTa | 1 | 5.7e-05 | 1.000000 | job 21847379 |
| LaTa | 12 | 1.4e-06 | 1.000000 | job 21847379 |
| Qwen3-0.6B | 1 | 2.325e-06 | 1.000000 | job 22080571 |
| Qwen3-0.6B | 28 | 4.625e-05 | 1.000000 | job 22080571 |
| KaLM-mini | 1 | 9.537e-06 | 1.000000 | job 22099472 |
| KaLM-mini | 24 | 2.074e-05 | 1.000000 | job 22099472 |

Each row is what that job's own parity stage logged over all 1,705 files, and
the same values sit in the `parity` block of its `run_info.json`. The absolute
numbers are not comparable across rows, because activation scales differ by
orders of magnitude between layers and models; the mean cosine is what makes
them read the same, and in relative terms every row is about 1e-06. This
confirms that text loading, pooling and token filtering are identical to the
paper's pipeline, so any difference in the numbers below is caused by the
fine-tuning, not by the harness.

## LaTa results

Test-set scores. Layer index is the subscript; Task A and Task B select layers
independently, both on train metrics. Task B figures are percentages.
(The paper's `tables/finetune_ceiling.tex` prints plain cells since issue #219; its layers are listed in `tables/selected_layers.tex`, `tab:selected_layers`.)

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

The practical consequence for the paper: on LaTa, fine-tuning and ABTT are not
additive. They arrive at the same place, and the correction gets there without
labels. *That is a statement about LaTa.* Qwen3-0.6B and KaLM-mini behave
differently; see *Do the three models agree?*

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

## Qwen3-0.6B: the same recipe on a model that never saw Latin

One GPU job, `slurm/resubmit/finetune_qwen_ceiling.sbatch`: parity check,
contrastive fine-tuning, extraction of all 28 layers. Everything except the
encoder matches LaTa, and the dev carve is a function of the split and the
seed, so every model trains on the same 499 pairs from the same 162 directories
and holds out the same 28.

**Pooling.** Mean pooling with the `tokenizer_empty` filter at max_length 512,
because that is what the paper's Qwen3-0.6B rows use, not the last-token
pooling a decoder embedding model is normally used with. The parity check
confirms it: re-extracting with the pre-trained weights reproduces the paper's
cache at about 1e-06 relative on layers 1 and 28, mean cosine 1.000000.

**Recipe delta: gradient checkpointing, and nothing else.** Qwen3-0.6B is 596M
parameters against LaTa's 110M encoder, and 32 sequences of 512 tokens through
28 blocks does not fit beside fp32 AdamW state on one A100-40GB. Activations
are recomputed in the backward pass instead of stored, so the gradients are
identical and only the memory bill changes. It is recorded as
`grad_checkpointing` in `run_info.json` and printed in the job log. KaLM-mini
runs with the same flag for the same reason: at 494M parameters it is in
Qwen3-0.6B's size class rather than LaTa's, and the flag buys memory headroom at
no cost to the objective, the schedule or the batch.

### Qwen3-0.6B's dev curve

| Epoch | Train loss | Dev dir. acc.@1 | Dev AUROC |
|---|---|---|---|
| 0 (pre-trained) | | 1.000 (71/71) | 0.9660 |
| 1 | 0.4426 | 0.986 (70/71) | 0.9991 |
| 2 | 0.0313 | 0.986 | 0.9996 |
| **3 (selected)** | **0.0006** | **1.000** | **0.99988** |
| 4 | 0.00004 | 1.000 | 0.99984 |
| 5 | 0.00003 | 1.000 | 0.99985 |
| 6 | 0.00002 | 1.000 | 0.99986 |

Patience fired after epoch 6 and epoch 3 was selected. The curve is nothing
like LaTa's, and the difference is the point: **Qwen3-0.6B's pre-trained
encoder already routes all 71 dev files correctly**, so accuracy has no
headroom and the AUROC tiebreak does all the work. The first run of this job
selected epoch 0 for exactly that reason; see *A dev pool at its resolution
limit* below.

Train loss reaches 4e-05 by epoch 4, so the 499 pairs are memorised. Whatever
supervision buys here is bounded by what those pairs can teach.

### Results

| System | Task A AUROC | Cosine gap | Assignment acc. | Dir. acc.@1 |
|---|---|---|---|---|
| Qwen3-0.6B (pre-trained) | 0.966₂₆ | 0.024₂₆ | 82.3₂₈ | 80.3₂₈ |
| Qwen3-0.6B (pre-trained) + ABTT | 0.973₂ | 0.553₂ | 91.5₅ | 89.4₅ |
| Qwen3-0.6B (fine-tuned) | 0.996₂₈ | 0.684₂₈ | 91.4₂₈ | 90.8₂₈ |
| Qwen3-0.6B (fine-tuned) + ABTT | **0.994₂₇** | **0.716₂₇** | **92.1₂₇** | **91.4₂₇** |

Five-seed Task B, seeds 42 to 46, at each system's selected layer:

| System | Layer | Dir. acc.@1 | Existing | New |
|---|---|---|---|---|
| Qwen3-0.6B (pre-trained) | 28 | 0.824 ± 0.010 | 0.736 | 0.939 |
| Qwen3-0.6B (pre-trained) + ABTT | 5 | 0.905 ± 0.004 | 0.867 | 0.954 |
| Qwen3-0.6B (fine-tuned) | 28 | 0.917 ± 0.006 | 0.872 | 0.976 |
| Qwen3-0.6B (fine-tuned) + ABTT | 27 | **0.923 ± 0.002** | 0.887 | 0.969 |

**The ABTT sweep is a real sweep here, not a boundary hit.** Across the 28
fine-tuned layers `abtt_optimal` picks $D=10$ twelve times, $D=7$ eleven times
and $D \in \{2,3,5\}$ five times; the selected Task B layer takes $D=2$. On
LaTa the sweep pinned $D=10$ at every layer. A fine-tuned Qwen layer needs far
fewer components removed, which is consistent with contrastive training having
already flattened most of the common direction: the fine-tuned cosine gap at
layer 28 is 0.684 before ABTT, against 0.024 for the pre-trained encoder.

## KaLM-mini: the same recipe on the best zero-shot row in the paper

One GPU job, `slurm/resubmit/finetune_kalm_ceiling.sbatch`: parity check,
contrastive fine-tuning, extraction of all 24 layers. Everything except the
encoder matches the other two, and the dev carve is a function of the split and
the seed, so all three models train on the same 499 pairs from the same 162
directories and hold out the same 28.

**Why this model.** LaTa answers "does supervision beat the correction on the
Latin-pretrained encoder" and Qwen3-0.6B answers "is that about Latin
pre-training". Neither answers "can supervision beat the paper's *best*
zero-shot cell", because neither owns it. KaLM-mini does: 91.7 assignment
accuracy at layer 3 under ABTT, and 89.4 directory accuracy at rank 1, the top
of the headline table on the first and tied with Qwen3-0.6B on the second. A
ceiling on that row is the one a reviewer would ask for.

**Pooling.** Mean pooling with the `tokenizer_empty` filter at max_length 512,
because that is what the paper's KaLM-mini rows use, not the last-token pooling
its `hidden_lasttok_tokempty/` sibling cache holds. The parity check confirms
it: re-extracting with the pre-trained weights reproduces the paper's cache to a
max absolute difference of 9.537e-06 at layer 1 and 2.074e-05 at layer 24, mean
cosine 1.000000 at both, which is what job 22099472 logged over all 1,705 files
and what the `parity` block of `run_info.json` holds.

**Recipe delta: gradient checkpointing and `--trust_remote_code`, and nothing
else.** The checkpointing argument is Qwen's argument at 494M parameters rather
than 596M. The remote code is a correctness requirement: the checkpoint maps
`AutoModel` to its own `modeling.Qwen2Model` with `is_causal: false`, so the
stack attends bidirectionally, and the flag is what makes these weights the
weights the paper's KaLM-mini rows were extracted from. Both are recorded in
`run_info.json`.

### KaLM-mini's dev curve

| Epoch | Train loss | Dev dir. acc.@1 | Dev AUROC |
|---|---|---|---|
| 0 (pre-trained) | | 0.972 (69/71) | 0.9567 |
| **1 (selected)** | **0.1716** | **1.000 (71/71)** | **0.9997** |
| 2 | 0.0035 | 0.986 (70/71) | 0.9994 |
| 3 | 0.0001 | 0.986 | 0.9995 |
| 4 | 0.00003 | 0.986 | 0.9995 |

Patience fired after epoch 4 and epoch 1 was selected. Unlike Qwen3-0.6B, this
model's pre-trained encoder does *not* start at the top of the pool: it misses
two dev files, so one epoch of training is a measurable gain and the selector
has something to select. Epochs 2 to 4 sit one file below epoch 1, which is
inside the `1/n_dev` tie window, so AUROC decides between them and epoch 1 wins
on 0.9997 against 0.9994. The window is doing work here too, just in the other
direction: it makes those three epochs *eligible* rather than vetoed, and they
still lose.

Train loss is 3e-05 by epoch 4, so the 499 pairs are memorised, as on
Qwen3-0.6B. Whatever supervision buys is bounded by what those pairs can teach.

### Results

| System | Task A AUROC | Cosine gap | Assignment acc. | Dir. acc.@1 |
|---|---|---|---|---|
| KaLM-mini (pre-trained) | 0.972₂₃ | 0.056₂₃ | 87.5₂₃ | 85.9₂₃ |
| KaLM-mini (pre-trained) + ABTT | 0.981₁ | 0.568₁ | 91.7₃ | 89.4₃ |
| KaLM-mini (fine-tuned) | **0.997₂₄** | 0.636₂₄ | 92.4₂₄ | 91.7₂₄ |
| KaLM-mini (fine-tuned) + ABTT | 0.994₂₄ | **0.651₂₄** | **93.4₂₄** | **92.5₂₄** |

Five-seed Task B, seeds 42 to 46, at each system's selected layer:

| System | Layer | Dir. acc.@1 | Existing | New |
|---|---|---|---|---|
| KaLM-mini (pre-trained) | 23 | 0.873 ± 0.007 | 0.811 | 0.953 |
| KaLM-mini (pre-trained) + ABTT | 3 | 0.909 ± 0.006 | 0.874 | 0.954 |
| KaLM-mini (fine-tuned) | 24 | 0.926 ± 0.008 | 0.879 | 0.987 |
| KaLM-mini (fine-tuned) + ABTT | 24 | **0.930 ± 0.005** | 0.893 | 0.978 |

**ABTT still adds after fine-tuning, and by less than it adds anywhere else.**
Single-seed it lifts routing 91.7 to 92.5 (+0.8 points) and assignment accuracy
92.4 to 93.4; over five seeds 0.926 to 0.930 (+0.4 points, under one standard
deviation of either estimate). Compare LaTa's +3.6 single-seed and Qwen's +0.6.
The fine-tuned cosine gap at layer 24 is already 0.636 before ABTT, against
0.056 for the pre-trained encoder at its own best layer, so contrastive training
has done most of the flattening that ABTT would otherwise do. As on LaTa, Task A
AUROC moves the other way, 0.997 to 0.994.

**The $D$ sweep is a partial boundary hit.** Across the 24 fine-tuned layers
`abtt_optimal` selects $D=10$ nineteen times and $D \in \{2,3,5,7\}$ five times,
including the selected Task B layer, which takes $D=10$. That is between LaTa
(10 everywhere) and Qwen3-0.6B (12 of 28). The generated caption states the
count rather than the verdict, so the reader can see which it is.

## Do the three models agree?

**No, and that is the finding.** One ceiling lands on its own zero-shot ABTT
row and two land above theirs.

| | LaTa | Qwen3-0.6B | KaLM-mini |
|---|---|---|---|
| Zero-shot + ABTT, dir. acc.@1 | 86.1 | 89.4 | 89.4 |
| Fine-tuned + ABTT, dir. acc.@1 | 85.2 | **91.4** | **92.5** |
| Five-seed, zero-shot + ABTT | 0.877 ± 0.004 | 0.905 ± 0.004 | 0.909 ± 0.006 |
| Five-seed, fine-tuned + ABTT | 0.877 ± 0.008 | **0.923 ± 0.002** | **0.930 ± 0.005** |
| Five-seed margin | -0.01 pts | **+1.8 pts** | **+2.1 pts** |

For LaTa, supervision lands where the label-free correction already is: 0.8767
against 0.8766 over five seeds, which is one ten-thousandth apart. For
Qwen3-0.6B it goes 1.8 points past it (0.9227 against 0.9051), which is 4.9
standard deviations of the zero-shot estimate, 9.9 of the tighter fine-tuned one
and 4.4 of the difference. For KaLM-mini it goes 2.1 points past it (0.9296
against 0.9088), 3.5 standard deviations of the zero-shot estimate, 4.1 of the
fine-tuned one and 2.7 of the difference. Both of the latter also clear every
zero-shot ABTT cell in the headline table, whose best is 91.7 assignment
accuracy and 89.4 directory accuracy at rank 1: Qwen3-0.6B reaches 92.1 and
91.4, KaLM-mini 93.4 and 92.5.

**The pattern is one model out of three, and the one is LaTa.** KaLM-mini
matters most to the reading, because it removes the remaining escape route.
Qwen3-0.6B could be dismissed as a stronger model beating a weaker model's
correction; KaLM-mini owns the *best zero-shot ABTT cell in the paper*, and
supervision still clears it by more than Qwen's margin. So the honest
one-sentence answer to Siddique's question is: **the ceiling is not a property
of the pipeline, it is a property of the model.** On the Latin-pretrained
encoder the parameter-free correction reaches what supervision buys; on both
multilingual encoders supervision still has room above it, including above the
best cell the correction produces anywhere in the paper. The paper's claim has
to be stated for LaTa rather than as a general fact about post-processing, and
#194 and #210 are what turned that from an assumption into a measurement.

**No model is dropped for its result.** Two of these three ceilings sit above
the zero-shot rows the paper advertises, which is the uncomfortable direction,
and all three are in the table. `build_headline_tables.py` derives each model's
verdict clause from that model's own cells, so a model that clears a cell cannot
inherit another model's "below everything", and the bare defaults name all three
runs so a re-run cannot quietly ship a two-model block.

Two caveats belong next to that reading, and both cut the same way:

- **The ceilings are flattered.** 206 of the 535 test query files (38.5%) sit in
  a directory that supplied training pairs, and witnesses inside a directory
  are near-duplicate hand copies. No test file was trained on, but Qwen's
  1.8-point and KaLM's 2.1-point margins over their own ABTT rows are upper
  bounds on those margins, not estimates of them.
- **499 pairs are memorised within four epochs on both decoder models.** This is
  a ceiling at this training budget in the literal sense: more epochs will not
  help, and the interesting question of what more *data* would buy is untouched.

### A dev pool at its resolution limit

The first run of this job (22080103) selected **epoch 0** and extracted the
pre-trained weights, which made every fine-tuned row identical to its
pre-trained row. That was not a training failure. The rule from #123 was "best
dev directory accuracy@1, with dev AUROC breaking exact ties", and Qwen3-0.6B
starts at 71 of 71: accuracy cannot improve, one file moved the wrong way at
epoch 1, and 1 file out of 71 is 1.4 points, which is not an exact tie, so the
tiebreak was unreachable. A 1.4-point accuracy difference vetoed an AUROC gain
from 0.966 to 0.9998.

One file is the entire resolution of a 71-file pool, so a difference at or
below it is not a measurement. `is_better_checkpoint` in
`src/finetune_pairs.py` now treats accuracy differences within `1/n_dev` as
ties and lets AUROC decide; anything larger still wins on accuracy outright.
**LaTa's published selection does not move**: its curve gains two files at
epoch 3 and its last four epochs are already exact ties, so epoch 7 is selected
either way. `tests/test_finetune_ceiling_pairs.py` replays both runs' measured
curves and pins LaTa at 7 and Qwen3-0.6B at 3.

The re-run under the fixed rule (22080571) selected epoch 3. Its epochs 1 and 2
reproduce the first run's to four decimals, but epoch 3 reads 71/71 where the
first run read 70/71: at a train loss of 6e-04 one borderline dev file flips
under ordinary GPU non-determinism. That is one more reason to treat a 71-file
dev pool as the weak link in this protocol, and it is worth a line in the
paper's limitations whichever model is being discussed.

**KaLM-mini exercises the same window from the other side, and needed no
re-run.** Its pre-trained encoder misses two of the 71 dev files, so epoch 1 is
a two-file gain and wins on accuracy outright; epochs 2 to 4 then sit one file
below it, which the window makes a tie rather than a loss, and AUROC keeps epoch
1 because 0.9997 beats 0.9994. The rule therefore decides this run at the
tiebreak in both directions and lands on the trained checkpoint either way.

## Benchmark v1 re-run (LaTa)

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

Six jobs, two per model. Only the training halves need a GPU. Scoring reads
cached `.npy` files, which the repo's budget rule sends to the CPU partition,
so it must not sit inside the GPU reservation.

```bash
# GPU: fine-tune, extract every layer, parity-check against the paper's cache.
sbatch slurm/resubmit/finetune_lata_ceiling.sbatch
sbatch slurm/resubmit/finetune_qwen_ceiling.sbatch
sbatch slurm/resubmit/finetune_kalm_ceiling.sbatch

# CPU: Task A, Task B, 5-seed Task B, comparison CSVs.
sbatch slurm/resubmit/finetune_lata_ceiling_eval.sbatch
sbatch slurm/resubmit/finetune_qwen_ceiling_eval.sbatch
# ... then this one, which also writes the generated table.
sbatch slurm/resubmit/finetune_kalm_ceiling_eval.sbatch
```

**Order matters, because `tables/finetune_ceiling.tex` has exactly one
writer.** The table carries every ceiling, so the job for the last model added
renders it. Since #210 that is the KaLM eval job: its two `--tex_extra_run`
specs read LaTa's and Qwen3-0.6B's comparison CSVs and saved caption facts back
off disk and print them above its own rows. The LaTa and Qwen eval jobs write
CSVs only. That is what stops a re-run of one model from silently dropping the
others from the table; when a fourth model is added, the writer role moves to
its eval job the same way.

All six accept `CODE_ROOT` (scripts, `src/`, `data/`) and `REPO_ROOT` (the
`runs/` tree) as environment overrides; they are the same path in a normal
checkout and differ only when submitting from a git worktree. The KaLM eval job
also accepts `TEX_OUT`, for rendering the table somewhere other than
`overleaf_drafts/`.

Regenerating only the table, after the CSVs exist, needs no job and no
recompute: `--stages report` rebuilds the comparison from the model's
`*_layer_results.csv`, reuses the saved 5-seed aggregate, and rewrites the
table.

The three functions that decide whether a ceiling is honest live in
`src/finetune_pairs.py` rather than in the CLI: the directory-level dev carve,
the directory-disjoint batching, and the checkpoint selector. That module
imports nothing heavier than pandas, so `tests/test_finetune_ceiling_pairs.py`
runs on a clean CI checkout instead of being skipped for want of torch.

### One script, one model at a time

`scripts/resubmit/finetune_ceiling.py` (named `finetune_lata_ceiling.py` until
#194) takes the model as a parameter. `--model_name` picks the encoder and the
cache the parity check diffs against, `--display_name` names the rows
("LaTa", "Qwen3-0.6B", "KaLM-mini"), `--trust_remote_code` is passed through to
both loaders for a checkpoint that ships its own modelling code, and
`--results_prefix` keeps the models' CSVs apart in one results directory. A
seq2seq checkpoint contributes its encoder stack; any other checkpoint is loaded
with `AutoModel`, the way `extract_encoder_cli.py` loads the decoder-only
models, so the fine-tuned vectors live in the same space as the zero-shot rows
they are compared against. `count_blocks` finds the block list by attribute
rather than by model name, which is why a third architecture needed no code
change.

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

With several ceilings in one table the same rule applies per model. Each
`CeilingSection` carries its own facts, so the caption says "ABTT rows for LaTa
... select $D=10$ everywhere" and "ABTT rows for Qwen3-0.6B ... 12 of 28 layer
rows select the top of the grid" rather than one model's claim standing for the
rest, and the epoch sentence is named the same way. The pair count is the one
statement made jointly, and only because every carve comes from one split and
one seed; if two runs ever disagree on it, the clause drops the number instead
of quoting one model's count for the others.

Outputs:

LaTa writes into `runs/active/resubmit/finetune/`; Qwen3-0.6B writes into
`runs/active/resubmit/finetune/qwen3_0.6b/`; KaLM-mini writes into
`runs/active/resubmit/finetune/kalm_mini/`. Result CSVs share
`runs/active/resubmit/results/finetune/` and are kept apart by their prefix,
`finetune_lata`, `finetune_qwen3_0.6b` and `finetune_kalm_mini`.

| Path | Contents |
|---|---|
| `<out dir>/dev_curve.csv` | per-epoch train loss and dev metrics |
| `<out dir>/selection.json` | selected epoch and its dev metric |
| `<out dir>/dev_directories.csv`, `train_pairs.csv` | the exact dev carve and training pairs |
| `<out dir>/encoder_best.pt` | selected encoder weights |
| `<out dir>/run_info.json` | config, parity report, caption facts, whether gradient checkpointing was on |
| `runs/active/resubmit_finetune_bases/phase9_bases/<slug>-ft/hidden_mean_tokempty/` | fine-tuned embeddings at every layer, plus the `meta.csv` recording their row order |
| `.../results/finetune/<prefix>_layer_results.csv` | every layer x method row |
| `.../results/finetune/<prefix>_ceiling_comparison.csv` | the comparison table above |
| `.../results/finetune/<prefix>_mseed_*.csv` | 5-seed Task B |
| `overleaf_drafts/tables/finetune_ceiling.tex` | generated table rows |

Nothing under `runs/` is committed.

### The records were flattened once, and put back

Until #194 both jobs dumped `run_info.json` wholesale, so the scoring job
deleted the GPU job's `parity` report, its `selection`, its `train_seconds` and
its `grad_checkpointing` flag, leaving a file that looked complete and read
`grad_checkpointing: false, parity_check: false` for both models of the day.
`merge_run_info` in the CLI now folds each job's record into the file and drops
nothing; a scoring job's own `config` and `total_seconds` land under
`report_config` and `report_total_seconds` so the training job's stay the record
of how the weights were made. `tests/test_finetune_run_info.py` pins that.

The two files already on disk were rebuilt by
`scripts/resubmit/restore_finetune_run_info.py`, which takes `selection` from
the `selection.json` written beside each checkpoint, `config` from the committed
sbatch, and the parity numbers from the GPU job's log (Qwen3-0.6B, job 22080571)
or from the parity table above (LaTa, job 21847379, whose log is no longer under
`slurm/logs`). **`train_seconds` survived nowhere, so it is absent rather than
guessed**, and a `restored` block in each file names every source and lists what
could not be recovered. Re-running the GPU job would regenerate all of it
first-hand; that is the right fix whenever the checkpoints are retrained, and
not worth an A100 to recover a JSON file.

KaLM-mini's record never needed restoring: it was written under the merge rules
from the start. Its file carries `train_seconds` (60.9), `parity`, `selection`,
`grad_checkpointing`, `n_blocks` and `device` from the GPU job, and
`caption_facts` beside the namespaced `report_config`, `report_total_seconds`
and `report_device` from the scoring job, with no `restored` block.
`caption_facts` belongs to the scoring job and not to the GPU job, because it is
written only on the `--tex_out` branch, which only the eval job takes. That is
what the two rebuilt files are approximating.

**`device` is job-scoped too, and learned that the hard way.** It was merged
rather than namespaced until #210, so the scoring job, which the budget rule
sends to the CPU partition, rewrote every finished record to say the weights
were trained on `cpu`, next to a `parity` block and a `train_seconds` that could
only have come from a GPU. `JOB_SCOPED_KEYS` in the CLI now covers `config`,
`total_seconds` and `device`, and `tests/test_finetune_run_info.py` pins that a
second and third scoring pass still leave the training job's values in place.
The one record already damaged was repaired in place from the two job logs and
carries a `repaired` block naming each source.

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

Qwen3-0.6B, 2026-09-14, from `sacct`:

| Job | Partition | Elapsed | Reserved | State |
|---|---|---|---|---|
| 22080103 `ft_qwen_ceiling` (epoch-0 run) | `gpuA100x4`, 1x A100-40GB | **00:03:49** | 00:30:00 | COMPLETED |
| 22080194 `ft_qwen_eval` (epoch-0 run) | `cpu`, 8 cores | 00:09:36 | 01:00:00 | COMPLETED |
| 22080571 `ft_qwen_ceiling` (reported) | `gpuA100x4`, 1x A100-40GB | **00:05:07** | 00:20:00 | COMPLETED |
| 22080685 `ft_qwen_eval` | `cpu`, 8 cores | 00:13:18 | 00:20:00 | CANCELLED |
| 22081016 `ft_qwen_eval` (reported) | `cpu`, 8 cores | 00:05:07 | 00:40:00 | COMPLETED |

**GPU cost: 536 seconds of A100 wall time over two jobs** (229 s + 307 s),
against 1,800 + 1,200 seconds of reservation, which is what SLURM charges. The
second job is the one whose numbers are reported; the first selected epoch 0
under the old rule (see *A dev pool at its resolution limit*). Each run is
about 2.6x LaTa's elapsed time for 5.4x the parameters and 28 layers instead of
12, which gradient checkpointing pays for. Training is up to 8 epochs of 32
steps at 32 sequences of up to 512 tokens; extraction is two passes over 1,705
files, one for the parity check and one for the selected weights. Peak memory
was well inside the 40GB card.

**The CPU eval's `--time` is 40 minutes on purpose.** The same 28-layer sweep
measured 00:09:36 and 00:05:07 on two nodes, and on a third it was still at
layer 18 of 28 after 13 minutes, which is why 22080685 was cancelled rather than
left to hit a 20-minute wall. The evaluate stage writes its CSV only once all 28
layers are done, so a timeout loses the whole sweep; the reservation is sized
for the slow node, not the fast one.

KaLM-mini, 2026-09-15, from `sacct`:

| Job | Partition | Elapsed | Reserved | State |
|---|---|---|---|---|
| 22099472 `ft_kalm_ceiling` | `gpuA100x4`, 1x A100-40GB | **00:02:39** | 00:15:00 | COMPLETED |
| 22099517 `ft_kalm_eval` | `cpu`, 8 cores | 00:04:37 | 00:40:00 | COMPLETED |

**GPU cost: 159 seconds of A100 wall time**, against a 900-second reservation,
which is what SLURM charges. `run_info.json` splits that into 61 seconds of
training (4 epochs of 32 steps before patience fired) and 126 seconds for the
whole GPU stage including both extraction passes over 1,705 files. It landed
between LaTa's 89 s and Qwen3-0.6B's 307 s, as its parameter count and layer
count predict, and the run needed no retry: the dev carve gave the selector
headroom at epoch 0, so the failure mode of #194's first Qwen job did not arise.

The CPU eval's 40-minute reservation is the one sized for Qwen's slow node. This
24-layer sweep took 00:04:37 on `cn099`, so the margin is 8.7x on this node and
the reservation stays where it is until a node is measured that needs more.

Seeds: 42 throughout (dev carve, batch order, Torch/NumPy/Python RNGs), and
42 to 46 for the multi-seed Task B protocol, for every model.
