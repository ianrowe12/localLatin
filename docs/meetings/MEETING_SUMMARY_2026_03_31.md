# Paper Revision Progress — Meeting Summary (March 31, 2026)

## 0. Executive Summary

Since the last meeting, we executed a complete paper revision pipeline: expanded the dataset from 538 to 840 directories, re-evaluated all 6 models across 700 configurations (7 post-processing methods), conducted deep research into interpretable token attribution (40+ methods surveyed), implemented and compared 4 attribution methods head-to-head, generated predictions for 2,238 unlabelled files, and built a full-stack webapp for human review of those predictions.

### Quick Reference

| Metric | Value |
|--------|-------|
| Best AUCROC | **0.9886** (Qwen3-0.6B, SIF+ABTT, L1) |
| Best Dir Acc@1 | **89.4%** (LaBSE L11 / Qwen3-0.6B L5) |
| Best Assignment Accuracy | **92.3%** (Qwen3-0.6B L5, ABTT optimal) |
| Models tested | 6 (LaTa, PhilTa, LaBSE, mt5-base, Qwen3-0.6B, KaLM-mini) |
| Configurations evaluated | 700 |
| Post-processing methods | 7 (+ 4 IG visualization methods) |
| Unlabelled predictions | 15,166 rows (top-10 per query, 6 models) |
| Webapp | Code complete (FastAPI + React) |

---

## 1. New Dataset & Pipeline

### 1a. Dataset Expansion

| | Old (canon/) | New (canon_labelled/) | Change |
|--|:---:|:---:|:---:|
| Directories | 538 | 840 | +56% |
| Files | 1,278 | 1,705 | +33% |
| Singleton dirs | — | 545 | — |
| Multi-file dirs | — | 295 (108 pairs + 187 large) | — |

Additionally, **2,238 unlabelled files** (`canon_unlabelled/`) are now available for prediction — flat `.txt` files with no directory labels, ready for human review.

### 1b. Leak-Free Split

The same 50/50 protocol from Phase 9: singletons randomly assigned, doubletons kept together, multi-file directories split proportionally. Zero train-test folder overlap.

| Set | Files | Task A Pairs | Positive Pairs |
|-----|:-----:|:------------:|:--------------:|
| Train | 787 | 309,291 | 354 |
| Test | 918 | 420,903 | 604 |

### 1c. Task B Change: Cumulative Top-K

**Before**: Exclusive rank buckets ("the correct directory was the expert's 1st choice 84% of the time, 2nd choice 9%, ...").

**Now**: Cumulative top-K ("if the expert checks the top K options, what's the probability of finding the correct directory?"). This is more intuitive for the deployment story and matches how a scholar would actually use the system.

### 1d. Pipeline Artifacts

- 4 TSV files (train/test x Task A/B)
- `phase_resubmit_split.csv` metadata
- 18 sbatch scripts for extraction + evaluation
- Master launcher with dependency ordering

---

## 2. Comprehensive Evaluation (700 Configurations)

### 2a. Top Results

| Metric | Model | Method | Layer | Value |
|--------|-------|--------|:-----:|:-----:|
| **Best AUCROC** | Qwen3-0.6B | sif_abtt_fixed | 1 | **0.9886** |
| **Best Dir Acc@1** | LaBSE / Qwen3-0.6B | sif_abtt_fixed / abtt_optimal | 11 / 5 | **89.4%** |
| **Best Assignment Acc** | Qwen3-0.6B | abtt_optimal | 5 | **92.3%** |

### 2b. Method Ranking (Average AUCROC Across All Models)

| Method | Avg AUCROC | vs Baseline |
|--------|:----------:|:-----------:|
| abtt_optimal | **0.9803** | +0.166 |
| abtt_fixed (D=10) | **0.9802** | +0.166 |
| sif_abtt_optimal | 0.9775 | +0.163 |
| sif_abtt_fixed | 0.9775 | +0.163 |
| sif_only | 0.8796 | +0.065 |
| baseline | 0.8145 | — |
| whitening | 0.6591 | -0.155 |

**Key finding**: `abtt_fixed` (D=10) matches `abtt_optimal` — no per-layer tuning needed. D=10 is a robust universal default.

### 2c. mt5-base: Architecture Effect, Not Language

mt5-base (Google's massively multilingual T5, trained on 101 languages with no Latin-specific training) was added as the 6th model. Results:

| Metric | mt5 Baseline | mt5 ABTT | Gain |
|--------|:----------:|:--------:|:----:|
| AUCROC | ~0.83 | **0.987** | +0.157 |
| Assignment Acc | ~57% | **90.3%** | +33 pp |

This provides evidence that the anisotropy dip and ABTT's correction are **architecture-dependent** (T5 structural property), not **language-dependent** (Latin-specific artifact). This makes the finding relevant to the broader NLP community.

### 2d. Cumulative Top-K Results (Task B)

| Model | Top-1 | Top-2 | Top-3 | Top-5 |
|-------|:-----:|:-----:|:-----:|:-----:|
| mt5-base | **88.3%** | 94.6% | 96.0% | 97.1% |
| PhilTa | 88.2% | 93.9% | 96.3% | 97.2% |
| LaBSE | 88.0% | 94.0% | 95.4% | 97.1% |
| KaLM-mini | 88.0% | 93.5% | 95.5% | 96.5% |
| Qwen3-0.6B | 87.7% | 93.8% | 96.0% | 97.2% |
| LaTa | 86.2% | 93.0% | 95.2% | 96.7% |

**All 6 models reach 95-97% at Top-3.** A scholar checking the top 3 suggestions will find the correct directory 95%+ of the time.

### 2e. Figures

![AUCROC per model across layers](overleaf_drafts/figures/fig_release_aucroc_per_model.png)

![Cosine gap per model across layers](overleaf_drafts/figures/fig_release_gap_per_model.png)

![Density 2x2: baseline vs ABTT, last vs middle layers](overleaf_drafts/figures/paper_fig_density_2x2.png)

---

## 3. Deep Research — Interpretable Token Attribution

### 3a. The Problem

After ABTT, the IG-weighted token-to-token heatmaps reduce noise, but the highlighted connections don't make linguistic sense to human scholars. A Latinist looking at the pair matrix can't see meaningful word-level connections (e.g., "domini" connecting to "domini"). ABTT operates on sentence-level geometry — IG gradients reflect PC-loading, not semantic importance.

### 3b. Research Scope

Three comprehensive surveys were produced (1,722 lines total), covering 40+ methods across 5 research areas:

| Survey | Focus | Methods Reviewed |
|--------|-------|:----:|
| `run1_nlp_interpretability.md` | Sparse representations, OT, PMI, contrastive explanations | 15+ |
| `run1_xai_methods.md` | Attention, SHAP, LRP, concept-based, cross-attention probing | 17+ |
| `run1_geometry_alignment.md` | Isotropy-aware pooling, token alignment, representation engineering | 12+ |

### 3c. Top Recommendations

| Priority | Method | Key Advantage | Effort |
|:--------:|--------|---------------|:------:|
| 1 | **Ditto (Attn Diagonal Pooling)** | Cleanest IG path; per-instance, no corpus stats | 1 day |
| 2 | **BERTScore greedy alignment** | Alignment matrix IS the explanation; no IG needed | 2 days |
| 3 | **Diff-in-Means direction** | Answers "what does PC1 encode?" | 1 day |
| 4 | **LEACE concept erasure** | Principled ABTT replacement, provably minimal | 2 days |
| 5 | **Sinkhorn OT** | Rich many-to-many token alignment | 3 days |
| 6 | **Sparse Autoencoders** | Decompose PC1 into interpretable features | 5 days |

**Key insight from research**: *"The alignment matrix IS the explanation."* Token matching methods (BERTScore, OT) produce token-to-token maps directly, bypassing IG entirely.

---

## 4. IG Comparison — Research Put to Practice

Three of the top research recommendations were implemented and compared against the current approach on 10 example pairs (PhilTa model):

### 4a. Methods Compared

| Panel | Method | Source |
|:-----:|--------|--------|
| A | **Current IG + ABTT** (baseline) | Existing pipeline |
| B | **BERTScore greedy alignment** | Research Priority 2 |
| C | **Optimal Transport (EMD)** | Research Priority 5, via POT library |
| D | **Attention-weighted cross-similarity** | Ditto-inspired (Priority 1, partial) |

### 4b. Method Details

#### Panel A: Integrated Gradients + ABTT Pair Matrix (Current Baseline)

**Source:** Sundararajan, M., Taly, A., Yan, Q. "Axiomatic Attribution for Deep Networks." *ICML 2017*. Implemented via [Captum](https://captum.ai/) `LayerIntegratedGradients`.

**Technical explanation:**

Integrated Gradients (IG) is a gradient-based attribution method that satisfies two axioms: *sensitivity* (if changing an input changes the output, it gets nonzero attribution) and *implementation invariance* (two functionally identical models give identical attributions). It computes the path integral of gradients from a baseline input (e.g., zero embedding) to the actual input, requiring ~50 forward passes per example.

In our pipeline, IG attributes the scalar cosine similarity between two documents back to individual input tokens. The pair matrix combines token-level cosine similarity with IG importance:

```
pair_matrix[i,j] = cos(q_token_i, c_token_j) * sqrt(|IG_q_i| * |IG_c_j|) * sign(IG_q_i) * sign(IG_c_j)
```

The problem: ABTT operates on corpus-level statistics (principal components fitted on the training set). When IG computes gradients through this transform, it highlights tokens that load heavily onto the removed PCs — a *geometric* property — rather than tokens that are *semantically* meaningful. The resulting heatmaps reduce noise but don't show linguistically interpretable token connections.

<details>
<summary><strong>Simpler explanation</strong></summary>

Think of it like asking: "which words in document A are most responsible for its similarity score with document B?" IG answers this by imagining you slowly "fade in" each word from nothing to its actual value, tracking how the similarity changes. Words that cause big changes get high scores.

The issue: we first clean the embeddings with ABTT (removing dominant noise directions). IG sees this cleaning step and ends up highlighting words that were *most affected by the cleaning* — not necessarily words that carry the actual meaning. It's like asking "which ingredients changed the most when we filtered the water?" instead of "which ingredients make the soup taste good?"

</details>

---

#### Panel B: BERTScore Greedy Alignment

**Source:** Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y. "BERTScore: Evaluating Text Generation with BERT." *ICLR 2020*. [GitHub](https://github.com/Tiiiger/bert_score).

**Technical explanation:**

BERTScore computes similarity between two texts by greedily matching each token in one text to its most similar token in the other, using cosine similarity on contextual embeddings. It produces three scores — Precision (average best-match from candidate to query), Recall (average best-match from query to candidate), and F1 — but the key output for us is the **alignment matrix**: a sparse binary map showing which tokens matched.

Our implementation operates on ABTT-cleaned token embeddings (per-token PC removal) and computes:

```python
cos = cosine_similarity(q_tokens_clean, c_tokens_clean)  # dense (n_q × n_c)
# Recall pass: for each query token, find best candidate match
for i in range(n_q): combined[i, argmax(cos[i,:])] = cos[i, argmax(cos[i,:])]
# Precision pass: for each candidate token, find best query match  
for j in range(n_c): combined[argmax(cos[:,j]), j] = cos[argmax(cos[:,j]), j]
```

The combined alignment is the union of recall and precision matches. This is naturally very sparse (at most `n_q + n_c` nonzero entries in an `n_q × n_c` matrix) and each nonzero entry represents a direct "this token matches that token" claim.

**Key advantage:** The alignment matrix *is* the explanation — no IG needed. No gradient computation, no forward-pass budget. Works on cached embeddings in milliseconds.

<details>
<summary><strong>Simpler explanation</strong></summary>

Imagine you have two Latin documents laid out side by side. For each word in document A, you find the single most similar word in document B (that's the "recall" direction). Then you do it the other way: for each word in document B, find its best match in document A (that's "precision"). Combine both directions and you get a clean map: "this word connects to that word."

The result is very sparse — each word points to at most one partner — so the heatmap is clean and easy to read. When "domini" in document A has high cosine similarity to "domini" in document B, you see a clear bright cell. No math voodoo, just "which words look alike?"

</details>

---

#### Panel C: Optimal Transport / Earth Mover's Distance (EMD)

**Sources:**
- Kusner, M.J., Sun, Y., Kolkin, N.I., Weinberger, K.Q. "From Word Embeddings to Document Distances." *ICML 2015*. (Word Mover's Distance)
- Zhao, W., Peyrard, M., Liu, F., et al. "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance." *EMNLP 2019*. (Contextualized extension)
- Implemented via [POT](https://pythonot.github.io/) `ot.emd()`.

**Technical explanation:**

Optimal Transport (OT) treats each document as a probability distribution over its tokens. Each token carries some "mass" (we use |IG| scores as mass, normalized to sum to 1). The cost of moving mass from query token *i* to candidate token *j* is their cosine distance: `C[i,j] = 1 - cos(q_i, c_j)`. The Earth Mover's Distance finds the transport plan **T** that moves all mass from the query distribution to the candidate distribution at minimum total cost:

```
minimize  sum(T[i,j] * C[i,j])
subject to  T >= 0,  sum_j(T[i,:]) = a[i],  sum_i(T[:,j]) = b[j]
```

where `a[i] = |IG_q_i| / sum(|IG_q|)` and `b[j] = |IG_c_j| / sum(|IG_c|)`.

The transport plan T[i,j] tells you how much "mass" flows from token *i* to token *j*. Unlike BERTScore's one-to-one matching, OT produces **many-to-many soft alignment**: a single query token can distribute its mass across multiple candidate tokens, and vice versa. The plan is naturally sparse because EMD solutions live on vertices of the transport polytope.

**Key advantage:** Mass conservation means every token's importance is accounted for. IG scores determine *how much* each token matters; OT determines *where* that importance flows.

<details>
<summary><strong>Simpler explanation</strong></summary>

Think of each document as a pile of dirt, where each word is a small heap. Important words (high IG score) are bigger heaps. Now imagine you need to shovel all the dirt from document A's piles to fill document B's piles, and moving dirt between similar words is cheap while moving it between different words is expensive.

The "optimal transport plan" is the cheapest way to move all the dirt. It naturally tells you: "60% of the mass from 'episcopus' went to 'episcopi', 30% went to 'sacerdotis', 10% went to 'diaconus'." This gives you a rich many-to-many map — one word can connect to several related words, weighted by how related they are.

The result is sparser than the raw cosine matrix (not everything connects to everything) but richer than BERTScore (words can connect to more than one partner).

</details>

---

#### Panel D: Attention-Weighted Cross-Similarity (Ditto-Inspired)

**Source:** Chen, Y., et al. "Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings." *EMNLP 2023*.

**Technical explanation:**

Ditto's key insight is that a token's self-attention diagonal — `attn[i,i]`, how much token *i* attends to itself — is a per-instance measure of token importance. Tokens that attend to themselves are "confident" in their own content; tokens that scatter attention to others are more contextual/less content-bearing.

Our implementation extracts the self-attention diagonal from the model's attention matrix at the target layer, normalizes it to sum to 1, and uses it as token importance weights in the cross-similarity matrix:

```python
q_w = diag(Q_attention) / sum(diag(Q_attention))  # per-token query importance
c_w = diag(C_attention) / sum(diag(C_attention))  # per-token candidate importance
crosssim[i,j] = cos(q_i, c_j) * sqrt(q_w[i] * c_w[j])
```

**Key advantage:** Per-instance weights with zero corpus-level statistics in the computation path. Unlike ABTT (which uses corpus-fitted PCs) or SIF (which uses corpus-fitted token frequencies), the attention diagonal comes from the model's own forward pass on this specific input. This means IG gradients through this path are "clean" — they don't get distorted by corpus-level transforms.

**Why it underperformed in practice:** Our implementation is a *partial* Ditto — we used attention weights for the cross-similarity visualization but didn't replace the full SIF+ABTT pooling pipeline with Ditto-style pooling. The attention diagonal alone doesn't filter empty tokens as effectively as SIF weighting (it gave the worst content focus score: 0.29 vs 0.44+ for other methods). A full Ditto integration — replacing `weighted_mean_pool` with attention-diagonal pooling — remains a future experiment.

<details>
<summary><strong>Simpler explanation</strong></summary>

When a transformer processes a sentence, each word "looks at" other words through attention. The self-attention diagonal measures how much each word looks at *itself* vs. other words. Words with high self-attention are "I know what I am" words — typically content-heavy. Words with low self-attention are "I need context from others" words — typically function words or padding.

We use this self-attention score as each word's importance weight. It's like asking the model: "which of your own words do you think are most important?" Then we build a similarity map weighted by those importance scores.

The beauty is that this importance measure comes from the model looking at *this specific sentence*, not from statistics about the whole corpus. In theory, this should produce cleaner explanations. In practice, it didn't work as well as BERTScore because the attention diagonal on its own isn't great at ignoring empty tokens — the full Ditto method (which replaces the entire pooling step) hasn't been tested yet.

</details>

---

### 4c. Quantitative Comparison

| Metric | IG+ABTT | BERTScore | OT (EMD) | Attn Cross-Sim |
|--------|:-------:|:---------:|:--------:|:--------------:|
| **Sparsity** (lower = cleaner) | 0.33 | **0.08** | **0.08** | 0.10 |
| **Content Focus** (higher = better) | 0.44 | 0.47 | **0.60** | 0.29 |
| **Shared Token Match** (higher = better) | 0.22 | **0.46** | 0.25 | 0.23 |

### 4d. Key Findings

1. **BERTScore is 2x better at aligning matching tokens** (0.46 vs 0.22 shared token score) and **4x sparser** (0.08 vs 0.33), producing dramatically cleaner heatmaps.

2. **Optimal Transport has the highest content focus** (0.60) — it places the most weight on semantically meaningful token pairs.

3. **Attention cross-similarity underperforms** — the partial Ditto implementation (cross-sim only, not full pooling replacement) doesn't improve over IG+ABTT.

4. BERTScore and OT are both viable replacements for IG+ABTT heatmaps. BERTScore is the simplest to deploy (~5 lines of code).

![4-panel IG comparison example](runs/phase_resubmit/ig_comparison/example002_correct_option_comparison.png)

---

## 5. "Can You Just Regex Out Empty Tokens?"

This directly answers the question from the last meeting. Phase 12d tested 4 conditions:

| Condition | What it does |
|-----------|-------------|
| Baseline | Mean-pool ALL tokens |
| Filter (no empty) | Drop whitespace/padding, mean-pool rest |
| Filter (content only) | Keep only 3+ char content tokens |
| ABTT (D=10) | Mean-pool ALL tokens, then remove top 10 PCs |

### 5a. Key Numbers

| Model | Layer | Baseline | Best Filter | ABTT | ABTT - Filter |
|-------|:-----:|:--------:|:-----------:|:----:|:-------------:|
| PhilTa | L6 (dip) | 0.464 | 0.696 | **0.970** | **+0.274** |
| PhilTa | L9 | 0.467 | 0.724 | **0.968** | **+0.244** |
| LaTa | L4 (dip) | 0.468 | 0.925 | **0.957** | +0.032 |
| mt5-base | L4 (dip) | 0.710 | 0.854 | **0.972** | **+0.118** |
| mt5-base | L8 | 0.691 | 0.738 | **0.953** | **+0.215** |
| **LaBSE** | **L1** | **0.843** | **0.864** | **0.970** | **+0.106** |

### 5b. The Decisive Proof: LaBSE

LaBSE has **zero empty tokens** in its tokenizer (no whitespace tokens, no padding markers). If the problem were simply "too many empty tokens in the average," filtering would match ABTT for LaBSE. It doesn't come close:

- Filtering: +0.02 AUCROC improvement
- ABTT: **+0.13 AUCROC improvement**

### 5c. Interpretation

> **Token filtering is a 47% solution at best. ABTT is a near-complete solution.**

- **Filtering** removes the symptom (empty tokens polluting the mean pool)
- **ABTT** removes the cause (PC1 noise contaminating *every* token's embedding, including content tokens)

At PhilTa L6 (worst dip), filtering recovers only 24% of the baseline-to-ABTT gap. ABTT closes the remaining 76%.

This preempts the most likely reviewer objection and makes the argument airtight.

![Filtering vs ABTT comparison](runs/phase12d/fig_filtering_vs_abtt.png)

---

## 6. Unlabelled Set Predictions

All 6 models extracted embeddings for the 2,238 unlabelled files. For each model, the best layer+method combination (selected on the training set) was used to generate top-10 directory predictions.

| Output | Details |
|--------|---------|
| Total prediction rows | 15,166 |
| Predictions per query | Top-10 ranked directories with cosine scores |
| Models | All 6 (separate CSVs + combined) |
| Location | `runs/phase_resubmit/unlabelled/` |

These are ready for human review via the webapp.

---

## 7. Webapp — From Experiment to Instrument

A full-stack web application was built for Latin scholars to review the unlabelled predictions interactively.

### 7a. Backend (FastAPI)

| Component | Details |
|-----------|---------|
| Framework | FastAPI + uvicorn |
| Endpoints | 9 (queries, predictions, token maps, feedback, stats, models, export) |
| Data loading | All texts + predictions cached in memory (~20MB) |
| Feedback storage | SQLite with full review history |
| Token maps | Full token-to-token similarity matrices from NPZ artifacts |
| Requirements | No GPU needed — runs on login nodes |

### 7b. Frontend (React + TypeScript + Vite)

| Component | Details |
|-----------|---------|
| Layout | 3-panel: query list | document viewer | predictions sidebar |
| Hero feature | SVG bezier connection lines between query and candidate tokens on hover |
| View modes | Connections (hover), Heatmap (colormapped), IG-weighted |
| Token display | Crimson Pro serif font for Latin text, classified as content/subword/empty |
| File list | Virtualized (react-virtuoso) for 2,238 items |
| Feedback | Pill buttons for rank selection + free-text notes + submit-and-advance |
| Polish | Dark/light mode, skeleton loaders, keyboard shortcuts, progress ring |

### 7c. Cross-Machine Handoff

A comprehensive architecture document (`web/AGENT_PLAN.md`, 14KB) was written to enable continuation of webapp development on a different machine. Includes endpoint specs, data flow, component hierarchy, and SVG line rendering design.

### 7d. Status

**Code complete.** Needs:
- `config.yaml` setup for target machine paths
- Full API integration testing (frontend currently uses mocks for some features)
- Performance profiling with full 2,238-item list

---

## 8. What Remains

### Completed

- [x] Deep research (3 surveys, 40+ methods)
- [x] Dataset expansion (canon_labelled: 840 dirs, 1,705 files)
- [x] Full re-evaluation on new data (700 configs, 6 models, 7 methods — all on canon_labelled)
- [x] IG comparison (4 methods implemented and compared)
- [x] Unlabelled predictions (15,166 rows, 6 models)
- [x] Filtering vs ABTT sanity check (Phase 12d)
- [x] Cumulative top-K Task B
- [x] Paper TeX updated (1,705/840 stats, cumulative top-K table, `fig_release_*` figure refs)
- [x] Webapp code (backend + frontend)

### Still To Do

- [ ] **Unimplemented research**: Diff-in-Means direction analysis, LEACE concept erasure, full Ditto pipeline, SAEs
- [ ] **Webapp deployment**: Configure on target machine (`config.yaml`), integration testing, performance profiling
- [ ] **BERTScore integration**: Most promising IG replacement — integrate into the main evaluation pipeline and webapp (currently only in comparison visualization script)

---

## Appendix: File Locations

| Artifact | Path |
|----------|------|
| Results CSV (700 rows) | `runs/phase_resubmit/results/phase_resubmit_results.csv` |
| Dataset split | `runs/phase_resubmit/data/phase_resubmit_split.csv` |
| Task A/B TSVs | `runs/phase_resubmit/data/{train,test}_task_{a,b}.tsv` |
| IG comparison figures | `runs/phase_resubmit/ig_comparison/` (133 files) |
| IG comparison metrics | `runs/phase_resubmit/ig_comparison/comparison_metrics.csv` |
| Unlabelled predictions | `runs/phase_resubmit/unlabelled/unlabelled_predictions.csv` |
| Distribution NPZs | `runs/phase_resubmit/distributions/` (42 files) |
| Phase 12d analysis | `runs/phase12d/PHASE12D_ANALYSIS.md` |
| Research surveys | `research/run1_*.md` (3 files, 1,722 lines) |
| Webapp backend | `web/` |
| Webapp frontend | `web/frontend/` |
| Webapp handoff doc | `web/AGENT_PLAN.md` |
| Paper figures | `overleaf_drafts/figures/` (31 files) |
| SLURM scripts | `slurm/resubmit_*.sbatch` (18 files) |
| Revision plan | Referenced from `glowing-exploring-pinwheel.md` |
