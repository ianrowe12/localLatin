# Phase 9 Experiment 1: Semantic Equivalence Detection & Directory Assignment

## What We Asked

Given a corpus of 1,278 Latin manuscript fragments organized into 538 directories (where each directory represents a single original text), can we determine whether a new fragment belongs to an existing text or is something entirely new? Think of it like Stack Overflow duplicate detection: "does this file belong to an existing topic, or is it a new topic?"

We tested three language models (LaTa, PhilTa, LaBSE), extracted representations from every internal layer, and compared five post-processing methods to find the best pipeline for this task.

---

## Experimental Setup

### The Split

We split the 1,278 files into a perfect **50/50 split** (639 train, 639 test):

| Category | Count | Train | Test |
|----------|-------|-------|------|
| **Singletons** (1 file per dir) | 279 dirs | 79 | 200 |
| **Doubletons** (2 files per dir) | 74 dirs | 74 | 74 |
| **Multi-file** (3+ per dir) | 185 dirs | 486 | 365 |
| **Total** | 538 dirs | **639** | **639** |

In the test set:
- **320 files** have at least one same-directory partner in test ("Existing" -- they should be matched to a known topic)
- **319 files** have no same-directory partner in test ("New" -- they should be flagged as novel topics)
- **427 positive pairs** in train, **222 positive pairs** in test

### Leak-Free Design

All fitting (token probabilities for SIF, principal components for ABTT, PCA for whitening, threshold selection) was done **only on the train set**. Test-set evaluation uses test files only -- no file ever sees itself or its train-set partners during evaluation.

### Models Tested

| Model | Type | Layers | What It Is |
|-------|------|--------|------------|
| **LaTa** | Seq2Seq (T5) | 0-12 hidden + 1-12 FF1 | Latin-adapted T5 from bowphs |
| **PhilTa** | Seq2Seq (T5) | 0-12 hidden + 1-12 FF1 | Philological T5 from bowphs |
| **LaBSE** | Encoder (BERT) | 0-12 hidden | Google's multilingual sentence encoder |

### Post-Processing Methods

| Method | What It Does |
|--------|-------------|
| **Baseline** | Mean-pool token embeddings, no post-processing |
| **SIF only** | Weight tokens by inverse frequency before pooling (down-weights common words) |
| **SIF + ABTT (D=10)** | SIF pooling + remove top 10 principal components (reduces anisotropy) |
| **SIF + ABTT (optimal D)** | Same as above, but D is tuned per layer to maximize train AUCROC |
| **PCA Whitening** | Mean-pool + decorrelate and normalize variance via PCA |

---

## Key Findings

### 1. The Anisotropy Dip Is Real and Dramatic

**The single most striking result** (see Figure 1): when you use plain mean-pooled embeddings from the middle layers of LaTa and PhilTa, performance catastrophically collapses.

| Model | Layer 0 AUCROC | Worst Layer | Worst AUCROC | Drop |
|-------|---------------|-------------|-------------|------|
| **LaTa** | 0.930 | L8 | 0.463 | -50% |
| **PhilTa** | 0.952 | L6 | 0.464 | -51% |
| **LaBSE** | 0.910 | L1 | 0.843 | -7% |

For the T5 models, the baseline AUCROC drops to near **random chance** (~0.46) in layers 3-11. This is the "anisotropy dip" -- middle-layer representations become so directionally concentrated that cosine similarity loses all discriminative power. The embeddings all point roughly the same direction, so everything looks equally similar.

LaBSE is far more resistant to this effect. Its worst layer still scores 0.843, and performance actually **improves** through the layers, peaking at L11 (0.948). This makes sense: LaBSE was explicitly trained for semantic similarity, so its representations are better conditioned for this task at every depth.

> **[See Fig 1: `fig1_dip_graph.png`]** — The gray baseline line plummets in middle layers for LaTa/PhilTa while the red SIF+ABTT line stays flat.

### 2. SIF + ABTT Completely Solves the Dip

**This is the headline result.** Applying SIF weighting + ABTT principal component removal makes the anisotropy dip disappear entirely:

| Model | Method | Worst Layer AUCROC | Best Layer AUCROC | Range |
|-------|--------|-------------------|-------------------|-------|
| **LaTa** | SIF+ABTT(opt) | 0.957 (L2) | 0.971 (L3) | 0.014 |
| **PhilTa** | SIF+ABTT(opt) | 0.967 (L12) | 0.980 (L9) | 0.013 |
| **LaBSE** | SIF+ABTT(opt) | 0.955 (L5) | 0.978 (L12) | 0.023 |

With SIF+ABTT, **every single layer of every model scores above 0.955 AUCROC**. The catastrophic 50-point drop becomes a gentle 1-2 point ripple. This means you don't have to cherry-pick the "right" layer anymore -- the post-processing makes the choice nearly irrelevant.

SIF alone (without ABTT) also helps dramatically, raising the floor to ~0.81-0.91 AUCROC, but it doesn't fully neutralize the dip. ABTT is what pushes it over the top.

> **[See Fig 2: `fig2_method_heatmap.png`]** — The heatmap makes the contrast stark: solid green for SIF+ABTT columns, red for whitening.

### 3. Whitening Fails Completely

PCA whitening -- which has been proposed in the literature as a fix for anisotropy -- **does not work** for this task:

| Model | Best Whitening AUCROC | Best SIF+ABTT AUCROC |
|-------|----------------------|---------------------|
| LaTa | 0.589 | 0.971 |
| PhilTa | 0.612 | 0.980 |
| LaBSE | 0.578 | 0.978 |

Whitening never exceeds 0.612 AUCROC on any model at any layer. Even worse, its assignment accuracy reveals the problem: it sets the threshold at tau=0.000, meaning it classifies **everything** as "Existing" (100% existing accuracy, 0% new accuracy, 50% overall). It's essentially flipping a coin. The decorrelation destroys the signal rather than enhancing it.

> **[See Fig 4: `fig4_assignment_accuracy.png`]** — Whitening's bars show the failure mode clearly: 100% existing, 0% new, ~50% overall.

### 4. Best Configurations: PhilTa Wins Overall

The **top 5 configurations** by AUCROC:

| Rank | Model | Layer | Method | D | AUCROC | Acc@1 | Assignment |
|------|-------|-------|--------|---|--------|-------|------------|
| 1 | **PhilTa** | L9 | SIF+ABTT(opt) | 3 | **0.9798** | 0.888 | 0.862 |
| 2 | **LaBSE** | L12 | SIF+ABTT(opt) | 1 | **0.9782** | 0.897 | 0.861 |
| 3 | **PhilTa** | L8 | SIF+ABTT(opt) | 2 | **0.9780** | 0.891 | 0.858 |
| 4 | **PhilTa** | L7 | SIF+ABTT(opt) | 2 | **0.9768** | 0.894 | 0.867 |
| 5 | **PhilTa** | L10 | SIF+ABTT(opt) | 3 | **0.9765** | 0.878 | 0.859 |

PhilTa occupies 4 of the top 5 spots, with its sweet spot at layers 7-10. LaBSE's L12 is essentially tied for the top. LaTa's best (0.971 at L3) is strong but slightly behind.

For **retrieval accuracy** (Acc@1), the best configurations achieve ~90%:
- LaTa: 0.916 (FF1 L4, SIF+ABTT fixed)
- PhilTa: 0.913 (hidden L4, SIF+ABTT fixed)
- LaBSE: 0.909 (hidden L11, SIF+ABTT fixed)

> **[See Fig 8: `fig8_summary_table.png`]** — Full breakdown of best layer, D, AUCROC, Acc@1, and assignment accuracy per model and method.

> **[See Fig 6: `fig6_acc_at_1.png`]** — Retrieval accuracy follows the same dip pattern as AUCROC, and SIF+ABTT flattens it out.

### 5. The Cosine Gap Tells the Story

The "gap" metric (average cosine similarity of same-directory pairs minus different-directory pairs) shows how well-separated the two classes are:

| Method | Mean Gap | Max Gap |
|--------|----------|---------|
| Baseline | 0.061 | 0.336 |
| SIF only | 0.223 | 0.444 |
| **SIF+ABTT (optimal)** | **0.557** | **0.646** |
| SIF+ABTT (D=10) | 0.530 | 0.597 |
| Whitening | 0.109 | 0.301 |

With SIF+ABTT, same-directory pairs average ~0.60 cosine similarity while different-directory pairs average ~0.001. That's a massive 0.6-point gap on a [-1, 1] scale. The threshold histogram shows this visually: the two distributions barely overlap.

> **[See Fig 3: `fig3_cosine_gap.png`]** — The red SIF+ABTT line holds a 0.5-0.6 gap at every layer while baseline collapses to near zero.

> **[See Fig 5: `fig5_threshold_histogram.png`]** — The bimodal distribution for PhilTa L9: same-directory pairs clustered around 0.6-0.8, different-directory pairs centered at 0.0, with a clean threshold at tau=0.508.

### 6. Directory Assignment Works in Practice

The practical question: "Should this new file be assigned to an existing directory or flagged as a new topic?" Results for the best SIF+ABTT configurations:

| Model | Existing Acc | New Acc | Overall |
|-------|-------------|---------|---------|
| LaTa (best) | 0.894 | 0.850 | 0.854 |
| PhilTa (best) | 0.888 | 0.837 | 0.862 |
| LaBSE (L11) | 0.903 | 0.884 | **0.894** |

LaBSE L11 with SIF+ABTT (D=2) achieves the best balanced assignment at **89.4% overall** -- it correctly routes ~90% of "existing" files to their topic and correctly identifies ~88% of "new" files as novel. This is a strong practical result for an automated triaging system.

An interesting asymmetry: **SIF-only** (no ABTT) scores 92% on "New" detection but only 51-74% on "Existing" matching. It's very conservative -- it calls most things "new." ABTT balances this by widening the gap, allowing a more centered threshold.

> **[See Fig 4: `fig4_assignment_accuracy.png`]** — Compare the SIF-only bars (high red/new, low blue/existing) vs SIF+ABTT bars (balanced across both).

### 7. Optimal D Is Small and Model-Dependent

The number of principal components to remove (D) varies by model:

| Model | Typical Optimal D | Pattern |
|-------|------------------|---------|
| **LaTa** | 1-3 | D=1 early layers, D=2-3 later |
| **PhilTa** | 1-3 | D=1 early, D=2-3 mid/late |
| **LaBSE** | 1-10 | High D early (7-10), low D late (1-3) |

The T5 models need very little correction -- just 1-3 components. LaBSE shows a striking pattern: early layers need aggressive correction (D=7-10) while the final layers barely need any (D=1-2). This suggests LaBSE's later layers are already well-conditioned, and the anisotropy is concentrated in fewer dimensions there.

The fixed D=10 setting works well across the board (never more than 0.7% below optimal), so if you don't want to tune, D=10 is a safe default.

> **[See Fig 7: `fig7_optimal_d.png`]** — LaTa/PhilTa show a clean low-D staircase; LaBSE's early layers spike to D=7-10 then taper.

### 8. Hidden States vs. FF1

Both hidden states and FF1 (feed-forward intermediate) representations work well with SIF+ABTT:

| Model | Best Hidden AUCROC | Best FF1 AUCROC |
|-------|-------------------|-----------------|
| LaTa | 0.971 | 0.974 |
| PhilTa | 0.980 | 0.974 |

FF1 is competitive and sometimes slightly better (LaTa FF1 L1 at 0.974 is LaTa's overall best). The FF1 representations don't show the same dramatic dip in the baseline -- they're more stable across layers, though still significantly boosted by SIF+ABTT.

> **[See Fig 1b: `fig1b_dip_ff1.png`]** — FF1 baseline is more stable than hidden-state baseline, but SIF+ABTT still lifts it to the same 0.96+ ceiling.

---

## Summary Table: Best Result per Model

| Model | Best AUCROC | Config | Acc@1 | Assignment |
|-------|------------|--------|-------|------------|
| **PhilTa** | **0.980** | hidden L9, SIF+ABTT(D=3) | 0.888 | 0.862 |
| **LaBSE** | **0.978** | hidden L12, SIF+ABTT(D=1) | 0.897 | 0.861 |
| **LaTa** | **0.974** | FF1 L1, SIF+ABTT(D=10) | 0.900 | 0.872 |

---

## What This Means for the Paper

1. **SIF+ABTT is the recommended pipeline.** It eliminates the anisotropy problem, works across all models and layers, and requires minimal tuning (D=10 as default is fine).

2. **Layer choice barely matters with the right post-processing.** This is a key selling point -- practitioners don't need to do expensive layer selection.

3. **PhilTa and LaBSE are the strongest models**, nearly tied. PhilTa edges ahead on AUCROC; LaBSE wins on balanced assignment accuracy.

4. **Whitening is not a viable alternative** despite its prominence in the representation learning literature. For this domain and task, it destroys the signal.

5. **The system is practically useful**: ~90% accuracy on the "new vs. existing topic" classification task means it could meaningfully assist scholars in organizing manuscript fragments.

---

## Figures Reference

All figures are in `runs/phase9/figures_full/`:

| Figure | File | What It Shows |
|--------|------|--------------|
| **Fig 1** | `fig1_dip_graph.png` | The anisotropy dip: AUCROC vs layer for hidden states |
| **Fig 1b** | `fig1b_dip_ff1.png` | Same for FF1 representations |
| **Fig 2** | `fig2_method_heatmap.png` | Best AUCROC per (model/repr, method) heatmap |
| **Fig 3** | `fig3_cosine_gap.png` | Cosine gap (separability) across layers |
| **Fig 4** | `fig4_assignment_accuracy.png` | Existing vs New classification accuracy by method |
| **Fig 5** | `fig5_threshold_histogram.png` | Similarity distribution with threshold line |
| **Fig 6** | `fig6_acc_at_1.png` | Retrieval accuracy (Acc@1) vs layer |
| **Fig 7** | `fig7_optimal_d.png` | Optimal D chosen per model and layer |
| **Fig 8** | `fig8_summary_table.png` | Best results summary table |

All figures are also available as PDFs for direct paper inclusion.

---

## Raw Data

- Results CSV: `runs/phase9/phase9_experiment1_results.csv` (315 rows, 22 columns)
- Split CSV: `runs/phase9/phase9_split.csv` (1,278 rows)
- Train/Test TSVs: `runs/phase9/train.tsv`, `runs/phase9/test.tsv`
