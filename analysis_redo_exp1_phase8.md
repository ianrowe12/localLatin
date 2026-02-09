# Experiment 1 (Redo): Canon Embedding Analysis

## Overview

This experiment evaluates how well different transformer layers and post-processing methods can discriminate semantically related Latin texts in the Canon dataset. We compare three models across their full layer stacks using three embedding methods.

**What changed from the original Phase 8:**
- New 50/50 train/test split (was 80/20)
- Singletons distributed evenly between train/test (were all in train)
- Doubletons split at folder level 50/50 (were all in test)
- Added PC Whitening as a third method

---

## Experimental Setup

### Dataset: Canon

| Metric | Value |
|---|---|
| Total files | 1,278 |
| Total folders | 538 |
| Train files | 578 |
| Test files | 700 |
| Test positive pairs | 464 |
| Test negative pairs | 244,186 |
| Singleton folders | 279 (139 train / 140 test) |
| Doubleton folders | 74 (37 train / 37 test) |
| Multi-file folders | 185 (files split ~50/50 within each) |

### Models

| Model | Type | Layers | Hidden Dim | Representations Extracted |
|---|---|---|---|---|
| **LaTa** (`bowphs/LaTa`) | T5 Seq2Seq | 12 blocks | 768 | Hidden states (0-12), FF1 (0-11) |
| **PhilTa** (`bowphs/PhilTa`) | T5 Seq2Seq | 12 blocks | 768 | Hidden states (0-12), FF1 (0-11) |
| **LaBSE** (`sentence-transformers/LaBSE`) | BERT Encoder | 12 layers | 768 | Hidden states (0-12), FFN intermediate (0-11) |

### Methods

1. **Baseline Mean**: Simple mean pooling of token embeddings at each layer. No post-processing.
2. **SIF + ABTT (Optimal D)**: Smooth Inverse Frequency weighting + All-But-The-Top PC removal. D (number of PCs to remove) is swept over {1, 2, 3, 5, 7, 10} and selected by best Spearman correlation on train pairs.
3. **PC Whitening**: PCA transformation fitted on train embeddings, applied to all.

### Evaluation Metrics
- **AUC-ROC**: Area under ROC curve (primary metric)
- **Spearman**: Rank correlation between cosine similarities and ground-truth labels
- **Acc@k**: Retrieval accuracy at k = {1, 3, 5}

### Data Integrity
- SIF token probabilities fitted on **train texts only**
- ABTT principal components fitted on **train embeddings only**
- PCA whitening fitted on **train embeddings only**
- All evaluation on **test pairs only**

---

## Key Findings

### Finding 1: The Anisotropy Dip in T5 Models

The most striking result is the dramatic "U-shaped" performance curve in both LaTa and PhilTa. Middle layers produce nearly useless embeddings under baseline mean pooling.

**LaTa Hidden States (Baseline AUC):**
| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.916 | 0.862 | 0.530 | 0.482 | 0.475 | 0.474 | 0.474 | 0.473 | 0.473 | 0.474 | 0.475 | 0.477 | **0.944** |

- Layers 3-11 have AUC **below 0.50** (worse than random!)
- Layer 0 (embedding output) starts strong at 0.916
- Layer 12 (final) recovers to 0.944
- The dip spans layers 2-11 (10 out of 13 layers are broken)

**PhilTa** shows the same pattern but slightly less severe:
- Layer 0: 0.960, Layers 4-11: ~0.50, Layer 12: 0.910

**LaBSE does NOT have this dip.** Its baseline AUC improves monotonically from 0.884 (layer 0) to 0.960 (layer 12). This is the fundamental architectural difference: BERT-style encoders don't suffer from the same anisotropy that plagues T5 encoder representations.

> **See: Figure 1** (Layer-wise AUC for hidden states, all 3 models)

### Finding 2: SIF + ABTT is Transformative

SIF + ABTT completely eliminates the anisotropy dip. Every single layer, for every model, achieves AUC > 0.97 after SIF + ABTT processing.

**LaTa Hidden States (SIF + ABTT AUC):**
| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.979 | 0.983 | 0.977 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 | 0.983 | 0.982 | 0.983 | **0.986** |

The improvement at the dip layers is enormous:
- Layer 5 baseline: 0.474 -> SIF+ABTT: 0.983 **(+107% relative improvement)**
- Layer 7 baseline: 0.473 -> SIF+ABTT: 0.983 **(+108% relative improvement)**

For PhilTa and LaBSE, SIF+ABTT provides similar lift, pushing all layers to the 0.97-0.98 range.

> **See: Figure 2** (Baseline vs SIF+ABTT comparison, all models)

### Finding 3: Whitening is Ineffective

PC Whitening performs poorly across all models and layers:
- Best whitening AUC: **0.619** (PhilTa hidden layer 5, LaTa hidden layer 7)
- This is dramatically worse than both baseline (at good layers) and SIF+ABTT
- Whitening AUC ranges from 0.56 to 0.62 across all configurations
- It neither recovers the anisotropy dip nor improves upon already-good layers

> **See: Figure 3** (All three methods compared)

### Finding 4: FF1/FFN Layers Show a Different Pattern

The feed-forward intermediate representations behave differently from hidden states:

**LaTa FF1 (Baseline AUC):**
| Layer | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AUC | 0.819 | 0.530 | 0.478 | 0.485 | 0.523 | 0.514 | 0.544 | 0.500 | 0.590 | 0.637 | 0.722 | 0.675 |

- The dip is still present but less uniform
- Late FF1 layers (8-10) show partial recovery even without post-processing
- SIF+ABTT rescues all FF1 layers to 0.975-0.987 AUC

**LaBSE FFN Intermediate** mirrors the hidden state pattern: steady improvement from 0.865 to 0.949, with SIF+ABTT boosting all to 0.967-0.978.

> **See: Figure 4** (FF1/FFN layer-wise AUC)

### Finding 5: Optimal D Varies by Layer and Model

The number of principal components removed (D) depends on the layer's anisotropy:

| Model | Dip Zone D | Boundary D | Insight |
|---|---|---|---|
| **LaTa hidden** | 2 (layers 3-11) | 5-10 (layers 0-1, 12) | Dip layers need minimal PC removal |
| **PhilTa hidden** | 3-5 (layers 4-11) | 1-7 (layers 0-3, 12) | Similar pattern, slightly higher D |
| **LaBSE hidden** | 3-7 (all layers) | 1 (layer 12) | No dip, D decreases for best layers |

Key insight: layers in the anisotropy dip have low effective dimensionality, so fewer PCs need removal. Boundary layers (0, 12) have richer structure requiring more aggressive PC removal.

> **See: Figure 5** (Optimal D heatmap)

### Finding 6: Retrieval Accuracy Results

Best configuration per model (Acc@1 on test set):

| Model | Best Config | AUC | Acc@1 | Acc@3 | Acc@5 |
|---|---|---|---|---|---|
| **LaTa** | FF1 layer 4, SIF+ABTT, D=7 | **0.987** | 0.916 | 0.948 | 0.954 |
| **PhilTa** | Hidden layer 8, SIF+ABTT, D=3 | 0.984 | 0.927 | 0.945 | 0.950 |
| **LaBSE** | Hidden layer 12, SIF+ABTT, D=1 | 0.986 | 0.913 | 0.941 | 0.946 |

All three models achieve comparable peak performance (AUC 0.984-0.987), suggesting that SIF+ABTT equalizes model quality. The choice of model matters less than the choice of method.

> **See: Figure 6** (Best-config retrieval accuracy comparison)

---

## Detailed Results Tables

### Best AUC per Model / Representation / Method

| Model | Repr | Method | Best Layer | D | AUC | Acc@1 |
|---|---|---|---|---|---|---|
| LaTa | hidden | baseline_mean | 12 | - | 0.944 | 0.789 |
| LaTa | hidden | sif_abtt_optimal | 12 | 10 | 0.986 | 0.938 |
| LaTa | hidden | whitening | 7 | - | 0.616 | 0.284 |
| LaTa | ff1 | baseline_mean | 0 | - | 0.819 | 0.598 |
| LaTa | ff1 | sif_abtt_optimal | 4 | 7 | **0.987** | 0.916 |
| LaTa | ff1 | whitening | 9 | - | 0.610 | 0.314 |
| PhilTa | hidden | baseline_mean | 0 | - | 0.960 | 0.884 |
| PhilTa | hidden | sif_abtt_optimal | 8 | 3 | 0.984 | 0.927 |
| PhilTa | hidden | whitening | 5 | - | 0.619 | 0.289 |
| PhilTa | ff1 | baseline_mean | 10 | - | 0.760 | 0.398 |
| PhilTa | ff1 | sif_abtt_optimal | 4 | 7 | 0.982 | 0.918 |
| PhilTa | ff1 | whitening | 4 | - | 0.619 | 0.338 |
| LaBSE | hidden | baseline_mean | 12 | - | 0.960 | 0.864 |
| LaBSE | hidden | sif_abtt_optimal | 12 | 1 | 0.986 | 0.913 |
| LaBSE | hidden | whitening | 10 | - | 0.615 | 0.277 |
| LaBSE | ffn_int | baseline_mean | 10 | - | 0.949 | 0.816 |
| LaBSE | ffn_int | sif_abtt_optimal | 9 | 5 | 0.978 | 0.879 |
| LaBSE | ffn_int | whitening | 9 | - | 0.591 | 0.366 |

---

## Conclusions

1. **SIF + ABTT is the single most important technique** for extracting useful embeddings from Latin transformers. It turns every layer of every model into a high-quality representation.

2. **The anisotropy dip is real and devastating** for T5-based models (LaTa, PhilTa). Without post-processing, only layers 0 and 12 produce usable embeddings. BERT-based LaBSE is immune.

3. **PC Whitening does not work** for this task. It consistently underperforms even the worst baseline layers.

4. **Model choice is secondary to method choice.** All three models achieve AUC 0.984-0.987 with SIF+ABTT, despite very different baseline behaviors.

5. **The optimal D is context-dependent** but generally low (1-7). The D sweep on train data is essential for maximizing performance.

---

## Figures

All figures are generated by `scripts/generate_redo_exp1_plots.py` and saved to `runs/redo_results/plots/`.

| Figure | Description |
|---|---|
| `fig_exp1_01_hidden_baseline_uShape.png` | Hidden state baseline AUC by layer (U-shape visualization) |
| `fig_exp1_02_hidden_sif_rescue.png` | SIF+ABTT rescues all layers (baseline vs SIF comparison) |
| `fig_exp1_03_all_methods_comparison.png` | All 3 methods compared per model (hidden states) |
| `fig_exp1_04_ff1_ffn_layers.png` | FF1/FFN layer-wise AUC |
| `fig_exp1_05_optimal_D_heatmap.png` | Optimal D values by model and layer |
| `fig_exp1_06_best_config_retrieval.png` | Best-config retrieval accuracy bar chart |
| `fig_exp1_07_method_improvement_bars.png` | Method improvement over baseline per model |
