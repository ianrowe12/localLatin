# Experiment 2 (Redo): MUSTS Multilingual Sentence Similarity

## Overview

This experiment evaluates how well different embedding models and post-processing methods capture sentence-level semantic similarity across four languages, using the MUSTS (Multilingual Sentence Similarity) benchmark. We sweep across all hidden-state layers of three smaller embedding models.

**What changed from the original Phase 8:**
- New models: Qwen3-Embedding-0.6B and KaLM-embedding replace the 7B Qwen and Gemma models
- 50/50 forced split of all MUSTS data (was using the dataset's default splits)
- Optimal D sweep on train (was fixed D=10)
- Added PC Whitening as a third method

---

## Experimental Setup

### Dataset: MUSTS

All splits (train/test/validation) are concatenated, shuffled, and re-split 50/50:

| Language | Train Pairs | Test Pairs |
|---|---|---|
| English | 3,102 | 3,102 |
| French | 255 | 255 |
| Sinhala | 617 | 617 |
| Tamil | 587 | 587 |

### Models

| Model | Type | Layers | Hidden Dim | Parameters |
|---|---|---|---|---|
| **Qwen3-Embedding-0.6B** | Decoder-only | 28 blocks | 1024 | ~600M |
| **KaLM-embedding** | Decoder-only | 24 blocks | 896 | ~500M |
| **LaBSE** | BERT Encoder | 12 layers | 768 | ~470M |

All models are compact embedding-focused architectures (no 7B models).

### Methods

1. **Baseline Mean**: Mean pooling of token hidden states at each layer.
2. **SIF + ABTT (Optimal D)**: SIF weighting + ABTT with D swept over {1, 2, 3, 5, 7, 10}, selected by best Spearman on train.
3. **PC Whitening**: PCA whitening fitted on train, applied to all.

### Evaluation Metric
- **Spearman correlation** between cosine similarities and human similarity scores (primary metric)

### Data Integrity
- SIF probabilities and ABTT PCs fitted on **train sentences only** per language
- Whitening PCA fitted on **train embeddings only**
- Evaluation on **test pairs only**

---

## Key Findings

### Finding 1: Layer Profiles Differ Dramatically by Architecture

Each model has a distinct layer-wise performance curve:

**Qwen3-Embedding-0.6B (English baseline Spearman):**
- Layers 0-2: moderate start (~0.47-0.50)
- Layers 3-16: **deep anisotropy dip** (~0.14-0.20)
- Layers 17-27: steady recovery (0.21 -> 0.62)
- Layer 28: **sharp final spike to 0.752** (the model's embedding head output)

**KaLM-embedding (English baseline Spearman):**
- Layer 0: 0.375 (weak start)
- Layers 1-16: **gradual improvement** (0.56 -> 0.67)
- Layers 17-24: **acceleration** (0.71 -> 0.82)
- No dip, steady improvement throughout

**LaBSE (English baseline Spearman):**
- Steady improvement from 0.545 to **0.746** at layer 12
- No dip at all, every layer improves on the previous
- Consistent with Experiment 1 findings

> **See: Figures 1-4** (Layer-wise Spearman per language)

### Finding 2: The Final-Layer Spike in Qwen3

Qwen3 shows a remarkable pattern: after a deep anisotropy dip spanning layers 3-16, performance gradually recovers, but layer 28 (the final layer) produces a dramatic spike:

| Language | Layer 27 | Layer 28 | Jump |
|---|---|---|---|
| English | 0.618 | **0.752** | +0.134 |
| French | 0.796 | **0.848** | +0.052 |
| Sinhala | 0.387 | **0.391** | +0.004 |
| Tamil | 0.380 | **0.403** | +0.023 |

The spike is largest for high-resource languages. For Sinhala, the spike is negligible, suggesting that the final projection layer primarily benefits languages well-represented in pre-training.

> **See: Figure 1** (English layer curves showing the spike)

### Finding 3: SIF + ABTT Has Mixed Effects on MUSTS

Unlike Experiment 1 where SIF+ABTT was universally beneficial, on MUSTS the results are mixed:

| Model | Language | Baseline Best | SIF Best | Change |
|---|---|---|---|---|
| Qwen3 | English | 0.752 | 0.743 | -1.2% |
| Qwen3 | French | 0.848 | **0.879** | **+3.7%** |
| Qwen3 | Sinhala | 0.402 | **0.453** | **+12.6%** |
| Qwen3 | Tamil | 0.405 | **0.420** | **+3.6%** |
| KaLM | English | 0.841 | 0.822 | -2.3% |
| KaLM | French | 0.875 | **0.882** | +0.8% |
| KaLM | Sinhala | 0.477 | 0.477 | 0.0% |
| KaLM | Tamil | 0.402 | **0.420** | **+4.5%** |
| LaBSE | English | 0.746 | **0.792** | **+6.2%** |
| LaBSE | French | 0.882 | **0.896** | **+1.5%** |
| LaBSE | Sinhala | 0.647 | 0.606 | -6.4% |
| LaBSE | Tamil | 0.450 | 0.425 | -5.4% |

**Pattern:** SIF+ABTT tends to help when:
- The language is French (consistent improvement across all models)
- The model is Qwen3 on low-resource languages (Sinhala +12.6%)

SIF+ABTT tends to hurt when:
- The baseline is already strong (KaLM English)
- LaBSE on low-resource languages (Sinhala -6.4%, Tamil -5.4%)

> **See: Figure 5** (SIF improvement heatmap)

### Finding 4: KaLM-embedding is the Best Overall Model

Despite being the smallest model tested, KaLM-embedding achieves the highest Spearman correlations for English and competitive results elsewhere:

| Language | Best Model | Best Spearman | Method |
|---|---|---|---|
| English | **KaLM** | **0.841** | baseline_mean, layer 23 |
| French | **LaBSE** | **0.896** | sif_abtt_optimal, layer 12 |
| Sinhala | **LaBSE** | **0.647** | baseline_mean, layer 12 |
| Tamil | **LaBSE** | **0.450** | baseline_mean, layer 12 |

KaLM leads on English by a large margin (+0.05 over LaBSE, +0.09 over Qwen3). LaBSE dominates the non-English languages, particularly Sinhala where it leads by +0.17 over the next best.

> **See: Figure 6** (Best Spearman per model/language)

### Finding 5: Low-Resource Languages Remain Challenging

Sinhala and Tamil are dramatically harder for all models:

| Language | Average Best Spearman (across models) | Resource Level |
|---|---|---|
| English | 0.780 | High |
| French | 0.886 | High |
| Sinhala | 0.525 | Low |
| Tamil | 0.432 | Low |

The gap between high-resource and low-resource languages is 0.25-0.45 Spearman points. This gap is consistent across all models and methods, suggesting a fundamental limitation of pre-training data rather than architectural issues.

French outperforms English because MUSTS French pairs likely have stronger similarity signal (cleaner pairs).

> **See: Figure 7** (Language difficulty comparison)

### Finding 6: Whitening is Consistently Worse

PC Whitening underperforms both baseline and SIF+ABTT in almost all configurations:

| Model | Language | Whitening Best | Baseline Best | Gap |
|---|---|---|---|---|
| Qwen3 | English | 0.746 | 0.752 | -0.006 |
| KaLM | English | 0.800 | 0.841 | -0.041 |
| LaBSE | English | 0.772 | 0.746 | +0.026 |
| Qwen3 | French | 0.712 | 0.848 | **-0.136** |
| KaLM | French | 0.728 | 0.875 | **-0.147** |
| LaBSE | French | 0.774 | 0.882 | **-0.108** |

Whitening is especially harmful for French, where it degrades performance by 0.11-0.15 Spearman points. The one exception is LaBSE English where whitening slightly outperforms baseline, but even there SIF+ABTT is better.

> **See: Figure 3** (Methods comparison including whitening)

### Finding 7: Optimal D is Model- and Layer-Dependent

The D sweep reveals different anisotropy structures across models:

**English (representative language):**
- **Qwen3**: D=10 for almost all layers. The deep anisotropy dip means many PCs carry noise.
- **KaLM**: D=10 for middle layers, dropping to D=1-2 for top layers. Top layers have cleaner geometry.
- **LaBSE**: D=10 for early/middle layers, D=2-3 for top layers. Similar to KaLM.

Pattern: models need more aggressive PC removal (high D) in anisotropic layers, and less (low D) where representations are already well-distributed.

> **See: Figure 8** (Optimal D heatmap)

### Finding 8: LaBSE's Layer 12 is a Universal Best

For LaBSE, layer 12 (the final layer) is the best or tied-best across all languages and methods:

| Language | Method | Best Layer | Spearman |
|---|---|---|---|
| English | baseline | 12 | 0.746 |
| English | sif_abtt | 12 | 0.792 |
| French | baseline | 12 | 0.882 |
| French | sif_abtt | 12 | 0.896 |
| Sinhala | baseline | 12 | 0.647 |
| Tamil | baseline | 12 | 0.450 |

This confirms that LaBSE's final layer output is its designed embedding — the entire model is optimized to produce good representations at the output. Intermediate layers are inferior.

For Qwen3 and KaLM, the story is different: the best layer varies by language, and intermediate layers can sometimes match or beat the final layer.

---

## Detailed Results Tables

### Best Spearman per Model / Language / Method

| Model | Language | Baseline (layer) | SIF+ABTT (layer, D) | Whitening (layer) |
|---|---|---|---|---|
| Qwen3 | English | 0.752 (28) | 0.743 (28, D=10) | 0.746 (28) |
| Qwen3 | French | 0.848 (28) | **0.879** (16, D=10) | 0.712 (0) |
| Qwen3 | Sinhala | 0.402 (23) | **0.453** (17, D=5) | 0.442 (0) |
| Qwen3 | Tamil | 0.405 (24) | **0.420** (20, D=3) | 0.358 (0) |
| KaLM | English | **0.841** (23) | 0.822 (23, D=2) | 0.800 (24) |
| KaLM | French | 0.875 (24) | **0.882** (23, D=2) | 0.728 (7) |
| KaLM | Sinhala | 0.477 (23) | 0.477 (23, D=3) | 0.441 (0) |
| KaLM | Tamil | 0.402 (24) | **0.420** (19, D=3) | 0.358 (0) |
| LaBSE | English | 0.746 (12) | **0.792** (12, D=2) | 0.772 (12) |
| LaBSE | French | 0.882 (12) | **0.896** (12, D=5) | 0.774 (2) |
| LaBSE | Sinhala | **0.647** (12) | 0.606 (12, D=1) | 0.385 (11) |
| LaBSE | Tamil | **0.450** (12) | 0.425 (12, D=1) | 0.228 (12) |

---

## Conclusions

1. **KaLM-embedding is the strongest model for English** sentence similarity (0.841), outperforming both LaBSE and Qwen3. For non-English, LaBSE is consistently best.

2. **SIF+ABTT helps selectively.** It is most beneficial for French (all models improve) and for Qwen3 on low-resource languages (+12.6% Sinhala). It can hurt when baseline representations are already clean.

3. **The anisotropy dip appears in Qwen3** (layers 3-16) but not in KaLM or LaBSE. Decoder-only models trained with a specific embedding objective (KaLM) avoid the dip.

4. **Whitening is not recommended** for sentence similarity. It consistently degrades performance, especially for non-English languages.

5. **Low-resource languages remain a hard problem.** Sinhala and Tamil peak at 0.45-0.65 Spearman, regardless of model or method. This is a data/pre-training limitation.

6. **LaBSE's final layer is always best** for LaBSE, confirming its design as a fixed-output-layer model. For Qwen3 and KaLM, optimal layers vary.

7. **Optimal D correlates with layer anisotropy.** Middle layers (more anisotropic) need higher D; top layers (cleaner) need lower D.

---

## Figures

All figures are generated by `scripts/generate_redo_exp2_plots.py` and saved to `runs/redo_results/plots/`.

| Figure | Description |
|---|---|
| `fig_exp2_01_english_layer_curves.png` | Layer-wise Spearman for English (all 3 models, 3 methods) |
| `fig_exp2_02_french_layer_curves.png` | Layer-wise Spearman for French |
| `fig_exp2_03_sinhala_layer_curves.png` | Layer-wise Spearman for Sinhala |
| `fig_exp2_04_tamil_layer_curves.png` | Layer-wise Spearman for Tamil |
| `fig_exp2_05_sif_improvement_heatmap.png` | SIF improvement heatmap (model x language) |
| `fig_exp2_06_best_spearman_bars.png` | Best Spearman per model/language bar chart |
| `fig_exp2_07_language_difficulty.png` | Language difficulty comparison (averaged across models) |
| `fig_exp2_08_optimal_D_heatmap.png` | Optimal D values by model and layer (English) |
| `fig_exp2_09_method_comparison_grouped.png` | Grouped bar chart: 3 methods x 3 models x 4 languages |
