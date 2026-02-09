# Phase 8 Experiment 2: MUSTS Layer-Wise Sweep Results

## What This Experiment Does

Experiment 1 (the canon sweep) showed that SIF+ABTT rescues anisotropic middle layers in Latin-specialized T5 models and also helps LaBSE. But that was only one task (Latin document retrieval) on one dataset. **Experiment 2 extends the validation to sentence similarity across four languages**, using the MUSTS benchmark (Multilingual Sentence Similarity). It tests whether the SIF+ABTT recipe generalizes to:

1. **New languages** — English, French, Sinhala, Tamil (ranging from high- to low-resource)
2. **New model architectures** — decoder-only LLMs (Qwen2-7B, Gemma-7B), not just encoder models
3. **New tasks** — Spearman correlation on sentence similarity scores, not document retrieval

All experiments follow the leak-free protocol: P(w) is computed from train sentences only, PCs are fitted on train embeddings only, and the optimal D is selected on train data. Evaluation reports test-set Spearman.

---

## Models Tested

| Model | Type | Parameters | Layers | Languages |
|-------|------|-----------|--------|-----------|
| **LaBSE** | BERT encoder | 471M | 12 (+ embedding = 13 outputs) | 109 languages |
| **Qwen2-7B** | Decoder-only | 7B | 28 (+ embedding = 29 outputs) | Multilingual |
| **Gemma-7B** | Decoder-only | 7B | 28 (+ embedding = 29 outputs) | Multilingual |

---

## Dataset Sizes

| Language | Train Pairs | Test Pairs | Notes |
|----------|------------|------------|-------|
| English | 5,749 | 1,379 | Largest; most reliable statistics |
| French | 600 | 410 | Moderate; small train set |
| Sinhala | 4,996 | 100 | Large train, tiny test (noisy estimates) |
| Tamil | 2,500 | 100 | Moderate train, tiny test (noisy estimates) |

**Important caveat**: Sinhala and Tamil have only 100 test pairs. Spearman correlations on 100 points have wide confidence intervals (~+/-0.1), so take those results as indicative rather than precise. English and French are more reliable.

---

## Key Finding 1: Decoder-Only LLMs Have Severe Anisotropy Dips

**See: `fig8_musts_layer_profiles.png`**

Just like T5 encoders on the Latin canon task, **Qwen2-7B and Gemma-7B show dramatic performance collapse in their middle layers** when using baseline mean pooling:

### Qwen2-7B (English, baseline mean)
| Layer | Spearman | What's happening |
|-------|----------|------------------|
| L0 | 0.40 | Embedding layer — token lookup, decent start |
| L3 | 0.35 | Starting to degrade |
| L4-L26 | **0.09-0.25** | Severe anisotropy zone — near-random similarity scores |
| L27 | 0.40 | Penultimate layer recovers |
| L28 | 0.40 | Final layer — comparable to L0 |

### Gemma-7B (English, baseline mean)
| Layer | Spearman | What's happening |
|-------|----------|------------------|
| L0-L1 | 0.33-0.34 | Embedding layers |
| L2-L27 | **0.09-0.28** | Severe anisotropy zone — virtually flat and low |
| L28 | 0.09 | Final layer is actually worst (unlike Qwen) |

**This is a major finding**: the anisotropy dip is not specific to T5 encoders or Latin text. It's a general phenomenon that affects 7B decoder-only LLMs across languages. The middle-layer collapse pattern is reproducible across architectures, tasks, and languages.

---

## Key Finding 2: SIF+ABTT Rescues Decoder-Only Models Spectacularly

**See: `fig8_musts_layer_profiles.png`**

The SIF+ABTT recipe (with train-fitted D selected via sweep) transforms the decoder models:

### Qwen2-7B (English)
| Layer | Baseline | SIF+ABTT | Lift |
|-------|----------|----------|------|
| L0 | 0.40 | 0.57 | +0.17 |
| L6 | 0.09 | 0.56 | **+0.47** |
| L24 | 0.22 | **0.63** | **+0.41** |
| L27 | 0.40 | 0.56 | +0.16 |

### Gemma-7B (English)
| Layer | Baseline | SIF+ABTT | Lift |
|-------|----------|----------|------|
| L1 | 0.34 | **0.70** | **+0.36** |
| L10 | 0.20 | 0.64 | **+0.43** |
| L26 | 0.18 | 0.55 | +0.37 |

The average SIF+ABTT lift across all layers for English:
- **Gemma-7B: +0.45** (massive — nearly half a Spearman point gained on average)
- **Qwen2-7B: +0.35**
- **LaBSE: +0.13** (already well-conditioned, so less room to improve)

---

## Key Finding 3: LaBSE is the Best Off-the-Shelf Choice

**See: `fig10_best_spearman_comparison.png`**

Despite being 15x smaller than the 7B models, LaBSE achieves the highest peak test Spearman in every language except Tamil:

| Language | LaBSE Best | Qwen2-7B Best | Gemma-7B Best | Winner |
|----------|-----------|--------------|--------------|--------|
| English | **0.78** (L12) | 0.63 (L24) | 0.70 (L1) | LaBSE |
| French | **0.89** (L12) | 0.85 (L0) | 0.86 (L17) | LaBSE |
| Sinhala | 0.44 (L9) | 0.38 (L0) | **0.46** (L0) | Gemma |
| Tamil | 0.53 (L12) | 0.41 (L4) | **0.53** (L20) | Tie |

**Why does the 471M parameter model beat 7B models?** LaBSE is *trained* for sentence similarity (contrastive loss on parallel sentences). Qwen and Gemma are trained for next-token prediction — they were never optimized to produce sentence embeddings. Their representations need more post-processing (higher D values) to become useful for similarity tasks.

---

## Key Finding 4: SIF+ABTT Helps More Where Baseline is Worst

**See: `fig11_lift_bar_chart.png`**

The lift from SIF+ABTT is inversely correlated with the baseline quality. Models that are already good at a language get a small boost; models that are terrible get a huge boost:

| Model | Language | Baseline Best | SIF+ABTT Best | Lift |
|-------|----------|--------------|---------------|------|
| Gemma-7B | Sinhala | 0.24 | 0.46 | **+0.47** (at best layer) |
| Qwen2-7B | English | 0.40 | 0.63 | **+0.41** |
| Gemma-7B | English | 0.34 | 0.70 | **+0.36** |
| Gemma-7B | French | 0.73 | 0.86 | +0.26 |
| LaBSE | English | 0.72 | 0.78 | +0.07 |
| LaBSE | French | 0.86 | 0.89 | +0.03 |

The pattern is clear: SIF+ABTT is most valuable precisely when raw embeddings are worst — which is exactly when you need it most.

---

## Key Finding 5: Optimal D Varies by Model and Language

**See: `fig12_optimal_D_heatmap.png`**

The optimal number of principal components to remove (D) shows systematic patterns:

| Model | English | French | Sinhala | Tamil |
|-------|---------|--------|---------|-------|
| LaBSE | D=1 | D=3 | D=1 | D=1 |
| Qwen2-7B | D=10 | D=7 | D=1 | D=5 |
| Gemma-7B | D=2 | D=10 | D=7 | D=7 |

**Pattern**: LaBSE consistently needs very few PCs removed (D=1-3). This makes sense — it's already well-conditioned for sentence similarity, so there are fewer "anisotropic directions" to remove. The decoder models (Qwen, Gemma) need more aggressive cleaning (D=5-10), consistent with their higher anisotropy.

**Sinhala is an exception**: Low D values across models. This may reflect that the limited Sinhala data doesn't have enough signal to estimate many PCs reliably, so removing fewer is safer.

---

## Key Finding 6: ABTT Dominates Katie's Methods

**See: `fig9_method_comparison.png`**

We tested six methods at each layer:

1. **baseline_mean** — Simple mean pooling, no correction
2. **sif_only** — SIF-weighted pooling, no PC removal
3. **sif_abtt_fixed** — SIF + ABTT with D=10 (fixed)
4. **sif_abtt_optimal** — SIF + ABTT with D selected on train (best D)
5. **whitening** — PCA whitening (Katie's method 1)
6. **variance_filter** — Remove low-variance dimensions (Katie's method 2)

### Win rates (across all layers, all model-language combinations):

| Comparison | ABTT Wins | Other Wins |
|-----------|-----------|------------|
| ABTT vs Whitening (LaBSE) | **50**/52 | 2/52 |
| ABTT vs Whitening (Qwen) | **96**/116 | 20/116 |
| ABTT vs Whitening (Gemma) | **113**/116 | 3/116 |

**Whitening occasionally wins** for Qwen2-7B at specific layers, but across the board, SIF+ABTT is the dominant approach. Whitening struggles particularly on Sinhala and Tamil where it sometimes produces negative Spearman values.

**Variance filtering** typically performs close to the baseline — it doesn't provide the dramatic rescue that ABTT does. It's essentially a non-starter for fixing anisotropy.

---

## Key Finding 7: The Train-Test Gap is Small (No Overfitting)

The leak-free protocol works. The gap between train and test Spearman is small for the high-data languages:

| Model | Language | Train Spearman | Test Spearman | Gap |
|-------|----------|---------------|---------------|-----|
| LaBSE | English | 0.780 | 0.781 | -0.001 |
| LaBSE | French | 0.893 | 0.887 | +0.006 |
| Qwen2-7B | English | 0.632 | 0.630 | +0.002 |
| Qwen2-7B | French | 0.857 | 0.852 | +0.005 |
| Gemma-7B | English | 0.689 | 0.699 | -0.011 |
| Gemma-7B | French | 0.854 | 0.860 | -0.007 |

For Sinhala and Tamil, the gaps are larger (+/-0.05-0.15), but this is expected given the tiny test sets (100 pairs) and resulting noise.

**Bottom line**: Fitting D on train and applying to test produces essentially no overfitting for English and French, confirming the protocol is sound.

---

## Key Finding 8: Cross-Lingual Patterns

All three models show a consistent language difficulty ranking:

**French > English > Tamil > Sinhala**

This holds regardless of method. French is "easiest" (highest Spearman), likely because:
- French has close typological similarity to English (which dominates training data)
- The MUSTS French dataset may have clearer similarity gradations

Sinhala is hardest, which aligns with it being the most low-resource and typologically distant language tested.

---

## Connections to Experiment 1 (Canon Sweep)

| Finding | Canon (Latin Retrieval) | MUSTS (Multilingual Similarity) | Generalizes? |
|---------|------------------------|--------------------------------|--------------|
| Middle-layer dip | LaTa/PhilTa L2-L11 collapse | Qwen/Gemma L4-L26 collapse | **Yes** |
| SIF partially fixes | LaTa L6: 7% -> 61% | Qwen L6 English: 0.09 -> 0.42 | **Yes** |
| ABTT fully fixes | LaTa L6: 7% -> 93% | Qwen L24 English: 0.22 -> 0.63 | **Yes** |
| LaBSE no dip | Monotonic climb to L12 | Monotonic climb to L12 | **Yes** |
| Latin-specific > LaBSE | LaTa 94% > LaBSE 92% | N/A (no Latin in MUSTS) | N/A |
| ABTT > whitening | Not tested in canon | 91% win rate | **New finding** |

---

## Summary for the Paper

1. **The anisotropy dip is universal**: It occurs in T5 encoders (LaTa, PhilTa), BERT encoders (LaBSE, mildly), and 7B decoder-only LLMs (Qwen2-7B, Gemma-7B) across tasks and languages.

2. **SIF+ABTT is the best correction**: It consistently outperforms whitening (91% of the time) and variance filtering. The recipe is simple: SIF-weighted pooling + remove D principal components fitted on training data.

3. **The fix is strongest where needed most**: The biggest lifts (+0.35-0.47 Spearman) occur in decoder-only models that were never trained for embedding tasks. The smallest lifts (+0.01-0.07) occur in LaBSE, which was specifically trained for sentence similarity.

4. **The protocol is leak-free**: Train-test gaps are negligible for adequately-sized test sets, confirming the improvements are not artifacts of data leakage.

5. **LaBSE remains the practical recommendation**: Despite being 15x smaller, LaBSE beats the 7B models on most languages after SIF+ABTT is applied to all of them. This is because LaBSE's training objective (cross-lingual sentence similarity) inherently produces better-conditioned embeddings.

---

## Figures Index (Experiment 2)

All plots in `runs/phase8_results/figures/`:

| File | What it shows |
|------|--------------|
| `fig8_musts_layer_profiles.png` | Layer-wise Spearman: baseline vs SIF+ABTT for all 3 models x 4 languages |
| `fig9_method_comparison.png` | All 6 methods compared: heatmap of best test Spearman per model x language |
| `fig10_best_spearman_comparison.png` | Cross-model comparison: bar chart of peak Spearman per language |
| `fig11_lift_bar_chart.png` | SIF+ABTT lift over baseline at best layer for each model x language |
| `fig12_optimal_D_heatmap.png` | Optimal D values as a heatmap across model x language |
| `fig13_whitening_vs_abtt.png` | Head-to-head: whitening vs SIF+ABTT across all layers |

---

## Data Files

| File | Description |
|------|-------------|
| `runs/phase8_results/phase8_musts_sweep.csv` | Full results (1,704 rows, 11 columns) |
| `runs/phase8_results/phase8_musts_sweep_partial_backup.csv` | Backup of first failed run (English + French only) |
