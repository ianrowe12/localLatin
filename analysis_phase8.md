# Phase 8: Scientific Validation Results

## What This Phase Does

Phase 7 showed that SIF weighting + principal component removal (ABTT) dramatically improves Latin text retrieval. But Dr. Siddique identified a critical flaw: **data leakage**. We were computing statistics (token probabilities, mean vectors, principal components) on the *same data* we then evaluated on. This is like studying the answer key before taking a test — your score looks great, but it doesn't prove you actually learned anything.

Phase 8 fixes this by introducing a strict train/test split. All statistics are computed from training data only, and evaluation happens on held-out test queries. If the results still hold, we know the improvement is real.

---

## The Train/Test Split

**1,278 files** across **538 folders** were split as follows:

| Category | Folders | Files | Assignment | Rationale |
|----------|---------|-------|------------|-----------|
| Singletons (1 file) | 279 | 279 | Train | They can only form negative pairs, no retrieval value |
| Pairs (2 files) | 74 | 148 | Test | The only testable positive pair; splitting them would make one un-queryable |
| Large (3+ files) | 185 | 851 | Proportional (~80/20) | At least 2 train per folder to maintain intra-folder signal |

**Result**: 943 train files, 335 test query files, zero overlap, every test query has at least one same-folder partner.

**Evaluation protocol**: Test queries search against ALL 1,278 files (the full gallery). The split only controls which files' statistics are used for fitting.

---

## Models Tested

| Model | Type | Layers | Trained on |
|-------|------|--------|------------|
| **LaTa** (bowphs/LaTa) | T5 Seq2Seq encoder | 12 | Latin texts |
| **PhilTa** (bowphs/PhilTa) | T5 Seq2Seq encoder | 12 | Philosophical Latin texts |
| **LaBSE** | BERT encoder | 12 | 109 languages (NOT specifically Latin) |

---

## Key Finding 1: The Middle-Layer Dip is Real (and ABTT Still Fixes It)

**See: `fig1_middle_layer_dip.png`**

This is the central result. For LaTa and PhilTa, if you just extract hidden states with mean pooling (gray line), performance **collapses** in the middle layers:

### LaTa Hidden States (mean pooling baseline)
| Layer | Acc@1 | What's happening |
|-------|-------|------------------|
| L0 | **85.7%** | Embedding layer — just token lookups, decent starting point |
| L1 | 77.3% | Starting to degrade |
| L2 | 20.9% | Severe collapse begins |
| L3-L11 | **6.6%-9.3%** | Near-random performance. The model "knows" something, but you can't access it with cosine similarity |
| L12 | 82.1% | Final layer recovers — the model's output representation is usable again |

**Why does this happen?** The embeddings become *anisotropic* — all vectors point in nearly the same direction. When every embedding is similar to every other embedding, cosine similarity can't distinguish same-folder pairs from different-folder pairs.

**The evidence** (`fig3_anisotropy.png`): The "off-diagonal mean" measures how similar random pairs are. At L0 it's 0.67. By L3 it's **0.92** — meaning random documents have 92% cosine similarity. There's simply no room left to distinguish same-folder from different-folder.

### SIF + ABTT Fixes This Completely

With SIF-weighted pooling and ABTT (removing the top 10 principal components), performance is **flat and high across all layers**:

| Layer | Baseline (mean) | SIF + ABTT | Improvement |
|-------|----------------|------------|-------------|
| L0 | 85.7% | **94.0%** | +8.3pp |
| L6 | 6.6% | **93.4%** | +86.8pp |
| L8 | 7.2% | **93.7%** | +86.5pp |
| L12 | 82.1% | **93.1%** | +11.0pp |

After ABTT, the off-diagonal mean drops to essentially **0.000** across all layers. The anisotropy is completely removed, and the underlying retrieval signal is exposed.

**This result holds under the clean protocol** — no data leakage. P(w) was computed from train-only texts, and the principal components were fitted on train-only embeddings.

---

## Key Finding 2: SIF Pooling Alone Partially Fixes the Dip

**See: `fig1_middle_layer_dip.png` (green line)**

SIF pooling (weighting tokens by inverse frequency) without any PC removal gives a partial fix:

- LaTa L0: 85.7% -> 93.4% (SIF alone)
- LaTa L6: 6.6% -> 61.2% (SIF alone) — much better than random, but still far from 93%
- LaTa L12: 82.1% -> 91.0% (SIF alone)

SIF helps because it downweights common tokens that contribute most to the anisotropic direction. But it doesn't fully eliminate the problem — you need ABTT for that.

---

## Key Finding 3: FF1 Representations Show Even More Dramatic Dip

**See: `fig2_ff1_representations.png`**

FF1 (the feed-forward network's intermediate activation, captured just before the down-projection) shows the same pattern but more pronounced:

- LaTa FF1 baseline (mean): best is L1 at 63.6%, most layers are 9-31%
- PhilTa FF1 baseline (mean): best is L2 at 51.0%, L4 drops to **4.2%**
- With SIF+ABTT: consistently 90-94% across all layers

The FF1 space is 4x larger (3072 dims vs 768 for hidden states), so there's even more room for anisotropy to develop — and it does.

---

## Key Finding 4: LaBSE Behaves Differently (No Dip)

**See: `fig4_labse.png`**

LaBSE shows a fundamentally different layer-wise pattern:

| Layer | Baseline | Pattern |
|-------|----------|---------|
| L0 | 61.2% | Low starting point |
| L2 | 49.0% | Slight early dip |
| L9 | 75.2% | Steady climb |
| L12 | **88.1%** | Best at the final layer |

There is **no catastrophic middle-layer collapse**. Performance generally increases monotonically toward the final layer. This makes sense — LaBSE is trained explicitly for cross-lingual sentence similarity, so its intermediate representations are better-conditioned.

However, ABTT still helps LaBSE:
- Baseline best: 88.1% (L12)
- SIF+ABTT best: **92.2%** (L0 or L11)

ABTT gives LaBSE a +4pp boost and makes performance more uniform across layers.

**LaBSE FFN Intermediate** (`fig4_labse.png`, right panel): Similar to hidden states but noisier. Performance climbs from ~44% at L0 to ~85% at L11. ABTT brings it to ~91%.

---

## Key Finding 5: The Gap Metric Tells the Same Story

**See: `fig6_gap.png`**

The "gap" is `avg_similarity(same-folder pairs) - avg_similarity(different-folder pairs)`. A bigger gap means the model is better at distinguishing related from unrelated documents.

- **Baseline**: Gap collapses to near zero in middle layers (LaTa: 0.02 at L3-L11), confirming the embeddings lose all discriminative power.
- **After ABTT**: Gap is uniformly ~0.59 across all layers. The signal was always there — ABTT just uncovers it.
- **LaBSE**: Gap grows with depth (0.03 at L2 to 0.18 at L12 for baseline), consistent with no dip.

---

## Key Finding 6: Best Achievable Results Per Model

**See: `fig5_abtt_lift.png`**

| Model | Best Baseline | Best ABTT | Biggest Lift |
|-------|--------------|-----------|-------------|
| LaTa | 93.4% (L0 hidden/sif) | **94.0%** (L0 hidden/mean+ABTT) | +28.1pp on ff1/mean |
| PhilTa | 92.2% (L0 hidden/mean) | **93.1%** (L1 hidden/mean+ABTT) | +40.9pp on ff1/mean |
| LaBSE | 88.1% (L12 hidden/mean) | **92.2%** (L0 hidden/mean+ABTT) | +7.8pp on ffn_int/sif |

**Takeaway**: When you already pick the best layer AND use SIF pooling, the additional gain from ABTT is modest (+0.6-1.2pp). The massive gains (+28-41pp) come when rescuing "useless" layers, particularly FF1 with mean pooling.

---

## Key Finding 7: LaTa/PhilTa vs LaBSE on Latin

**See: `fig7_all_models.png`**

The Latin-specific models (LaTa, PhilTa) outperform the multilingual model (LaBSE) when using SIF+ABTT:

- LaTa SIF+ABTT: 92.5-94.0% consistently across all layers
- PhilTa SIF+ABTT: 91.9-93.1% consistently across all layers
- LaBSE SIF+ABTT: 85.4-92.2%, more variable across layers

LaBSE's lower ceiling makes sense — Latin is one of 109 languages it was trained on, with a much smaller share of training data. But ABTT still gets it within ~2pp of the Latin specialists at the best layer.

---

## Does the Data Leakage Fix Change the Conclusions?

**Yes and no.** The qualitative story is identical to Phase 7:
1. Middle layers suffer from anisotropy
2. SIF pooling partially fixes it
3. ABTT completely fixes it
4. The combination is robust across layers

The numbers are slightly different because we're now evaluating on held-out queries rather than the full dataset. But the effects are just as strong — ABTT still lifts middle layers from ~7% to ~93%. This confirms the findings are **not artifacts of data leakage**.

---

## What's Left (MUSTS Layer-Wise Sweep)

The canon sweep above evaluates retrieval (finding same-author Latin documents). The next step is the MUSTS sweep, which evaluates sentence similarity across four languages (English, French, Sinhala, Tamil) using Spearman correlation. This extends the findings beyond Latin and tests whether:

1. The optimal number of PCs to remove (D) varies by language
2. Low-resource languages (Sinhala, Tamil) need smaller D values
3. Katie's methods (PCA whitening, variance filtering) compete with SIF+ABTT

Submit these jobs to run the MUSTS experiments:
```bash
sbatch slurm/phase8_musts_labse.sbatch    # LaBSE, 4 langs, 12 layers (~4h)
sbatch slurm/phase8_musts_qwen.sbatch     # Qwen2-7B, 4 langs, 28 layers (~12h)
sbatch slurm/phase8_musts_gemma.sbatch    # Gemma-7B, 4 langs, 28 layers (~12h)
```

---

## Figures Index

All plots are in `runs/phase8_results/figures/`:

| File | What it shows |
|------|--------------|
| `fig1_middle_layer_dip.png` | The core result: hidden-state performance collapse and rescue (LaTa, PhilTa) |
| `fig2_ff1_representations.png` | Same story for FF1 (feed-forward intermediate activations) |
| `fig3_anisotropy.png` | Off-diagonal cosine similarity — explains WHY the dip happens |
| `fig4_labse.png` | LaBSE layer-wise results (hidden + FFN intermediate) — no dip, monotonic climb |
| `fig5_abtt_lift.png` | Bar chart: how much ABTT improves over baseline for each model/representation |
| `fig6_gap.png` | Cosine gap (same-folder vs different-folder similarity) across layers |
| `fig7_all_models.png` | All three models on one plot: baseline vs SIF+ABTT comparison |

---

## Data Files

| File | Description |
|------|-------------|
| `runs/phase8_results/meta_split.csv` | The locked train/test split (1,278 rows) |
| `runs/phase8_results/split_summary.json` | Split statistics |
| `runs/phase8_results/phase8_canon_sweep.csv` | Full results table (300 rows, 20 columns) |
