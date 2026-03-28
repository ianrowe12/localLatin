# Phase 10 Experiment 1: Duplicate Detection & Directory Assignment

## What We Asked

Given a new Latin manuscript fragment, should it be assigned to an existing directory (an already-known text) or flagged as something entirely new? This is the same directory assignment task from Phase 9, now extended with two new multilingual models and a new optimization criterion: **D is tuned to maximize Assignment Accuracy** rather than AUCROC.

We tested five models across three architecture families, five post-processing methods, and every internal layer -- totalling 845 configurations.

---

## Experimental Setup

### The Split

We split the 1,278 files into a **50/50 split** (639 train, 639 test):

| Category | Count |
|----------|-------|
| **Train files** | 639 |
| **Test files** | 639 |
| **"Existing" test files** (has same-directory partner) | 320 |
| **"New" test files** (no same-directory partner) | 319 |

### Leak-Free Design

All fitting (SIF token probabilities, ABTT principal components, PCA whitening, threshold tau) uses **only the train set**. The threshold tau is selected to maximize train-set F1 score over 200 evenly-spaced candidates.

### Models Tested

| Model | Architecture | Layers | Repr Types |
|-------|-------------|--------|------------|
| **LaTa** | T5 (Seq2Seq) | 0-12 | hidden + FF1 |
| **PhilTa** | T5 (Seq2Seq) | 0-12 | hidden + FF1 |
| **LaBSE** | BERT (Encoder) | 0-12 | hidden |
| **Qwen3-0.6B** | Qwen3 (Decoder) | 0-28 | hidden + FFN |
| **KaLM-mini** | Gemma (Decoder) | 0-24 | hidden + FFN |

### Post-Processing Methods

| Method | Description |
|--------|-------------|
| **Baseline** | Mean-pool, no correction |
| **SIF only** | SIF-weighted pooling (down-weight common tokens) |
| **SIF + ABTT (D=10)** | SIF + remove top 10 principal components |
| **SIF + ABTT (optimal D)** | SIF + ABTT with D tuned per layer to maximize train Assignment Accuracy |
| **PCA Whitening** | Mean-pool + PCA decorrelation + variance normalization |

### Key Change from Phase 9

In Phase 9, the optimal D was chosen to maximize AUCROC. In Phase 10, **optimal D maximizes Assignment Accuracy** -- the metric that directly measures the practical "route to existing folder or flag as new" decision. This change better aligns optimization with the downstream use case.

---

## Key Findings

### 1. PhilTa Wins the Assignment Task

Assignment Accuracy is our **primary metric** in Phase 10 -- it directly measures the practical triage decision: "assign this fragment to an existing directory, or flag it as new?" Unlike AUCROC (which measures ranking quality) or Acc@1 (which measures retrieval rank), Assignment Accuracy captures the end-to-end decision that a scholar would rely on.

While AUCROC is a tie, the practical assignment metric shows meaningful differences:

| Model | Assignment Acc | Existing Acc | New Acc | Config |
|-------|---------------|-------------|---------|--------|
| **PhilTa** | **88.7%** | 89.1% | 88.4% | FF1 L7, SIF+ABTT(D=10) |
| **LaTa** | 87.5% | 79.7% | 95.3% | hidden L0, SIF only |
| **LaBSE** | 87.2% | 90.3% | 84.0% | hidden L12, SIF+ABTT(D=10) |
| **KaLM-mini** | 86.9% | 75.6% | 98.1% | hidden L23, baseline |
| **Qwen3-0.6B** | 86.4% | 85.0% | 87.8% | FFN L22, SIF+ABTT(D=2) |

PhilTa achieves the **best-balanced** assignment at 88.7%, with nearly equal accuracy on existing (89.1%) and new (88.4%) files. This balance is crucial for practical use -- a system that only excels at one category is unreliable.

Notable asymmetries:
- **KaLM-mini** is extremely conservative: it flags 98.1% of new files correctly but only matches 75.6% of existing files. It defaults to "this is new."
- **LaBSE** shows the opposite tendency: strong existing matching (90.3%) but weaker novelty detection (84.0%).
- **Qwen3-0.6B** is well-balanced (85.0% / 87.8%) but slightly below PhilTa overall.

> **[See Fig 3: `fig3_assign_accuracy.png`]** -- Grouped bars showing the existing/new/overall breakdown per model.

### 2. Whitening Fails Catastrophically -- Again

PCA whitening is a complete failure across all five models, confirming Phase 9's findings with additional evidence:

| Model | Best Whitening AUCROC | Best Whitening Assignment |
|-------|----------------------|-------------------------|
| LaTa | 0.589 | 50.1% |
| PhilTa | 0.612 | 50.1% |
| LaBSE | 0.578 | 50.1% |
| Qwen3-0.6B | 0.611 | 50.1% |
| KaLM-mini | 0.619 | 50.1% |

Every model with whitening produces an assignment accuracy of exactly 50.1% -- essentially random. The mechanism is the same as Phase 9: whitening sets tau=0, classifying **everything** as "Existing" (100% existing accuracy, 0% new accuracy). The decorrelation destroys the discriminative signal in the cosine space.

This failure is now confirmed across five models from three architecture families. **Whitening is not a viable approach for this task.**

> **[See Fig 5: `fig5_assign_whitening.png`]** -- Side-by-side comparison: whitening (all existing, no new) vs SIF+ABTT (balanced).

### 3. The Cosine Gap Reveals Method Quality

The gap between same-folder and different-folder cosine similarities is the clearest diagnostic of method effectiveness:

| Method | Mean Gap | Max Gap |
|--------|----------|---------|
| Baseline | 0.067 | 0.336 |
| SIF only | 0.139 | 0.444 |
| Whitening | 0.114 | 0.339 |
| **SIF+ABTT (D=10)** | **0.510** | **0.597** |
| **SIF+ABTT (opt D)** | **0.514** | **0.603** |

SIF+ABTT opens a 0.5+ gap on average -- same-directory pairs score ~0.55-0.60 cosine similarity while different-directory pairs cluster near 0.0. This massive separation is what enables reliable thresholding for the assignment decision.

> **[See Fig 4: `fig4_assign_gap.png`]** -- Gap profiles across layers: SIF+ABTT maintains a high gap everywhere while baseline collapses.

### 4. Optimal D Trends Higher with Assignment Accuracy Optimization

In Phase 9 (where D was optimized for AUCROC), typical optimal D values were 1-3 for T5 models. Now, optimizing for Assignment Accuracy, D trends higher:

| Model | D Mode | D Mean | D Range |
|-------|--------|--------|---------|
| **PhilTa** | 10 | 9.2 | 5-10 |
| **LaBSE** | 10 | 9.3 | 7-10 |
| **LaTa** | 10 | 8.8 | 5-10 |
| **KaLM-mini** | 10 | 8.7 | 2-10 |
| **Qwen3-0.6B** | 10 | 8.4 | 1-10 |

D=10 is optimal for most layers and models. Assignment Accuracy rewards more aggressive component removal than AUCROC does -- removing more principal components creates a cleaner separation between the "same" and "different" distributions, even if it slightly reduces the ranking-based AUCROC.

This validates using D=10 as a universal default for practical assignment systems.

> **[See Fig 6: `fig6_assign_optimal_d.png`]** -- D=10 dominates across all models when optimizing for assignment.

### 5. AUCROC Is Saturated -- All Models Tied

With SIF+ABTT, AUCROC is essentially a solved problem. All five models converge to within a 0.5 pp band:

| Model | Best AUCROC | Config |
|-------|-----------|--------|
| **Qwen3-0.6B** | **0.9773** | hidden L21, SIF+ABTT(D=5) |
| **KaLM-mini** | 0.9754 | FFN L1, SIF+ABTT(D=10) |
| **PhilTa** | 0.9750 | hidden L10, SIF+ABTT(D=5) |
| **LaTa** | 0.9743 | FF1 L1, SIF+ABTT(D=10) |
| **LaBSE** | 0.9725 | hidden L0, SIF+ABTT(D=10) |

This saturation is precisely why we shifted to Assignment Accuracy as the optimization target -- AUCROC cannot differentiate models at this level, but Assignment Accuracy reveals meaningful differences (88.7% vs 86.4%, a 2.3 pp spread).

> **[See Fig 2: `fig2_assign_method_heatmap.png`]** -- The heatmap shows solid green for SIF+ABTT columns across all models, while whitening is uniformly red.

### 6. The Anisotropy Dip Transfers to This Task

The dip we observed in STS also manifests here, with the same architecture-dependent severity:

| Model | Best Baseline AUCROC | Worst Baseline AUCROC | Drop |
|-------|---------------------|----------------------|------|
| **LaTa** | 0.930 (L0) | 0.463 (L8) | **46.6 pp** |
| **PhilTa** | 0.952 (L0) | 0.464 (L6) | **48.8 pp** |
| **LaBSE** | 0.948 (L11) | 0.843 (L1) | **10.5 pp** |
| **Qwen3-0.6B** | 0.952 (L28) | 0.784 (L4) | **16.8 pp** |
| **KaLM-mini** | 0.958 (L23) | 0.865 (L5) | **9.4 pp** |

SIF+ABTT eliminates the dip entirely for all architectures.

> **[See Fig 1: `fig1_assign_dip.png`]** -- Baseline (dashed) vs SIF+ABTT (solid) profiles for all five models.

### 7. Retrieval Accuracy (Acc@1): A Secondary Metric

Acc@1 measures pure retrieval rank (is the correct partner at rank 1?) -- a useful but secondary metric compared to Assignment Accuracy. Latin-specialized models retain their edge here:

| Model | Best Acc@1 | Config |
|-------|-----------|--------|
| **LaTa** | **91.6%** | FF1 L4, SIF+ABTT(D=10) |
| **PhilTa** | **91.6%** | FF1 L7, SIF+ABTT(D=7) |
| **LaBSE** | 90.9% | hidden L11, SIF+ABTT(D=10) |
| **Qwen3-0.6B** | 89.1% | FFN L24, SIF+ABTT(D=10) |
| **KaLM-mini** | 89.1% | hidden L23, baseline |

LaTa and PhilTa tie at 91.6%, 2.5 pp ahead of Qwen3 and KaLM. This contrasts with the STS task where the new models won -- suggesting that Latin-specific training provides an edge for fine-grained within-corpus retrieval, even though the general models have better overall similarity estimation.

> **[See Fig 8: `fig8_assign_acc1.png`]** -- Layer-wise Acc@1 profiles for all five models.

### 8. Hidden States vs FFN: Both Matter for Assignment

Focusing on Assignment Accuracy (our primary metric), the best representation type varies by model:

| Model | Best Assignment Repr | Assignment Acc | Runner-up Repr | Assignment Acc |
|-------|---------------------|---------------|---------------|---------------|
| **PhilTa** | FF1 | **88.7%** | hidden | 87.3% |
| **LaTa** | hidden | **87.5%** | FF1 | 86.4% |
| **LaBSE** | hidden | **87.2%** | -- | -- |
| **KaLM-mini** | hidden | **86.9%** | FFN | 84.2% |
| **Qwen3-0.6B** | FFN | **86.4%** | hidden | 85.8% |

There is no universal winner. PhilTa and Qwen3 prefer FFN activations, while LaTa, LaBSE, and KaLM prefer hidden states. This motivates exhaustively evaluating both representation types.

---

## Summary Table: Best Result per Model (ranked by Assignment Accuracy)

| Model | Architecture | Assignment Acc | Existing | New | Acc@1 | AUCROC | Best Config |
|-------|-------------|---------------|----------|-----|-------|--------|-------------|
| **PhilTa** | T5 | **88.7%** | 89.1% | 88.4% | 91.6% | 0.975 | FF1 L7, SIF+ABTT(D=10) |
| **LaTa** | T5 | 87.5% | 79.7% | 95.3% | 91.6% | 0.974 | hidden L0, SIF only |
| **LaBSE** | BERT | 87.2% | 90.3% | 84.0% | 90.9% | 0.973 | hidden L12, SIF+ABTT(D=10) |
| **KaLM-mini** | Gemma | 86.9% | 75.6% | 98.1% | 89.1% | 0.975 | hidden L23, baseline |
| **Qwen3-0.6B** | Qwen3 | 86.4% | 85.0% | 87.8% | 89.1% | 0.977 | FFN L22, SIF+ABTT(D=2) |

> **[See Fig 7: `fig7_assign_summary.png`]** -- Visual summary table with best results highlighted.

---

## What This Means for the Paper

1. **Assignment Accuracy, not AUCROC or Acc@1, is the right optimization target.** AUCROC is saturated (~0.97 for all models) and cannot differentiate them. Acc@1 measures retrieval rank, not the real-world triage decision. Assignment Accuracy directly captures whether a fragment is correctly routed to its directory or flagged as new -- the decision a scholar actually needs.

2. **Optimizing D for Assignment Accuracy produces more aggressive (and appropriate) correction.** D=10 dominates across all models, compared to D=1-3 when optimizing for AUCROC (Phase 9). This stronger correction creates cleaner same/different separation, which is exactly what the threshold-based assignment decision requires.

3. **PhilTa is the recommended model for directory assignment** at 88.7% balanced accuracy -- the only model above 88% on both existing and new files.

4. **Whitening fails universally.** This is now confirmed across five models from three architectures. It should not be recommended for cosine-based duplicate detection.

5. **Latin-specific models retain an edge in retrieval** (Acc@1: 91.6% vs 89.1%), but **general multilingual models are competitive on assignment** (86-87% vs 87-89%). The gap is narrowing.

6. **The existing/new balance matters.** Models like KaLM-mini achieve high overall assignment through extreme conservatism (98% new detection, only 76% existing matching). PhilTa's balanced profile (89%/88%) is more practically useful for automated triage.

---

## Figures Reference

All figures are in `runs/phase10/experiment1/figures/`:

| Figure | File | What It Shows |
|--------|------|--------------|
| **Fig 1** | `fig1_assign_dip.png` | AUCROC dip: baseline vs SIF+ABTT by layer |
| **Fig 2** | `fig2_assign_method_heatmap.png` | Best AUCROC per (model x method) heatmap |
| **Fig 3** | `fig3_assign_accuracy.png` | Assignment accuracy breakdown: existing / new / overall |
| **Fig 4** | `fig4_assign_gap.png` | Cosine gap by layer: baseline vs SIF+ABTT |
| **Fig 5** | `fig5_assign_whitening.png` | Whitening vs SIF+ABTT assignment breakdown |
| **Fig 6** | `fig6_assign_optimal_d.png` | Optimal D per model and layer |
| **Fig 7** | `fig7_assign_summary.png` | Summary table (best per model) |
| **Fig 8** | `fig8_assign_acc1.png` | Retrieval Acc@1 by layer |

All figures are also available as PDFs for direct paper inclusion.

---

## Raw Data

- Assignment Results CSV: `runs/phase9/phase10_experiment1_results.csv` (845 rows, 22 columns)
- Phase 9 Split: `runs/phase9/phase9_split.csv` (1,278 rows)
- STS Results CSV: `runs/phase8_results/phase8_canon_sweep.csv` (724 rows, 20 columns)
