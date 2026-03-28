# Phase 10 Experiment 1: Semantic Textual Similarity (STS)

## What We Asked

Can newer multilingual embedding models outperform our Latin-specialized models (LaTa, PhilTa) and the strong LaBSE baseline on the STS retrieval task over a corpus of 1,278 Latin manuscript fragments? And do the same anisotropy phenomena we observed in Phase 8 also appear in decoder-only architectures?

We extended the Phase 8 canon sweep to include two new models -- Qwen3-Embedding-0.6B and KaLM-mini -- for a total of five models spanning three architecture families: Seq2Seq (T5), Encoder (BERT), and Decoder-only (Qwen, Gemma).

---

## Experimental Setup

### The Task

Given a test query fragment, retrieve the correct same-directory partner from a gallery of train fragments. Success = the correct partner is ranked first (Acc@1), within the top 3 (Acc@3), or top 5 (Acc@5). This is the same retrieval pipeline as Phase 8.

### Dataset Split

| | LaTa / PhilTa / LaBSE | Qwen3-0.6B / KaLM-mini |
|---|---|---|
| **Train gallery** | 943 | 639 |
| **Test queries** | 335 | 320 |

Note: the new models use the Phase 9 split (50/50), while the original three models use the Phase 8 split. Both splits are leak-free.

### Models Tested

| Model | Architecture | Layers | Parameters |
|-------|-------------|--------|------------|
| **LaTa** | T5 (Seq2Seq) | 0-12 hidden + 1-12 FF1 | Latin-adapted |
| **PhilTa** | T5 (Seq2Seq) | 0-12 hidden + 1-12 FF1 | Philological |
| **LaBSE** | BERT (Encoder) | 0-12 hidden + 0-12 FFN | Multilingual |
| **Qwen3-0.6B** | Qwen3 (Decoder) | 0-28 hidden + 0-27 FFN | 0.6B multilingual |
| **KaLM-mini** | Gemma (Decoder) | 0-24 hidden + 0-23 FFN | Multilingual |

### Post-Processing

| Method | Description |
|--------|-------------|
| **Baseline** | Mean-pool, no correction |
| **ABTT** | SIF pooling + remove top D=10 principal components |

---

## Key Findings

### 1. The New Models Match or Beat the Latin-Specialized Models

The headline result: **KaLM-mini achieves the best Acc@1 of any model at 95.0%**, narrowly edging out Qwen3-0.6B (94.7%) and LaTa (94.0%). The two new general-purpose multilingual models outperform the Latin-specialized T5 models on this task.

| Model | Best Acc@1 | Acc@3 | Acc@5 | Config |
|-------|-----------|-------|-------|--------|
| **KaLM-mini** | **95.0%** | 95.6% | 95.9% | FFN L2, ABTT |
| **Qwen3-0.6B** | **94.7%** | 95.3% | 95.6% | FFN L25, ABTT |
| **LaTa** | 94.0% | 96.7% | 97.6% | hidden L0, ABTT |
| **PhilTa** | 93.1% | 95.2% | 95.8% | hidden L1, ABTT |
| **LaBSE** | 92.2% | 95.2% | 96.1% | hidden L0, ABTT |

This is a surprising result: general-purpose models with no Latin-specific training surpass models fine-tuned for Latin text. It suggests that the multilingual pretraining in Qwen3 and KaLM captures enough cross-lingual structure to handle Latin effectively.

> **[See Fig 3: `fig3_sts_best_bar.png`]** -- Grouped bar chart comparing Acc@1/3/5 across all five models.

### 2. The Anisotropy Dip Is Architecture-Dependent

The catastrophic middle-layer collapse we documented in Phase 8 reappears, but its severity depends strongly on architecture:

| Architecture | Model | Worst Layer | Worst Acc@1 | Best Acc@1 | Drop |
|-------------|-------|------------|-------------|------------|------|
| **T5 (Seq2Seq)** | LaTa | L6 | 6.6% | 93.4% | **86.9 pp** |
| **T5 (Seq2Seq)** | PhilTa | L10 | 6.6% | 92.2% | **85.7 pp** |
| **BERT (Encoder)** | LaBSE | L2 | 49.0% | 88.1% | **39.1 pp** |
| **Decoder** | Qwen3-0.6B | L5 | 75.0% | 93.1% | **18.1 pp** |
| **Decoder** | KaLM-mini | L0 | 79.1% | 94.1% | **15.0 pp** |

Three tiers emerge:
- **T5 models**: catastrophic dip (~87 pp drop to near-chance performance)
- **BERT encoder**: moderate dip (~39 pp, but floor at 49%)
- **Decoder-only models**: mild dip (~15-18 pp, floor at 75-79%)

The decoder architectures (Qwen, KaLM) exhibit remarkably stable baseline performance. Even without any post-processing, their worst layer still retrieves correctly 75%+ of the time. This suggests that decoder-only pretraining produces more isotropic representations than the T5 encoder-decoder setup.

> **[See Fig 1: `fig1_sts_dip_hidden.png`]** -- Side-by-side dip profiles for all five models.
>
> **[See Fig 5: `fig5_sts_dip_severity.png`]** -- Dip magnitude comparison by architecture.

### 3. ABTT Universally Solves the Dip

Despite the varying dip severity, ABTT (SIF + remove top-D PCs) is uniformly effective across all architectures:

| Model | Baseline Floor | ABTT Floor | ABTT Ceiling | ABTT Range |
|-------|---------------|------------|--------------|------------|
| **LaTa** | 6.6% | 84.5% | 94.0% | 9.6 pp |
| **PhilTa** | 6.6% | 84.5% | 93.1% | 8.7 pp |
| **LaBSE** | 49.0% | 76.1% | 92.2% | 16.1 pp |
| **Qwen3-0.6B** | 75.0% | 89.1% | 94.7% | 5.6 pp |
| **KaLM-mini** | 79.1% | 84.4% | 95.0% | 10.6 pp |

With ABTT applied, the worst-case performance across all models and layers rises to at least 76%. For the decoder models, the worst ABTT layer still exceeds 84% -- making layer selection nearly irrelevant.

> **[See Fig 1: `fig1_sts_dip_hidden.png`]** -- The solid lines (ABTT) stay flat while dashed lines (baseline) plummet.

### 4. FFN Representations Are Surprisingly Strong

The intermediate FFN activations (FF1 for T5, FFN-intermediate for the others) show a different dip pattern and are competitive with hidden states:

| Model | Hidden Best | FFN Best | FFN Dip (baseline) |
|-------|-----------|----------|-------------------|
| **LaTa** | 94.0% (L0) | 92.8% (L1) | 86.9 pp |
| **PhilTa** | 93.1% (L1) | 88.4% (L7) | 84.2 pp |
| **LaBSE** | 88.1% (L12) | 84.8% (L11) | 48.7 pp |
| **Qwen3-0.6B** | 93.1% (L23) | 94.4% (L23) | 20.0 pp |
| **KaLM-mini** | 94.1% (L23) | 92.8% (L19) | 33.1 pp |

For Qwen3-0.6B, the FFN representation actually produces the **overall best Acc@1** (94.7% at FFN L25 with ABTT). This indicates that the FFN intermediate activations -- often overlooked in favor of hidden states -- contain strong semantic signal, especially in deeper decoder layers.

> **[See Fig 2: `fig2_sts_dip_ffn.png`]** -- FFN layer profiles show different dip dynamics than hidden states.

### 5. Cosine Gap Confirms the Story

The cosine gap (mean similarity of same-folder pairs minus different-folder pairs) with ABTT shows consistent separability:

| Model | Best Gap (ABTT) | Layer |
|-------|----------------|-------|
| **LaTa** | 0.598 | L1 |
| **PhilTa** | 0.589 | L2 |
| **LaBSE** | 0.629 | L11 |
| **Qwen3-0.6B** | 0.574 | L0 |
| **KaLM-mini** | 0.558 | L23 |

Without ABTT (panel A), baseline gaps collapse to near zero for T5 models in the middle layers -- the same anisotropy dip visible in a different metric. With ABTT (panel B), all five models are lifted to the 0.5-0.6 range, and the catastrophic collapse disappears entirely.

> **[See Fig 4: `fig4_sts_gap.png`]** -- Side-by-side: baseline gaps collapse for T5 (left) while ABTT lifts all models to consistent 0.5-0.6 separability (right).

---

## Summary Table: Best STS Result per Model

| Model | Architecture | Best Acc@1 | Acc@3 | Acc@5 | Config |
|-------|-------------|-----------|-------|-------|--------|
| **KaLM-mini** | Gemma (Decoder) | **95.0%** | 95.6% | 95.9% | FFN L2, ABTT |
| **Qwen3-0.6B** | Qwen3 (Decoder) | **94.7%** | 95.3% | 95.6% | FFN L25, ABTT |
| **LaTa** | T5 (Seq2Seq) | 94.0% | 96.7% | 97.6% | hidden L0, ABTT |
| **PhilTa** | T5 (Seq2Seq) | 93.1% | 95.2% | 95.8% | hidden L1, ABTT |
| **LaBSE** | BERT (Encoder) | 92.2% | 95.2% | 96.1% | hidden L0, ABTT |

---

## What This Means for the Paper

1. **General-purpose multilingual models can outperform domain-specific models** on Latin STS. This challenges the assumption that Latin NLP requires Latin-specific training.

2. **Decoder-only architectures are more robust to anisotropy.** Their baseline dip is 3-5x less severe than T5 models, and ABTT closes the remaining gap.

3. **FFN intermediate activations deserve attention.** Both Qwen3 and KaLM achieve their best results from FFN layers, not hidden states. This is a novel finding for embedding extraction.

4. **ABTT is architecture-agnostic.** The same D=10 correction works across T5, BERT, Qwen, and Gemma without per-model tuning.

5. **KaLM-mini (Gemma-based) is the recommended model** for Latin STS: it achieves the highest Acc@1 (95.0%) and shows the mildest anisotropy dip (15 pp vs 87 pp for T5).

---

## Figures Reference

All figures are in `runs/phase10/experiment1/figures/`:

| Figure | File | What It Shows |
|--------|------|--------------|
| **Fig 1** | `fig1_sts_dip_hidden.png` | The anisotropy dip: Acc@1 vs layer for hidden states |
| **Fig 2** | `fig2_sts_dip_ffn.png` | Same for FFN representations |
| **Fig 3** | `fig3_sts_best_bar.png` | Best Acc@1/3/5 grouped bar chart |
| **Fig 4** | `fig4_sts_gap.png` | Cosine gap across layers (ABTT) |
| **Fig 5** | `fig5_sts_dip_severity.png` | Dip severity comparison by architecture |

All figures are also available as PDFs for direct paper inclusion.

---

## Raw Data

- STS Results CSV: `runs/phase8_results/phase8_canon_sweep.csv` (724 rows, 20 columns)
- Phase 9 Split: `runs/phase9/phase9_split.csv` (1,278 rows)
