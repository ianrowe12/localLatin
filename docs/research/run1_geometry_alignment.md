# Run 1: Bridging Geometric Embedding Operations and Token-Level Interpretability

**Date**: 2025-03-25
**Scope**: Web research survey across 5 areas — isotropy-aware pooling, token alignment, representation engineering, disentangled representations, and representation similarity analysis.

---

## Executive Summary

Our core problem: ABTT (removing top-D PCs from mean-pooled embeddings) fixes retrieval but corrupts Integrated Gradients — IG highlights tokens loading onto removed PCs (a geometric property), not semantically meaningful tokens. After surveying ~40 methods across 5 research areas, three approaches stand out as highest-priority for Phase 13:

1. **Ditto (Diagonal Attention Pooling)** — Replace SIF-weighted mean pooling with the model's own self-attention diagonal as pooling weights. Zero training, fully differentiable, per-instance (no corpus-level statistics in the gradient path). This is the cleanest fix for IG attribution quality.

2. **Difference-in-Means + LEACE** — Learn a "same-document direction" from manuscript pairs, then use LEACE (closed-form concept erasure) to remove only the anisotropy concept while provably preserving the document-identity subspace. This replaces blind ABTT with targeted, principled direction removal.

3. **Token-Level Optimal Transport (BERTScore / Sinkhorn)** — Bypass mean pooling entirely. Compute document similarity from token-to-token matching via greedy cosine (BERTScore-style) or entropic optimal transport (Sinkhorn). The alignment matrix IS the explanation — no IG needed.

These three are complementary: Ditto fixes the pooling path for IG, LEACE fixes the PC removal for IG, and OT provides an entirely different explanation modality (token-pair alignment maps vs. per-token importance scores).

---

## Current Pipeline Bottleneck

```
tokens → SIF-weighted mean pool → ABTT (remove top-D PCs) → L2 norm → cosine sim
                ↑                        ↑
         corpus-level P(w)         corpus-level SVD
```

Both SIF and ABTT inject **corpus-level statistics** into the forward path. IG treats these as constants, so:
- SIF: gradients are scaled by fixed frequency weights — acceptable, since high-frequency tokens genuinely contribute less
- ABTT: gradients flow through `centered - centered @ pcs.T @ pcs` — the PC projection amplifies tokens that load onto removed PCs, which is a geometric artifact, not a semantic signal

Phase 12c showed ABTT shifts attribution from empty tokens to content words (content share: 13.8% → 72.6% for LaTa L4). But the specific token-pair connections highlighted by IG through ABTT don't make semantic sense to Latin experts — they reflect PC geometry, not textual relationships.

---

## Area 1: Isotropy-Aware Pooling

### Methods Found

| Method | Paper | Year | Summary |
|--------|-------|------|---------|
| **Ditto (Diagonal Attention)** | Chen et al. | 2023 | Use self-attention diagonal as pooling weights; per-instance, zero training |
| **Power-Mean Pooling** | Ruckle et al. | 2018 | Concatenate multiple power means (p=1,3,-inf,+inf); diversifies representation |
| **Soft-ZCA Whitening** | Wendler et al. | 2024 | ZCA with epsilon regularization; preserves coordinate alignment unlike PCA whitening |
| **WhiteningBERT** | Huang et al. | 2021 | PCA whitening on multi-layer averaged embeddings |
| **Cluster-Based Isotropy** | Rajaee & Pilehvar | 2021 | Per-cluster PC removal (stop words, content words, punctuation separately) |
| **BERT-flow** | Li et al. | 2020 | Normalizing flow to map anisotropic distribution to isotropic Gaussian |

### Feasibility Assessment

| Method | Frozen? | Clean IG? | Scholar-friendly? | Effort | Recommended? |
|--------|:-------:|:---------:|:-----------------:|:------:|:------------:|
| Ditto | Yes | **Excellent** | Very good | Trivial (~10 lines) | **YES — Priority 1** |
| Power-Mean | Yes | Excellent | Good | Trivial (~15 lines) | YES — Priority 2 |
| Soft-ZCA | Yes | Moderate | Moderate | Easy | YES — comparison |
| WhiteningBERT | Yes | Poor | Poor | Easy | Retrieval only |
| Cluster-Based | Yes | Moderate | Good | Moderate | Hybrid approach |
| BERT-flow | Partial | Poor | Poor | High (training) | No |

### Key Insight

The fundamental tension: **corpus-level statistics in the forward path corrupt per-instance IG attributions**. ABTT, whitening, and ZCA all introduce corpus-level transforms that IG treats as constants but which systematically distort token importance. The cleanest solution is **per-instance pooling signals** (Ditto's attention diagonal) with no post-pooling linear transforms.

### Recommended Next Steps

1. **Ditto**: Extract `output_attentions=True` from each model, take diagonal of layer L's attention matrix, use as `weights` in existing `weighted_mean_pool()`. Compare retrieval (Assignment Accuracy) and IG attribution quality against SIF+ABTT.

2. **Power-Mean**: Implement PM with p={1, 3, -1}, concatenate to 3x hidden_dim. Each component is differentiable. Test whether concatenated representation is naturally more isotropic.

3. **Uniformity/Alignment metrics** (from SimCSE): Compute for all candidate methods as evaluation framework.

### GitHub Repositories

- Power-Mean: [UKPLab/arxiv2018-xling-sentence-embeddings](https://github.com/UKPLab/arxiv2018-xling-sentence-embeddings)
- WhiteningBERT: [Jun-jie-Huang/WhiteningBERT](https://github.com/Jun-jie-Huang/WhiteningBERT)
- Soft-ZCA: [drndr/code_isotropy](https://github.com/drndr/code_isotropy)
- Cluster-Based: [Sara-Rajaee/clusterbased_isotropy_enhancement](https://github.com/Sara-Rajaee/clusterbased_isotropy_enhancement)
- BERT-flow: [bohanli/BERT-flow](https://github.com/bohanli/BERT-flow) (TensorFlow)

---

## Area 2: Token Alignment / Matching

### Methods Found

| Method | Paper | Year | Summary |
|--------|-------|------|---------|
| **BERTScore** | Zhang et al. | 2020 | Greedy max-cosine matching between token sets; P/R/F1 aggregation |
| **Word Mover's Distance** | Kusner et al. | 2015 | Optimal transport between token embedding sets |
| **MoverScore** | Zhao et al. | 2019 | WMD + IDF weighting on contextualized embeddings |
| **Word Rotator's Distance** | Yokoi et al. | 2020 | Decomposes into norm (importance) x direction (meaning); OT on angles |
| **Sinkhorn Divergence** | Cuturi | 2013 | Entropic OT; differentiable, O(n^2*K) per pair |
| **RWMD** | Kusner et al. | 2015 | Relaxed WMD; O(n^2) per pair, no transport plan |
| **ColBERT MaxSim** | Khattab & Zaharia | 2020 | Sum of per-query-token max cosine; designed for retrieval |

### Feasibility Assessment

| Method | Frozen? | Token-pair explns? | Scholar-friendly? | 816K pairs on A100 | Recommended? |
|--------|:-------:|:-----------------:|:-----------------:|:------------------:|:------------:|
| BERTScore | Yes | **Yes** (alignment matrix) | **High** | 5-15 min | **YES — Priority 1** |
| Sinkhorn | Yes | **Yes** (soft transport plan) | **High** | 10-30 min | **YES — Priority 2** |
| WRD | Yes | **Yes** (norm + direction) | **Very high** | 7-20 hrs (CPU) | Selected pairs |
| WMD | Yes | **Yes** (transport plan) | High | 7-20 hrs (CPU) | Selected pairs |
| RWMD | Yes | No (scalar only) | Moderate | 5-15 min | Fast screening |
| MoverScore | Yes | Yes | High | Same as WMD | = WMD + IDF |
| ColBERT | Yes | Partial (MaxSim) | Moderate | 5-15 min | Fast screening |

### Critical Finding: Anisotropy Affects Token-Level Methods Too

Token-level matching is NOT immune to anisotropy. If middle-layer token embeddings cluster in a narrow cone, all pairwise token cosines are high and undifferentiated, destroying the matching signal. **Per-token ABTT correction is needed**: compute PCs from all token embeddings across the corpus, project out top-D from each token before computing the similarity/cost matrix.

### Key Insight: The Alignment Matrix IS the Explanation

For all OT-based methods, the transport plan `T[i,j]` directly shows "token i in doc A corresponds to token j in doc B, with weight T[i,j]." This is qualitatively different from IG: token-pair alignment maps vs. per-token importance scores. A Latin scholar can inspect which words align across manuscripts without needing to understand gradients.

### Recommended Next Steps

**Phase 1 — Quick wins (1-2 days)**:
1. Save token-level embeddings (modify extraction to optionally save `[N, seq_len, dim]`)
2. BERTScore-style greedy matching with per-token ABTT
3. RWMD for fast approximate ranking

**Phase 2 — Full OT (2-3 days)**:
4. Sinkhorn via POT `ot.sinkhorn2()` for full corpus
5. WRD on subset to test if norm/direction decomposition reduces need for ABTT

**Phase 3 — Explanations (1-2 days)**:
6. Exact WMD on top retrieval pairs → transport plan heatmaps

### Storage Estimate

Token-level embeddings: 1,278 docs x 512 tokens x 768 dim x 4 bytes = ~2 GB/layer. Variable-length storage reduces to ~700 MB/layer.

### GitHub Repositories

- BERTScore: [Tiiiger/bert_score](https://github.com/Tiiiger/bert_score)
- MoverScore: [AIPHES/emnlp19-moverscore](https://github.com/AIPHES/emnlp19-moverscore)
- WRD: [eumesy/wrd](https://github.com/eumesy/wrd)
- POT (Sinkhorn, WMD): [PythonOT/POT](https://github.com/PythonOT/POT) (already installed)
- GeomLoss (GPU Sinkhorn): [kernel-operations.io/geomloss](https://www.kernel-operations.io/geomloss/)
- ColBERT: [stanford-futuredata/ColBERT](https://github.com/stanford-futuredata/ColBERT)
- Re-evaluating WMD: [joisino/reeval-wmd](https://github.com/joisino/reeval-wmd)

---

## Area 3: Representation Engineering / Steering

### Methods Found

| Method | Paper | Year | Summary |
|--------|-------|------|---------|
| **Difference-in-Means** | (classical) | — | Mean(same-folder pairs) - Mean(diff-folder pairs) = "document direction" |
| **LEACE** | Belrose et al. | 2023 | Closed-form linear concept erasure; provably erases exactly the target concept |
| **RepE / Reading Vectors** | Zou et al. | 2023 | Identify concept directions via activation differences |
| **INLP** | Ravfogel et al. | 2020 | Iterative null-space projection; superseded by LEACE |
| **Geometry of Truth** | Marks & Tegmark | 2023 | Linear separability of concepts in LLM representations |
| **Probing Classifiers** | (standard) | — | Linear probes on token embeddings predict folder membership |
| **Rogue Dimensions** | Timkey & van Schijndel | 2021 | 1-3 dimensions account for most anisotropy; zero them out |

### Feasibility Assessment

| Method | Frozen? | Token-level? | Scholar-friendly? | Effort | Recommended? |
|--------|:-------:|:------------:|:-----------------:|:------:|:------------:|
| Diff-in-Means | Yes | **Yes** | **High** | ~50 lines NumPy | **YES — Priority 1** |
| LEACE | Yes | **Yes** | High | pip install + ~100 lines | **YES — Priority 2** |
| Probing | Yes | Yes | Moderate | ~80 lines sklearn | YES — Priority 3 |
| RepE | Yes | Yes | High | ~30 lines custom | = Diff-in-Means |
| Geometry of Truth | Yes | Yes | High | Reference methodology | Informational |
| INLP | Yes | Yes | Moderate | Superseded by LEACE | No |
| Rogue Dims | Yes | Yes | Moderate | Trivial | Quick comparison |

### Key Insight: Decompose ABTT PCs Against Semantic Directions

The most impactful experiment is checking whether ABTT's PC1 aligns with the learned "same-document direction":

```python
# Learn document direction from training pairs
same_pairs = [emb_i - emb_j for (i,j) where folder_i == folder_j]
diff_pairs = [emb_i - emb_j for (i,j) where folder_i != folder_j]
doc_direction = mean(same_pairs) - mean(diff_pairs)
doc_direction /= norm(doc_direction)

# Check alignment with ABTT PCs
for k in range(D):
    print(f"cos(doc_direction, PC{k}) = {cos(doc_direction, pcs[k])}")
```

If `cos(doc_direction, PC1) ≈ 0`: PC1 is pure noise, ABTT is safe.
If `cos(doc_direction, PC1) >> 0`: ABTT is destroying useful signal.

**Token-level explanation**: For each token t, `score_t = h_t . doc_direction` gives "how much does this token carry the same-document signal?" This is exactly what Latin scholars want.

### LEACE as Principled ABTT Replacement

Instead of removing the top-D PCs blindly, LEACE removes **exactly the directions that predict a specified concept** while provably preserving everything orthogonal. Two applications:

1. **Erase anisotropy**: Define concept z = token frequency bin. LEACE removes directions encoding frequency, which is the hypothesized content of PC1.
2. **Preserve document identity**: Define concept z = folder membership. LEACE identifies the document-identity subspace. Remove its complement for noise reduction.

LEACE is a drop-in replacement for `EmbeddingCleaner`: `LeaceEraser.fit(X, z)` returns a projection matrix.

### Per-Token LEACE (Novel Application)

Reshape token hidden states to `(N_total_tokens, dim)`, label each token by a concept (e.g., frequency bin), fit LEACE, apply per-token before pooling. This gives "cleaned token embeddings" that are then SIF-weighted and pooled — IG flows through the clean path.

### Recommended Next Steps

**Priority 1 — PC1 Identity Experiment (1 day)**:
- Correlate PC1 projections with: token frequency (from `token_probabilities`), position, token category
- ~50 lines of NumPy on existing `.npy` files
- Directly answers: "what is PC1?"

**Priority 2 — Difference-in-Means Direction (1 day)**:
- Learn document direction from training pairs
- Measure `cos(doc_direction, PC_k)` for k=1..D
- Per-token projection as explanation

**Priority 3 — LEACE Concept Erasure (1-2 days)**:
- `pip install concept-erasure`
- Replace ABTT with targeted erasure
- Compare retrieval against blind ABTT

### GitHub Repositories

- LEACE: [EleutherAI/concept-erasure](https://github.com/EleutherAI/concept-erasure) (`pip install concept-erasure`)
- INLP: [shauli-ravfogel/nullspace_projection](https://github.com/shauli-ravfogel/nullspace_projection)
- RepE: [andyzoujm/representation-engineering](https://github.com/andyzoujm/representation-engineering)
- Geometry of Truth: [saprmarks/geometry-of-truth](https://github.com/saprmarks/geometry-of-truth)

---

## Area 4: Disentangled Representations

### Methods Found

| Method | Paper | Year | Summary |
|--------|-------|------|---------|
| **Sparse Autoencoders (SAEs)** | Cunningham et al. / Bricken et al. | 2023 | Overcomplete dictionary with L1 sparsity; each feature ideally monosemantic |
| **ICA** | (classical) | — | Independent component analysis; separates non-Gaussian sources |
| **Dictionary Learning** | Yun et al. | 2021 | Sparse coding via LASSO; similar to SAEs but convex |
| **Linear Concept Probing** | Park et al. / Nanda et al. | 2023 | Test linear representation hypothesis for specific concepts |
| **Structural Probes** | Hewitt & Manning | 2019 | Find syntax subspace; content = orthogonal complement |
| **NMF** | (classical) | — | Non-negative parts-based decomposition |
| **Information Bottleneck** | Tishby et al. | 1999 | Compress to keep task-relevant info only |

### Feasibility Assessment

| Method | Frozen? | Token-level? | Scholar-friendly? | Effort | Recommended? |
|--------|:-------:|:------------:|:-----------------:|:------:|:------------:|
| **SAEs** | Yes | **Yes** | Moderate (post-hoc labeling) | 3-5 days | **YES — Priority 4** |
| **ICA** | Yes | Yes | Low-moderate | 1 day | YES — quick test |
| **Linear Probing** | Yes | Yes | High | 1 day | **YES — Priority 1** |
| Dict. Learning | Yes | Yes | Moderate | 2-3 days | Redundant with SAEs |
| Structural Probes | Yes | Yes | High | 3-5 days | Only with treebank |
| NMF | Partial | Yes | Moderate | 1 day | No (sign mismatch) |
| Info. Bottleneck | No | No | Low | 1 week+ | No |

### Key Insight: SAEs Can Decompose PC1

After training an SAE on token embeddings at a dip layer, identify which SAE features align with PC1 (high dot product between decoder column and PC1). Remove those features' contributions, keep the rest. This is a **learned, feature-level version of ABTT** where each removed feature is individually interpretable: "Feature 47 fires on Latin conjunctions and aligns with PC1; Feature 203 fires on theological vocabulary and encodes content."

### SAE Training is Feasible at Your Scale

128K token embeddings x 768 dims is tiny by mechanistic interpretability standards. A 768→4096→768 SAE trains in ~5-30 minutes per layer on a single A100. Total for all 12 layers of LaTa: under 1 hour.

### ICA as Quick Baseline

```python
from sklearn.decomposition import FastICA
ica = FastICA(n_components=768)
S = ica.fit_transform(token_embeddings)  # Independent components
# Identify component most correlated with PC1/frequency
# Remove it; compare retrieval against ABTT
```

This runs in seconds and provides an interesting comparison: ICA exploits higher-order statistics (kurtosis) while PCA uses only variance.

### Recommended Next Steps

**Priority 1 — PC1 Correlation Analysis (1 day)**:
- For each token at dip layers: PC1 projection vs. log-frequency, position, token category
- 20-line NumPy experiment on existing data

**Priority 2 — ICA Comparison (1 day)**:
- FastICA on token embeddings → identify anisotropy component → remove → compare retrieval

**Priority 3 — SAE Decomposition (3-5 days)**:
- Train SAEs on token embeddings at dip layers (LaTa L4, PhilTa L6)
- Identify anisotropy-correlated features
- Label features post-hoc for interpretability story

### GitHub Repositories

- SAELens: [jbloomAus/SAELens](https://github.com/jbloomAus/SAELens)
- TransformerLens (SAE training): [neelnanda-io/TransformerLens](https://github.com/neelnanda-io/TransformerLens)
- Anthropic dictionary learning: [anthropics/dictionary_learning](https://github.com/anthropics/dictionary_learning)
- sparse_autoencoder: `pip install sparse-autoencoder`

---

## Area 5: CKA / Representation Similarity

### Methods Found

| Method | Paper | Year | Summary |
|--------|-------|------|---------|
| **Linear CKA** | Kornblith et al. | 2019 | Centered kernel alignment; compares representation geometry |
| **SVCCA** | Raghu et al. | 2017 | SVD + CCA; measures canonical correlations |
| **PWCCA** | Morcos et al. | 2018 | Projection-weighted CCA; variance-weighted correlations |
| **RSA** | Kriegeskorte et al. | 2008 | Compare representational dissimilarity matrices |
| **Cross-Similarity Matrix** | (direct computation) | — | C[i,j] = cos(h_A_i, h_B_j); simplest token-pair measure |
| **Procrustes** | Schonemann | 1966 | Best rigid rotation aligning two representation matrices |

### Critical Finding: CKA Requires Aligned Rows

Standard CKA compares two matrices with the **same rows** (same stimuli, different representations). For cross-document comparison (different tokens, same representation space), rows are unaligned. The feature-space workaround (d x d Gram matrices) loses token-level structure.

**No published work uses CKA as a document-to-document retrieval score.** CKA was designed for layer-to-layer comparison.

### Feasibility Assessment

| Method | Frozen? | Token-pair explns? | Scholar-friendly? | Scalability | Recommended? |
|--------|:-------:|:-----------------:|:-----------------:|:-----------:|:------------:|
| Cross-Sim Matrix | Yes | **Yes** | **High** | Excellent | **YES — use as basis** |
| Feature-space CKA | Yes | No (dimension-level) | Low | Excellent | Diagnostic only |
| SVCCA | Yes | No | Very low | Good | Dominated by CKA |
| PWCCA | Yes | No | Very low | Good | Dominated by CKA |
| RSA | Yes | Partial | Moderate | Good | Diagnostic only |
| Procrustes | Yes | Via permutation | Moderate | Good | Needs pre-alignment |

### Key Insight: The Cross-Similarity Matrix Is the Foundation

The simplest and most useful construct is `C = X_A @ X_B.T` (after per-token ABTT and L2-normalization), shape `[n_A, n_B]`. This is:
- The input to BERTScore (greedy max-matching on C)
- The cost matrix for WMD/Sinkhorn (1 - C)
- Directly interpretable: "token i in doc A has cosine X with token j in doc B"
- The basis for any aggregation strategy (max, mean, OT)

CKA adds nothing beyond this for cross-document retrieval. Use CKA only for layer selection and cross-model diagnostics.

### Recommended Next Steps

- Use **cross-similarity matrices** as the building block for all token-pair methods (Area 2)
- Use **CKA** only for layer analysis: which layers produce the most structurally similar representations for same-folder documents?
- Feature-space CKA per-layer as a diagnostic for choosing the best layer for token-level retrieval

---

## Cross-Cutting Comparison

### Top Methods Ranked by Overall Fit

| Rank | Method | Area | Frozen? | Token explns? | Scholar? | Effort | Key Advantage |
|:----:|--------|:----:|:-------:|:-------------:|:--------:|:------:|---------------|
| 1 | **Ditto (Attn Pooling)** | 1 | Yes | Per-token (via IG) | Very good | 1 day | Cleanest IG path; per-instance |
| 2 | **BERTScore + per-token ABTT** | 2 | Yes | Token-pair alignment | High | 2 days | Direct alignment maps |
| 3 | **Diff-in-Means Direction** | 3 | Yes | Per-token projection | High | 1 day | Answers "what is PC1?" |
| 4 | **LEACE Concept Erasure** | 3 | Yes | Per-token (via IG) | High | 2 days | Principled ABTT replacement |
| 5 | **Sinkhorn OT** | 2 | Yes | Soft transport plan | High | 3 days | Rich many-to-many alignment |
| 6 | **SAEs** | 4 | Yes | Per-feature per-token | Moderate | 5 days | Decompose PC1 into features |
| 7 | **Word Rotator's Distance** | 2 | Yes | Norm + direction | Very high | 3 days | May self-correct anisotropy |
| 8 | **Power-Mean Pooling** | 1 | Yes | Per-token (via IG) | Good | 1 day | Diversifies representation |
| 9 | **ICA** | 4 | Yes | Per-component | Low | 1 day | Quick comparison to PCA |
| 10 | **Soft-ZCA** | 1 | Yes | Moderate (via IG) | Moderate | 1 day | Tunable epsilon replaces D |

---

## Recommended Phase 13 Experiments

### Experiment 13a: PC1 Identity (1 day)
- **Method**: Correlate PC1 projections with token frequency, position, and category at each layer
- **Replaces/augments**: Understanding of what ABTT removes
- **Files to modify**: New script `scripts/run_phase13a_pc1_identity.py`
- **Data needed**: Existing `runs/phase9_bases/` embeddings + `token_probabilities` from `sif_abtt.py`
- **Success criterion**: r > 0.7 between PC1 and some interpretable property
- **Runtime**: Minutes (CPU only)

### Experiment 13b: Ditto Attention Pooling (2 days)
- **Method**: Replace SIF weights with attention diagonal in `weighted_mean_pool()`
- **Replaces**: SIF+ABTT pipeline
- **Files to modify**: `src/extract_hidden_cli.py` (add `output_attentions=True`), `src/sif_abtt.py` (add attention pooling), `src/retrieval_targets.py` (new `DittoCosSimTarget`)
- **Success criterion**: Assignment Accuracy within 2% of SIF+ABTT, with visually improved IG heatmaps
- **Runtime**: ~1 hour per model for extraction on A100

### Experiment 13c: Difference-in-Means + LEACE (2 days)
- **Method**: Learn document direction → check PC alignment → LEACE erasure of frequency concept
- **Replaces**: `EmbeddingCleaner` in `src/sif_abtt.py`
- **Files to modify**: New `src/concept_erasure.py`, modify `scripts/evaluate_vectors.py` to add LEACE method
- **Dependencies**: `pip install concept-erasure`
- **Success criterion**: Assignment Accuracy >= ABTT, with interpretable direction decomposition
- **Runtime**: Minutes (LEACE is closed-form)

### Experiment 13d: Token-Level BERTScore Retrieval (3 days)
- **Method**: Save token-level embeddings, compute BERTScore-F1 as retrieval similarity, per-token ABTT
- **Replaces**: Mean-pooled cosine similarity in `src/canon_retrieval.py`
- **Files to modify**: `src/extract_hidden_cli.py` (save full `[N, seq, dim]`), new `src/token_retrieval.py`
- **Success criterion**: Assignment Accuracy competitive with sentence-level; alignment maps validated by Latin expert
- **Runtime**: ~15 min per layer for 816K pairs on A100
- **Storage**: ~2 GB per layer

### Experiment 13e: SAE Decomposition (5 days)
- **Method**: Train SAEs on token embeddings at dip layers, identify anisotropy features, ablate selectively
- **Replaces/augments**: Understanding of anisotropy mechanism
- **Files to modify**: New `src/sae_analysis.py`, new `scripts/run_phase13e_sae.py`
- **Dependencies**: `pip install sae-lens` or custom ~50-line TopK SAE
- **Success criterion**: Identify specific SAE features correlated with PC1; feature ablation matches ABTT retrieval improvement
- **Runtime**: ~30 min per layer for training on A100

---

## References

### Area 1: Isotropy-Aware Pooling
- Chen, Y., et al. (2023). "Ditto: A Simple and Efficient Approach to Improve Sentence Embeddings." EMNLP 2023.
- Ruckle, A., et al. (2018). "Concatenated Power Mean Word Embeddings as Universal Cross-Lingual Sentence Representations." arXiv:1803.01400.
- Wendler, C., et al. (2024). "Isotropy Matters: Soft-ZCA Whitening of Embeddings for Semantic Code Search." ESANN 2025. arXiv:2411.17538.
- Huang, J., et al. (2021). "WhiteningBERT: An Easy Unsupervised Sentence Embedding Approach." Findings of EMNLP 2021.
- Su, J., et al. (2021). "Whitening Sentence Representations for Better Semantics and Faster Retrieval." arXiv:2103.15316.
- Rajaee, S. & Pilehvar, M.T. (2021). "A Cluster-based Approach for Improving Isotropy in Contextual Embedding Space." ACL 2021.
- Rajaee, S. & Pilehvar, M.T. (2022). "An Isotropy Analysis in the Multilingual BERT Embedding Space." Findings of ACL 2022.
- Li, B., et al. (2020). "On the Sentence Embeddings from Pre-trained Language Models." EMNLP 2020.
- Gao, T., et al. (2021). "SimCSE: Simple Contrastive Learning of Sentence Embeddings." EMNLP 2021.

### Area 2: Token Alignment / Matching
- Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.
- Kusner, M., et al. (2015). "From Word Embeddings to Document Distances." ICML 2015.
- Zhao, W., et al. (2019). "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance." EMNLP 2019.
- Yokoi, S., et al. (2020). "Word Rotator's Distance." EMNLP 2020.
- Cuturi, M. (2013). "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS 2013.
- Clark, E., et al. (2019). "Sentence Mover's Similarity: Automatic Evaluation for Multi-Sentence Texts." ACL 2019.
- Khattab, O. & Zaharia, M. (2020). "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT." SIGIR 2020.
- Sato, R., et al. (2022). "Re-evaluating Word Mover's Distance." ICML 2022.
- Atasu, K., et al. (2017). "Linear-Complexity Relaxed Word Mover's Distance with GPU Acceleration." arXiv:1711.07227.

### Area 3: Representation Engineering / Steering
- Zou, A., et al. (2023). "Representation Engineering: A Top-Down Approach to AI Transparency." arXiv:2310.01405.
- Belrose, N., et al. (2023). "LEACE: Perfect Linear Concept Erasure in Closed Form." ICML 2023.
- Ravfogel, S., et al. (2020). "Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection." ACL 2020.
- Marks, S. & Tegmark, M. (2023). "The Geometry of Truth: Emergent Linear Structure in Large Language Model Representations." arXiv:2310.06824.
- Li, K., et al. (2024). "Inference-Time Intervention: Eliciting Truthful Answers from a Language Model." NeurIPS 2024.
- Timkey, W. & van Schijndel, M. (2021). "All Bark and No Bite: Rogue Dimensions in Transformer Language Models Obscure Representational Quality." EMNLP 2021.
- Bis, D., et al. (2021). "Too Much in Common: Shifting of Embeddings in Transformer Language Models and its Implications." NAACL 2021.
- Mu, J. & Viswanath, P. (2018). "All-but-the-Top: Simple and Effective Postprocessing for Word Representations." ICLR 2018.

### Area 4: Disentangled Representations
- Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models." arXiv:2309.08600.
- Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Anthropic.
- Templeton, A., et al. (2024). "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic.
- Yun, Z., et al. (2021). "Transformer Visualization via Dictionary Learning: Contextualized Embedding as a Linear Superposition of Transformer Factors." arXiv:2103.15949.
- Hewitt, J. & Manning, C.D. (2019). "A Structural Probe for Finding Syntax in Word Representations." NAACL 2019.
- Park, K., et al. (2023). "The Linear Representation Hypothesis and the Geometry of Large Language Models." arXiv:2311.03658.
- Nanda, N., et al. (2023). "Emergent Linear Representations in World Models of Self-Supervised Sequence Models." arXiv:2309.00941.
- Engels, J., et al. (2024). "Not All Language Model Features Are Linear." arXiv:2405.14860.

### Area 5: CKA / Representation Similarity
- Kornblith, S., et al. (2019). "Similarity of Neural Network Representations Revisited." ICML 2019.
- Raghu, M., et al. (2017). "SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability." NeurIPS 2017.
- Morcos, A.S., et al. (2018). "Insights on Representational Similarity in Neural Networks with Canonical Correlation." NeurIPS 2018.
- Kriegeskorte, N., et al. (2008). "Representational Similarity Analysis — Connecting the Branches of Systems Neuroscience." Frontiers in Systems Neuroscience.
- Davari, M., et al. (2023). "On the Inadequacy of CKA as a Measure of Similarity in Deep Learning." ICLR 2023 Workshop.
