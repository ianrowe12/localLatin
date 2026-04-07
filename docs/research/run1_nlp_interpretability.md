# NLP Methods for Interpretable Token-to-Token Attribution Maps

**Date:** 2026-03-25
**Context:** Our IG + ABTT pair matrices (`cos(token_q, token_c) * sqrt(|ig_q| * |ig_c|) * sign`) produce denoised heatmaps but the token-to-token connections don't make linguistic/semantic sense to Latin scholars. This document surveys methods that could replace or complement ABTT/SIF to yield human-interpretable word-level connections.

**Current pipeline:** Mean-pool token embeddings with SIF weighting -> ABTT (remove top-D PCs) -> IG via Captum `LayerIntegratedGradients` -> pair matrix in `scripts/run_phase12e_visualize.py:82-93`.

---

## Table of Contents

1. [Sparse Representations](#1-sparse-representations)
2. [Optimal Transport / Earth Mover's Distance](#2-optimal-transport--earth-movers-distance)
3. [Mutual Information and PMI](#3-mutual-information-and-pmi)
4. [Contrastive Explanations](#4-contrastive-explanations)
5. [Cross-Category Synthesis & Recommendations](#5-cross-category-synthesis--recommendations)

---

## 1. Sparse Representations

### 1.1 Sparse Autoencoders (SAEs) for Mechanistic Interpretability

**Paper:** Bricken, T., Templeton, A., Batson, J., et al. "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning." Transformer Circuits Thread, Anthropic, 2023.
**Follow-up:** "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet." Anthropic, 2024.

**Core idea:** Train a sparse autoencoder (encoder W_enc, decoder W_dec, with L1 sparsity penalty) on the residual stream activations of a frozen transformer. The SAE learns an overcomplete dictionary (e.g., 16x expansion: 768-dim to 12,288 features) where each "feature direction" ideally corresponds to one interpretable concept (monosemantic). Human raters found ~70% of features mapped cleanly to single nameable concepts.

**Application to our Latin retrieval maps:** Instead of computing `cos(token_q_hidden, token_c_hidden)` in the raw (or ABTT-corrected) hidden space, train an SAE on per-token hidden states from LaTa/PhilTa/LaBSE. Each token activates a sparse set of interpretable features. The pair matrix becomes `cos(SAE_encode(token_q), SAE_encode(token_c))` -- each nonzero dimension has a nameable meaning (e.g., "verb morphology", "religious vocabulary", "manuscript formula").

**Frozen models:** Yes -- explicitly a post-hoc technique.

**Implementation:** Medium. Libraries: **SAELens** (`pip install sae-lens`), **EleutherAI/sparsify** (`pip install sparsify`, uses TopK activation). For our 768-dim hidden states and ~1,278 documents, training is very feasible on a single A100 (minutes to hours).

**Interpretable token-to-token maps:** Yes. Each feature dimension can be labeled (automated via LLM-based auto-interpretability or manual inspection for Latin-specific features).

**Next steps:**
1. Collect per-token hidden states (pre-ABTT) from `runs/phase9_bases/` at the optimal layer
2. Install `sparsify`; train a k-sparse SAE with 4x-16x expansion
3. Replace dense `cos(h_q, h_c)` with `cos(SAE(h_q), SAE(h_c))`
4. Inspect top-activating tokens per feature to label features

---

### 1.2 k-Sparse Autoencoders (OpenAI Scaling)

**Paper:** Gao, L., Dupre la Tour, T., et al. "Scaling and evaluating sparse autoencoders." ICLR 2025 (Oral).

**Core idea:** Uses TopK activation instead of L1 penalty to directly control sparsity level. Trained a 16M-latent SAE on GPT-4 activations. Clean scaling laws with respect to autoencoder size and sparsity.

**Application:** The k-sparse variant lets us set exactly how many features each token activates (e.g., k=10), making the pair matrix guaranteed sparse. Implemented in EleutherAI/sparsify.

---

### 1.3 SAEs for Dense Retrieval (Directly Relevant)

**Paper:** Kang, H., Wang, T., Xiong, C. "Interpret and Control Dense Retrieval with Sparse Latent Features." NAACL 2025 (Short Paper).

**Core idea:** Trains SAEs on dense retrieval embeddings using a **retrieval-oriented contrastive loss** (KL divergence between original and reconstructed similarity distributions). The sparse latent features retain nearly the same retrieval accuracy while being interpretable and controllable.

**Application:** Most directly applicable paper. Their contrastive loss ensures SAE features remain faithful to retrieval. We could train this on our SIF+ABTT embeddings, preserving retrieval performance while making the feature space interpretable.

**Implementation:** Low-medium. Standard SAE + contrastive loss term, ~200 lines of PyTorch.

---

### 1.4 Disentangling Dense Embeddings with SAEs

**Paper:** O'Neill, C., Ye, C., Iyer, K., Wu, J.F. "Disentangling Dense Embeddings with Sparse Autoencoders." arXiv, 2024.

**Core idea:** Applies SAEs to dense text embeddings from 420,000+ scientific paper abstracts. Introduces "feature families" -- groups of related features at different abstraction levels. Demonstrates interpretable features while maintaining semantic fidelity.

**Application:** Directly analogous -- they apply SAEs to sentence embeddings for search/retrieval. Our corpus is smaller but the approach transfers directly.

---

### 1.5 Dictionary Learning: Transformer Visualization via Transformer Factors

**Paper:** Yun, Z., Chen, Y., Olshausen, B.A., LeCun, Y. "Transformer visualization via dictionary learning: contextualized embedding as a linear superposition of transformer factors." DeeLIO Workshop @ NAACL, 2021.

**Core idea:** Decomposes contextualized embeddings as linear superpositions of learned "transformer factors" using classical sparse coding. Demonstrates hierarchical semantic structures: word-level polysemy disambiguation, sentence-level pattern formation, and long-range dependency tracking. Code: `github.com/zeyuyun1/TransformerVis`.

**Application:** Each token's hidden state is expressed as a sparse linear combination of dictionary atoms. The pair matrix would show which factors two tokens share. Hierarchical structure (word-level to long-range) maps well to what Latin scholars want to see.

**Implementation:** Low. `sklearn.decomposition.DictionaryLearning` or `SparseCoder`. No GPU needed.

**Interpretable maps:** Yes. Each dictionary atom captures a linguistic pattern. Token similarity decomposes into shared atom activations.

**Next steps:**
1. Collect per-token hidden states at optimal layers
2. Run `sklearn.decomposition.DictionaryLearning(n_components=K, alpha=sparsity)` where K is 2-4x hidden dim
3. Encode each token as sparse coefficients; visualize shared atoms between matched pairs

---

### 1.6 DB-KSVD: Scalable Dictionary Learning

**Paper:** Valentin, R., Katz, S.M., Vanhoucke, V., Kochenderfer, M.J. "DB-KSVD: Scalable Alternating Optimization for Disentangling High-Dimensional Embedding Spaces." arXiv, Stanford AI Lab, 2025.

**Core idea:** Modernizes K-SVD for transformer-scale data. Achieves **10,000x speedup** over naive K-SVD. Matches SAE performance on SAEBench interpretability benchmarks when applied to Gemma-2-2B embeddings.

**Application:** No neural network training needed -- pure alternating optimization. For our ~1,278 docs x ~100 tokens/doc at 768-dim, DB-KSVD would finish in seconds.

**Implementation:** Low. Available as Julia `KSVD.jl` via `juliacall`, or scikit-learn's `DictionaryLearning` (slower but pure Python).

---

### 1.7 Sparse Probing / Concept Erasure

**Paper:** Gurnee, W., Nanda, N. "Finding Neurons in a Haystack: Case Studies with Sparse Probing." NeurIPS 2023.

**Core idea:** Trains k-sparse linear classifiers on internal activations to predict human-defined features. By varying k, studies how sparsely features are represented.

**Application:** If we define Latin concepts (noun case, verb tense, religious terminology), we could train sparse probes to find which hidden dimensions encode them. The pair matrix could be reweighted to emphasize linguistically meaningful dimensions.

**Implementation:** Low. L1-regularized `sklearn.linear_model.LogisticRegression`. Challenge: requires ~100 annotated tokens per concept from a Latin scholar.

---

### 1.8 LEACE: Perfect Linear Concept Erasure

**Paper:** Belrose, N., et al. "LEACE: Perfect linear concept erasure in closed form." NeurIPS 2023.
**Related:** Ravfogel, S., et al. "Linear Adversarial Concept Erasure (RLACE)." ICML 2022.

**Core idea:** Removes a target concept from representations via closed-form linear projection. Unlike ABTT (which removes dominant PCs that may or may not be meaningful), LEACE removes *specific named concepts*.

**Application:** Use LEACE to subtract out unwanted confounds (document length, position, function words) from token hidden states before computing the pair matrix. Complementary to ABTT.

**Implementation:** Very low. `pip install concept-erasure` (EleutherAI). Needs labeled data for the concept to erase.

---

### 1.9 Non-Negative Matrix Factorization (Semi-NMF)

**Papers:** Lee & Seung, "Learning the parts of objects by non-negative matrix factorization." Nature, 1999. Ding, Li, Jordan, "Convex and Semi-Nonnegative Matrix Factorizations." IEEE TPAMI, 2010.

**Core idea:** NMF produces "parts-based" decomposition. Semi-NMF extends to mixed-sign data (like transformer hidden states): basis vectors W can have mixed signs, but coefficients H must be non-negative.

**Application:** The pair matrix becomes `H_q^T * H_c` -- a non-negative inner product in parts space. Because NMF produces parts-based decomposition, each component captures a localized, meaningful pattern. No cancellation due to non-negative coefficients.

**Implementation:** Very low. `sklearn.decomposition.NMF` for non-negative data; `nimfa` package for Semi-NMF. No GPU needed.

---

### 1.10 Post-Hoc Concept Bottleneck Models

**Paper:** Yuksekgonul, M., Wang, M., Zou, J. "Post-hoc Concept Bottleneck Models." ICLR 2023.

**Core idea:** Converts any pretrained black-box model into an interpretable concept bottleneck model *after training*. Projects frozen embeddings onto concept directions derived from text descriptions or labeled examples.

**Application:** Define Latin philological concepts (liturgical text, legal formula, classical prose style, late Latin vocabulary) and create concept embeddings. The pair matrix becomes `concept_scores(q)^T * concept_scores(c)` -- showing which *named philological concepts* two tokens share. **Maximally interpretable to scholars** because each dimension is explicitly named.

**Implementation:** Low-medium. Projection is just dot products. Challenge: defining the concept set for Latin philology (20-50 concepts with examples).

---

### 1.11 SPINE: Sparse Interpretable Neural Embeddings

**Paper:** Subramanian, A., Pruthi, D., Jhamtani, H., Berg-Kirkpatrick, T., Hovy, E. "SPINE: SParse Interpretable Neural Embeddings." AAAI 2018. Code: `github.com/harsh19/SPINE`.

**Core idea:** Denoising k-sparse autoencoder converting dense embeddings into sparse, interpretable, non-negative embeddings. Human evaluation shows much higher interpretability than original embeddings.

**Application:** Apply to per-token hidden states for non-negative sparse representations where each dimension is interpretable. Non-negativity + sparsity is attractive for pair matrices (additive, no cancellation).

---

### Sparse Representations: Summary Table

| Method | Interpretability | Labeled Data? | Implementation | GPU? | Best For |
|--------|-----------------|---------------|----------------|------|----------|
| SAE (k-sparse) | High (auto-discovered) | No | Medium (library) | Recommended | Automatic feature discovery |
| SAE + retrieval loss | High + retrieval-faithful | No | Medium | Yes | Preserving retrieval performance |
| DB-KSVD | High (matches SAE) | No | Low | No | Fast prototyping |
| Dict. Learning (Yun) | High (transformer factors) | No | Low (sklearn) | No | Quick baseline |
| Sparse Probing | Targeted | Yes (~100/concept) | Low | No | Testing specific hypotheses |
| LEACE | Indirect (erasure) | Yes (confounds) | Very Low | No | Removing known confounds |
| Semi-NMF | Medium-High (parts) | No | Low | No | Parts-based decomposition |
| Post-hoc CBM | Maximal (named) | Yes (concepts) | Low-Medium | No | Scholar-facing explanations |
| SPINE | High (sparse+non-neg) | No | Low | Optional | Non-negative interpretable reps |

---

## 2. Optimal Transport / Earth Mover's Distance

### 2.1 BERTScore: Greedy Token Alignment

**Paper:** Zhang, T., Kishore, V., Wu, F., Weinberger, K.Q., Artzi, Y. "BERTScore: Evaluating Text Generation with BERT." ICLR 2020.

**Core idea:** Compute pairwise cosine similarity matrix between all token embeddings. Use greedy maximum matching: for each query token, pick the candidate token with highest cosine similarity (recall), and vice versa (precision). Alignment is one-to-one per direction.

**Application:** We already compute the cosine matrix. BERTScore's greedy alignment extracts `argmax` per row/column, giving a sparse binary alignment overlay. Each token maps to exactly one partner.

**Implementation:** Zero new dependencies. Just `np.argmax(cos, axis=1)` on our existing cosine matrix.

**Interpretable maps:** Yes -- sparse alignment where each token has exactly one partner. Visualize as connecting lines overlaid on the dense heatmap.

**Limitation:** Greedy matching is one-to-one but not globally optimal. No mass conservation.

---

### 2.2 Word Mover's Distance (WMD)

**Paper:** Kusner, M.J., Sun, Y., Kolkin, N.I., Weinberger, K.Q. "From Word Embeddings to Document Distances." ICML 2015.

**Core idea:** Treat each document as a weighted point cloud in embedding space. Distance = Earth Mover's Distance -- minimum cost to transport mass of one distribution to the other. The optimal transport plan T[i,j] tells you how much "mass" moves from token i to token j.

**Application:** Each manuscript fragment is already a set of token embeddings with SIF weights. The transport plan T is sparse and satisfies mass conservation. It answers "which query tokens explain which candidate tokens?" with a globally optimal solution.

**Implementation:** Already available via `ot.emd()` in our installed POT library.

```python
import ot
C = 1.0 - cosine_matrix(q_hidden, c_hidden)  # cost matrix
C = np.clip(C, 0, None)
a = np.abs(q_ig) / np.abs(q_ig).sum()  # normalized IG weights
b = np.abs(c_ig) / np.abs(c_ig).sum()
T = ot.emd(a, b, C)  # transport plan matrix
```

**Interpretable maps:** Yes. The transport plan is naturally sparse. Complexity: O(n^3 log n) but for < 512 tokens, runs in milliseconds.

---

### 2.3 MoverScore: WMD with Contextual Embeddings

**Paper:** Zhao, W., Peyrard, M., Liu, F., Gao, Y., Meyer, C.M., Eger, S. "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance." EMNLP 2019.

**Core idea:** Extends WMD to contextual BERT/ELMo embeddings. Uses soft many-to-one alignments and IDF-based token weights (parallels our SIF weighting).

**Application:** Directly applicable -- we already have contextual hidden states. One query token can partially match multiple candidate tokens, which is realistic for Latin (one word form may correspond to a multi-word phrase due to morphological differences).

**Implementation:** Same as WMD recipe using POT + our layer-specific hidden states.

---

### 2.4 Sinkhorn Divergence: Regularized OT

**Paper:** Cuturi, M. "Sinkhorn Distances: Lightspeed Computation of Optimal Transport." NeurIPS 2013.

**Core idea:** Add entropic regularization to OT for smoother transport plans and faster computation. O(n^2) per iteration, converges in 50-100 iterations.

**Application:** Denser plan than exact EMD but mass-conserving and differentiable. Small epsilon gives near-EMD sparsity. Use Sinkhorn when exact EMD is too slow for long sequences (Qwen3-8B).

**Implementation:** `ot.sinkhorn(a, b, C, reg=0.05)`.

**Interpretable maps:** Moderate -- plan is smooth rather than sparse. Threshold small values to recover sparsity.

---

### 2.5 Unbalanced Optimal Transport for Word Alignment

**Paper:** Arase, Y., Bao, H., Yokoi, S. "Unbalanced Optimal Transport for Unbalanced Word Alignment." ACL 2023. Code: `github.com/yukiar/OTAlign`.

**Core idea:** Standard OT requires all mass to be transported. But in real text pairs, some tokens have no counterpart (null alignment). Unbalanced OT relaxes this: tokens can be "left behind" at a penalty cost.

**Application:** **Critical for our use case.** Latin manuscript fragments from different witnesses will have insertions, deletions, and paraphrases. Forcing every token to match produces spurious alignments. Unbalanced OT lets some tokens go unmatched ("unique to this fragment") -- linguistically meaningful and directly interpretable.

**Implementation:**
```python
T_unb = ot.unbalanced.sinkhorn_unbalanced(a, b, C, reg=0.05, reg_m=1.0)
# Or partial OT:
T_partial = ot.partial.partial_wasserstein(a, b, C, m=0.8)  # transport 80% of mass
```

**Interpretable maps:** Excellent. Shows which tokens match AND which are unmatched (unique content highlighted differently).

---

### 2.6 Fused Gromov-Wasserstein (FGW): Embedding + Position Matching

**Paper:** Vayer, T., Chapel, L., Flamary, R., Tavenard, R., Courty, N. "Fused Gromov-Wasserstein Distance for Structured Objects." Algorithms 2020 (extended from ICML 2019 workshop).

**Core idea:** Combines Wasserstein distance (matches tokens by embedding similarity) with Gromov-Wasserstein distance (preserves structural relationships, e.g., positional order). Parameter alpha controls the trade-off.

**Application:** **Most promising OT method for our case.** It can match tokens based on both semantic similarity AND sequential position. Latin scholars expect tokens in the same order to be aligned, even if embeddings don't match perfectly. The transport plan preserves both semantic and positional coherence.

**Implementation:**
```python
import ot
M = np.clip(1.0 - cosine_matrix(q_hidden, c_hidden), 0, None)  # feature cost
C1 = np.abs(np.arange(n_q)[:, None] - np.arange(n_q)[None, :]).astype(float)  # position structure
C2 = np.abs(np.arange(n_c)[:, None] - np.arange(n_c)[None, :]).astype(float)
C1 /= C1.max(); C2 /= C2.max(); M /= M.max()
a = np.abs(q_ig) / np.abs(q_ig).sum()
b = np.abs(c_ig) / np.abs(c_ig).sum()
T_fgw = ot.gromov.fused_gromov_wasserstein(M, C1, C2, a, b, loss_fun='square_loss', alpha=0.5)
```

**Interpretable maps:** Excellent. The plan respects word order (monotonically increasing, with crossings for reordering) -- looks like what a scholar would draw by hand.

---

### 2.7 WSMD: WMD + Self-Attention Structure

**Paper:** Yamagiwa, H., Yokoi, S., Shimodaira, H. "Improving Word Mover's Distance by Leveraging Self-Attention Matrix." EMNLP 2023 Findings. Code: `github.com/ymgw55/WSMD`.

**Core idea:** Uses Fused Gromov-Wasserstein where structure matrices are **self-attention matrices** from BERT, not positional distances. Self-attention captures which tokens attend to which other tokens, encoding syntactic/semantic structure. FGW then matches tokens that are both semantically similar and play similar structural roles.

**Application:** Highly relevant for T5 models (LaTa, PhilTa) where we can extract self-attention. The attention structure encodes syntactic relationships that positional distance alone misses. Directly addresses "denoised but not linguistically meaningful."

**Implementation:** Combine existing hidden state extraction with attention extraction (`output_attentions=True`), then use POT's FGW with attention matrices as structure.

---

### 2.8 Sparse OT for Rationalization

**Paper:** Swanson, K., Yu, L., Lei, T. "Rationalizing Text Matching: Learning Sparse Alignments via Optimal Transport." ACL 2020.

**Core idea:** Introduces constrained OT variants with provable sparsity bounds. The sparse transport plan serves as a "rationale" explaining which token pairs justify the similarity judgment.

**Application:** The sparsity constraint directly addresses scholars wanting few strong connections, not a dense heatmap. Best realized via `ot.emd()` (naturally sparse) or `ot.partial.partial_wasserstein()` (controlled sparsity).

---

### Optimal Transport: Summary Table

| Method | Sparsity | Position Aware | Handles Unmatched | POT Function | Interpretability |
|--------|----------|---------------|-------------------|-------------|-----------------|
| BERTScore greedy | Binary (1-to-1) | No | No | `np.argmax` | High |
| WMD / EMD | Naturally sparse | No | No | `ot.emd()` | High |
| MoverScore | Naturally sparse | No | No | `ot.emd()` | High |
| Sinkhorn | Dense (tunable) | No | No | `ot.sinkhorn()` | Medium |
| Unbalanced OT | Sparse + nulls | No | Yes | `ot.unbalanced.sinkhorn_unbalanced()` | Excellent |
| FGW (positional) | Sparse + ordered | Yes | No | `ot.gromov.fused_gromov_wasserstein()` | Excellent |
| FGW (attention/WSMD) | Sparse + structural | Yes (syntactic) | No | `ot.gromov.fused_gromov_wasserstein()` | Excellent |

---

## 3. Mutual Information and PMI

### 3.1 Corpus PPMI (Pointwise Mutual Information)

**Paper:** Church, K.W. & Hanks, P. "Word Association Norms, Mutual Information, and Lexicography." Computational Linguistics, 16(1), 1990.

**Core idea:** PMI(x, y) = log(P(x,y) / P(x)P(y)) measures how much more often two tokens co-occur than chance predicts. PPMI clamps negatives to zero.

**Application:** Compute co-occurrence counts from all canon/ texts. The PPMI score for each (query_token, candidate_token) pair tells you whether those Latin words tend to appear together in the same textual tradition more than chance. Inherently interpretable: "these words appear together 8x more than expected."

**Implementation:** Trivial. Count co-occurrences, compute marginals, take log ratio. `sklearn.feature_extraction.text.CountVectorizer` or raw counting. No GPU needed.

**Next steps:**
1. Build co-occurrence matrix from 1,278 canon files (document-level or windowed)
2. Compute PPMI matrix
3. Replace `sqrt(|ig_q| * |ig_c|)` with `PPMI(token_q, token_c)` in pair matrix
4. Compare heatmaps with scholars on same retrieval pairs

---

### 3.2 Contextual PMI via Masked Language Models

**Paper:** Ghosh, S., Kim, Y., et al. "Alignment via Mutual Information." CoNLL 2023, pp. 488-497.

**Core idea:** Use a masked sequence model to compute conditional PMI between source and target spans: PMI(span_s, span_t | context). Captures context-dependent associations that static PMI misses.

**Application:** Use LaTa's encoder in MLM mode to estimate P(token_q | context_q, token_c present) vs P(token_q | context_q). Captures context-dependent associations.

**Implementation:** Moderate. Requires O(N*M) forward passes per pair (expensive but feasible for the small number of pairs we visualize).

---

### 3.3 MINE (Mutual Information Neural Estimation)

**Paper:** Belghazi, M.I., et al. "Mutual Information Neural Estimation." ICML 2018.

**Core idea:** Trains a small statistics network to estimate MI between two continuous random variables using the Donsker-Varadhan representation.

**Application:** Estimate MI(h_q^i, h_c^j) between hidden-state vectors of token pairs. High MI means knowing one token's representation is highly informative about the other's. Replaces cosine similarity with a more general dependency measure.

**Implementation:** Multiple PyTorch implementations (`gtegner/mine-pytorch`, `mfederici/torch-mist`). Challenge: MINE requires many samples; we have only 1,278 documents.

**Interpretable maps:** Moderate. MI values indicate dependency strength but not *what* the dependency is.

---

### 3.4 Information-Theoretic Probing

**Paper:** Pimentel, T., et al. "Information-Theoretic Probing for Linguistic Structure." ACL 2020, pp. 4609-4622.

**Core idea:** Reconceptualize probing as estimating MI(R; L) between representations R and linguistic labels L. Use the most powerful probe possible for a tighter lower bound on MI.

**Application:** Train a probe to estimate how much information each token's hidden state carries about its manuscript identity (directory label). Tokens with high MI are "diagnostic" tokens. Weight pair matrix by per-token MI scores rather than IG scores.

**Implementation:** Moderate. MLP probe per layer, standard PyTorch.

---

### 3.5 CKA (Centered Kernel Alignment)

**Paper:** Kornblith, S., Norouzi, M., Lee, H., Hinton, G. "Similarity of Neural Network Representations Revisited." ICML 2019.

**Core idea:** CKA measures similarity between two sets of representations using HSIC. Invariant to isotropic scaling and orthogonal transformations.

**Application:** (a) **Layer selection:** pick the layer whose representations best align with manuscript-family gold structure. (b) **Token-level CKA decomposition:** per-token contributions to HSIC, though inherently a set-level measure.

**Implementation:** Low. `pip install centered-kernel-alignment`. O(n^2 * d).

---

### 3.6 Cross-RSA (Representational Similarity Analysis)

**Paper:** Kriegeskorte, N., Mur, M., Bandettini, P. "Representational Similarity Analysis." Frontiers in Systems Neuroscience, 2:4, 2008.
**NLP application:** Abnar, S., et al. "Blackbox Meets Blackbox." BlackBoxNLP @ ACL, 2019.

**Core idea:** Build Representational Dissimilarity Matrices (RDMs) capturing pairwise distances between stimulus representations. Compare RDMs across systems using Spearman correlation.

**Application:** Build within-document RDMs for query and candidate tokens. Compute **cross-RSA**: correlate each query token's similarity profile with each candidate token's profile. Tokens playing analogous structural roles get high cross-RSA scores.

**Implementation:** Low. `pip install rsatoolbox`. CPU only, fast.

**Interpretable maps:** High for structural correspondence ("token A in query plays the same role as token B in candidate").

**Next steps:**
1. For each retrieval pair, extract token-level hidden states
2. Build within-document RDMs
3. Compute cross-RSA: correlate similarity profiles
4. Use cross-RSA scores as pair matrix weights

---

### 3.7 Partial Information Decomposition (PID)

**Paper:** Dewan, S., et al. "DiffusionPID: Interpreting Diffusion via Partial Information Decomposition." NeurIPS 2024.

**Core idea:** Decompose information from two input tokens about an output into: *unique* from token A, *unique* from B, *redundant*, and *synergistic*. Token pairs that are synergistic highlight genuine cross-document connections.

**Application:** Principled replacement for `sqrt(|ig_q| * |ig_c|)`. Synergistic pairs = tokens whose combined effect on retrieval exceeds the sum of individual effects.

**Implementation:** High complexity. PID estimation for continuous variables is an active research area. No off-the-shelf library yet.

---

### 3.8 PPMI-SVD as Interpretable Baseline

**Papers:** Levy, O. & Goldberg, Y. "Neural Word Embedding as Implicit Matrix Factorization." NeurIPS 2014.
Arora, S., et al. "A Latent Variable Model Approach to PMI-based Word Embeddings." TACL, 2016.

**Core idea:** Our SIF + ABTT pipeline has a direct theoretical connection to PMI. Arora et al. (2017) derived SIF weighting and common component removal (precursor to ABTT) from a latent variable model that implicitly factorizes a PMI matrix. Making this explicit could help interpretability.

**Application:** Build PPMI-SVD embeddings from Latin corpus as a fully interpretable baseline. If PPMI-SVD pair scores already align with scholar judgments, the neural model's contribution is geometric (anisotropy correction) rather than semantic.

**Implementation:** Trivial. `scipy.sparse` + `scipy.sparse.linalg.svds`. Minutes on CPU.

---

### 3.9 Contextual Decomposition for Transformers (CD-T)

**Paper:** Hsu, A., et al. "Mechanistic Interpretation through Contextual Decomposition in Transformers." arXiv:2407.00886, 2024.
**Related:** Murdoch, W.J., Liu, P.J., Yu, B. "Beyond Word Importance: Contextual Decomposition to Extract Interactions from LSTMs." ICLR 2018.

**Core idea:** Analytically decompose each activation into "relevant" (beta) and "irrelevant" (gamma) components, propagated through attention and FFN layers. No perturbation needed -- traces how each token's contribution flows through the network.

**Application:** Could replace IG entirely. Avoids known issues with IG baselines (zero embedding is not meaningful for transformers). Analytically traces contribution flow in a single forward pass.

**Implementation:** Moderate. Requires implementing decomposition rules for each layer type (attention, FFN, LayerNorm). ~2x speedup over path patching. No off-the-shelf library for arbitrary architectures yet.

---

### MI/PMI: Summary Table

| Method | Token-to-Token? | Frozen Model? | Interpretability | Complexity | GPU? |
|--------|-----------------|---------------|-----------------|-----------|------|
| Corpus PPMI | Yes (direct) | Model-free | Excellent | Trivial | No |
| Contextual PMI | Yes (direct) | Yes | High | Moderate | Yes |
| MINE | Yes (pairwise MI) | Yes | Moderate | Moderate | Helpful |
| IT Probing | Per-token, not pairwise | Yes | Good | Moderate | No |
| CKA | Set-level | Yes | Moderate | Low | No |
| Cross-RSA | Yes (structural) | Yes | High | Low | No |
| PID | Yes (synergy) | Yes | Excellent (theory) | High | Helpful |
| PPMI-SVD baseline | Yes (direct) | Model-free | Excellent | Trivial | No |
| CD-T | Per-token, decomposable | Yes | High | Moderate | No |

---

## 4. Contrastive Explanations

### 4.1 Contrastive IG

**Paper:** Yin, K. & Neubig, G. "Interpreting Language Models with Contrastive Explanations." EMNLP 2022.

**Core idea:** Instead of "which tokens matter for this prediction?", ask "which tokens explain why the model predicted A *instead of* B?" Computes contrastive gradient norm: Frobenius norm of gradient of (log p(target) - log p(foil)) w.r.t. token embeddings.

**Application:** Redefine scalar target as `cos_sim(query, true_match) - cos_sim(query, hard_negative)`. IG on this contrastive target highlights tokens that specifically explain why fragment A retrieves B rather than its nearest incorrect neighbor C.

**Implementation:** Low. Create a `ContrastiveCosSimTarget` in `retrieval_targets.py` returning `cos_sim(query, partner) - cos_sim(query, foil)`. Use existing `LayerIntegratedGradients`.

**Next steps:**
1. Add `ContrastiveCosSimTarget(nn.Module)` to `retrieval_targets.py`
2. Select foils as top-1 incorrect retrieval result per query
3. Compare contrastive vs. non-contrastive IG attributions on Phase 12e pairs
4. Evaluate with Latin scholars

---

### 4.2 Integrated Jacobians for Siamese Encoders (MOST RELEVANT)

**Paper:** Moeller, L., Nikolaev, D., Pado, S. "An Attribution Method for Siamese Encoders." EMNLP 2023, pp. 15818-15827.

**Core idea:** Generalizes IG to models with *two* inputs by computing the Jacobian of the similarity function w.r.t. both inputs simultaneously, integrating along a path from baseline to actual inputs. Output is a **token-pair attribution matrix** (not two independent per-token vectors). Inherits IG's formal guarantees: completeness (attributions sum to the actual similarity score).

**Application:** **This is the most natural replacement for our current approach.** Instead of computing IG separately for query and candidate then multiplying (`cos * sqrt(|ig_q| * |ig_c|) * sign`), Integrated Jacobians directly produce the cross-sequence token-token interaction matrix. Each cell (i,j) tells you how much the interaction between query token i and candidate token j contributes to the final cosine similarity. Theoretically grounded, not an ad-hoc product.

**Implementation:** Medium. Not in Captum. Loop over embedding dimensions: for each dim d, compute IG of the d-th dimension of pooled_query w.r.t. candidate tokens. ~2 minutes per pair on a P100 for dim-768.

**Token-to-token maps:** Yes -- this is exactly what it produces. The matrix is guaranteed to sum to the actual similarity score.

**Next steps:**
1. Implement `integrated_jacobian()` looping over embedding dimensions
2. For each dim d, compute `d(sim)/d(query_token_i) * d(sim)/d(cand_token_j)` along integration path
3. Test on 5-10 Phase 12e pairs; compare to current IG-product maps
4. Budget: ~2 min/pair on A100 for 768-dim models (LaTa, PhilTa, LaBSE)

---

### 4.3 BiLRP (Bilinear Layer-wise Relevance Propagation)

**Paper:** Vasileiou, A. & Eberle, O. "Explaining Text Similarity in Transformer Models." NAACL 2024. Code: `github.com/oeberle/BiLRP_explain_similarity`.

**Core idea:** Extends LRP to bilinear similarity functions (dot product / cosine). Decomposes similarity into contributions from *pairs* of features across two inputs. Tested on BERT, SBERT, multilingual BERT. Validated on biomedical text retrieval (specialized vocabulary like Latin).

**Application:** Directly produces token-token relevance matrices. Signed scores show which token pairs push similarity up or down. Also showed corpus-level aggregation by POS tags.

**Implementation:** Medium-High. Code available but requires LRP propagation rules specific to each layer type. Would need adaptation for T5 architecture. ~2 minutes per pair for dim-768.

---

### 4.4 SHAP / KernelSHAP for Token Importance

**Paper:** Lundberg, S. & Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017.
**NLP extensions:** TokenSHAP (Gold et al., arXiv 2407.10114, 2024); TokenShapley (arXiv 2507.05261, ACL Findings 2025).

**Core idea:** Treat each token as a player in a cooperative game. Shapley values assign each token a fair share of the total similarity score by considering all possible subsets.

**Application:** Theoretically principled per-token importance accounting for interactions (unlike IG's single integration path).

**Implementation:** Medium. Captum provides `KernelShap` and `ShapleyValueSampling`. Exact Shapley is 2^n (impossible for >20 tokens); KernelSHAP samples ~2000 perturbations.

**Token-to-token maps:** Not directly -- gives per-token importance for one sequence. Same limitation as current IG.

---

### 4.5 Leave-One-Out / Occlusion / Feature Ablation

**Paper:** Li, J., Monroe, W., Jurafsky, D. "Understanding Neural Networks through Representation Erasure." arXiv 1612.08220, 2016.

**Core idea:** Mask/remove one token at a time, re-run forward pass, measure similarity change. For interaction matrix: mask *both* query token i and candidate token j, measure interaction effect beyond individual drops.

**Application:** The interaction variant produces a true cross-sequence matrix. Can serve as **ground truth** for validating whether IG-product or Integrated Jacobians better approximate the true interaction.

**Implementation:** Very Low. Captum's `FeatureAblation`. Individual: 50 forward passes for 50-token query. Interaction (50x50 pairs): 2500 forward passes -- feasible.

---

### 4.6 Attention Rollout

**Paper:** Abnar, S. & Zuidema, W. "Quantifying Attention Flow in Transformers." ACL 2020.

**Core idea:** Raw attention in layer L doesn't account for information mixing. Rollout recursively multiplies attention matrices: `R_L = 0.5*A_L * R_{L-1} + 0.5*I`.

**Application:** We already save attention matrices in Phase 12e. Rollout gives a single matrix mapping each output position to input tokens across all layers. Use rollout importance as alternative weighting in pair matrices.

**Implementation:** Very Low. ~10 lines of numpy.

```python
R = np.eye(seq_len)
for A in attention_layers:
    R = 0.5 * A @ R + 0.5 * np.eye(seq_len)
```

**Token-to-token maps:** Within-sequence only. Does not directly produce cross-sequence maps for encoder-only models.

---

### 4.7 Shapley Interaction Index / Archipelago

**Paper:** Tsang, M., Rambhatla, S., Liu, Y. "How does This Interaction Affect Me?" NeurIPS 2020.
**Also:** Sundararajan, M., et al. "The Shapley Taylor Interaction Index." ICML 2020.
**Library:** `shapiq` (`pip install shapiq`).

**Core idea:** Extends Shapley values to pairwise feature interactions. Measures the *synergy* between features i and j: how much their joint contribution exceeds the sum of their individual contributions.

**Application:** **Most theoretically principled cross-sequence token-token interaction maps.** Define features as all tokens from both sequences. Interaction value between query token i and candidate token j directly measures synergistic contribution to cosine similarity.

**Implementation:** High computational cost. For 100 total features, 10,000 interaction terms each requiring multiple model evaluations. `shapiq` provides efficient approximation. Archipelago's `ArchDetect` pre-screens for significant interactions.

---

### 4.8 Gradient x Input + SmoothGrad

**Paper:** Ancona, M., et al. "Towards Better Understanding of Gradient-based Attribution Methods." ICLR 2018.
**SmoothGrad:** Smilkov, D., et al. "SmoothGrad: Removing Noise by Adding Noise." ICML Workshop 2017.

**Core idea:** `attribution_i = gradient_i * input_i`. Single backward pass (vs. IG's 40-200 steps). SmoothGrad averages over N noisy copies to reduce noise. Ancona et al. showed Gradient x Input, epsilon-LRP, IG, and DeepLIFT are "strongly related" with provable equivalence conditions.

**Application:** If attribution rankings match IG (test empirically), this is 40x faster for full-corpus runs. SmoothGrad with n_samples=50 as middle ground.

**Implementation:** Very Low. Captum has `InputXGradient` and `NoiseTunnel`.

---

### 4.9 Survey Reference

**Paper:** Opitz, J., Moeller, L., Michail, A., Pado, S., Clematide, S. "Interpretable Text Embeddings and Text Similarity Explanation: A Survey." EMNLP 2025. `aclanthology.org/2025.emnlp-main.1135/`

Covers the full taxonomy: (a) inherently interpretable embeddings (sparse representations, set-based/late-interaction), (b) post-hoc explanations (interaction attribution via Integrated Jacobians and BiLRP, surrogates, probing). **Single most relevant reference for our project.**

---

### Contrastive: Summary Table

| Method | Cross-Sequence Map? | Principled? | Implementation | Speed |
|--------|-------------------|-------------|----------------|-------|
| Contrastive IG | Per-token (not pairwise) | Yes | Low | Fast (same as IG) |
| **Integrated Jacobians** | **Yes (native)** | **Yes (completeness)** | **Medium** | **~2 min/pair** |
| BiLRP | Yes (native) | Yes (conservation) | Medium-High | ~2 min/pair |
| SHAP | Per-token only | Yes (Shapley axioms) | Medium | Slow |
| LOO interaction | Yes (if mask pairs) | Yes (direct) | Very Low | O(n*m) fwd passes |
| Attention Rollout | Within-sequence only | Debated | Very Low | Near-instant |
| **Shapley Interaction** | **Yes (native)** | **Yes (game theory)** | **High** | **Very slow** |
| Grad x Input | Per-token only | Approximate | Very Low | 40x faster than IG |

---

## 5. Cross-Category Synthesis & Recommendations

### Top Recommendations by Priority

#### Tier 1: Highest Impact, Lowest Effort (Start Here)

| # | Method | Category | Why | Effort |
|---|--------|----------|-----|--------|
| 1 | **WMD via `ot.emd()`** | OT | Naturally sparse transport plan, mass-conserving, directly replaces pair matrix. Already have POT installed. | ~30 lines |
| 2 | **Contrastive IG** | Contrastive | Just change the target function to include a foil. Answers "why A not C?" | ~50 lines in `retrieval_targets.py` |
| 3 | **BERTScore greedy alignment** | OT | `np.argmax` on existing cosine matrix. Sparse overlay. | ~5 lines |
| 4 | **Corpus PPMI** | MI/PMI | Model-free, maximally interpretable, tests whether co-occurrence statistics suffice. | ~40 lines |

#### Tier 2: High Impact, Medium Effort

| # | Method | Category | Why | Effort |
|---|--------|----------|-----|--------|
| 5 | **Integrated Jacobians** | Contrastive | Theoretically principled cross-sequence token-token matrix. Replaces ad-hoc IG-product formula. | Custom implementation, ~200 lines |
| 6 | **FGW (positional)** | OT | Position-aware OT -- aligns tokens by both meaning and word order. Scholars' intuition. | ~20 lines using POT |
| 7 | **Unbalanced/Partial OT** | OT | Handles unmatched tokens (insertions/deletions in manuscripts). | ~10 lines using POT |
| 8 | **Dictionary Learning (sklearn)** | Sparse | Quick sparse decomposition, no GPU, interpretable factors. | ~30 lines |

#### Tier 3: High Impact, Higher Effort (Research Investment)

| # | Method | Category | Why | Effort |
|---|--------|----------|-----|--------|
| 9 | **SAE (k-sparse)** | Sparse | Auto-discovers interpretable features. Best for understanding *what* drives similarity. | Install sparsify, train SAE |
| 10 | **Post-hoc CBM** | Sparse | Maximally interpretable to scholars (named philological concepts). | Requires concept curation |
| 11 | **FGW + attention (WSMD)** | OT | Richest structural signal. Requires extracting attention matrices. | Modify extraction CLIs |
| 12 | **LOO interaction matrix** | Contrastive | Ground truth for validating all other methods. O(n*m) fwd passes. | ~50 lines + compute time |

### Recommended Experiment Plan

**Phase A (1-2 days):** Run Tier 1 methods on 5-10 existing Phase 12e pairs. Compare all four heatmaps side-by-side with current IG-product baseline. Have a Latin scholar rank which maps make the most sense.

**Phase B (1 week):** Implement top 2 methods from Phase A at scale. Add Integrated Jacobians and FGW. Build a comparison dashboard.

**Phase C (2-3 weeks):** Invest in SAEs or Post-hoc CBMs if sparse features prove promising. Run LOO interaction as ground truth validation.

### Key References (Must-Read)

1. **Opitz et al. 2025** -- "Interpretable Text Embeddings and Text Similarity Explanation: A Survey" (EMNLP 2025). Comprehensive taxonomy of all methods above.
2. **Moeller et al. 2023** -- "An Attribution Method for Siamese Encoders" (EMNLP 2023). Integrated Jacobians -- the principled replacement for IG-product.
3. **Vayer et al. 2020** -- "Fused Gromov-Wasserstein Distance" (Algorithms 2020). Position-aware OT alignment.
4. **Kang et al. 2024** -- "Interpret and Control Dense Retrieval with Sparse Latent Features" (NAACL 2025). SAEs specifically for retrieval.
5. **Arase et al. 2023** -- "Unbalanced OT for Word Alignment" (ACL 2023). Handles missing correspondences.
