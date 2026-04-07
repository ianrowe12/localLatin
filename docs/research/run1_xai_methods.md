# XAI Methods for Latin Manuscript Retrieval Explanation

**Date:** 2026-03-25
**Context:** Evaluating alternatives to Integrated Gradients (IG) for producing interpretable token-level explanations of embedding similarity between Latin manuscript fragments. Current pipeline: IG via Captum's `LayerIntegratedGradients` (~40-50 forward passes), pair matrix = `cosine_sim(q_token, c_token) * sqrt(|IG_q| * |IG_c|) * sign`. Problem: highlighted token connections don't correspond to what a Latin scholar would expect.

---

## Quick Comparison Table

| Method | Pairwise (q x c)? | Forward Passes | Backward Passes | Cost vs IG | Frozen Models? | Implementation |
|--------|:------------------:|:--------------:|:---------------:|:----------:|:--------------:|----------------|
| **Attention Rollout** | No (single-seq) | 1 | 0 | ~50x cheaper | Yes | Manual (easy) |
| **ALTI+** | Yes (enc-dec) | 1 | 0 | ~50x cheaper | Yes | `stopes` (Meta) |
| **Chefer ICCV 2021** | Yes (enc-dec) | 1 | 1 | ~25x cheaper | Yes | [GitHub](https://github.com/hila-chefer/Transformer-MM-Explainability) |
| **GlobEnc** | No (single-seq) | 1 | 0 | ~50x cheaper | Yes | [GitHub](https://github.com/mohsenfayyaz/GlobEnc) |
| **SHAP Interaction** | Yes | ~20K-80K | 0 | 500-2000x more | Yes (black-box) | `shapiq` |
| **KernelSHAP** | No (marginal) | ~200-2,500 | 0 | 5-50x more | Yes (black-box) | `shap` |
| **AttnLRP** | No (single-seq) | 2 | 1 | ~40x cheaper | Yes | `lxt` (pip) |
| **BiLRP** | **Yes (native)** | 1 | ~2*h (~1,536) | ~30x more | Yes | [GitHub](https://github.com/alevas/xai_similarity_transformers) |
| **Integrated Jacobians** | **Yes (native)** | ~50-1000 | ~50-1000 | 1-20x more | Approx. only | [EACL 2024](https://aclanthology.org/2024.eacl-long.125/) |
| **TCAV** | No (global) | 1+1 per concept | 1 per concept | ~10x cheaper | Yes | `captum.concept.TCAV` |
| **Probing Classifiers** | No (diagnostic) | 1 | 0 | ~50x cheaper | Yes | sklearn |
| **Sparse Autoencoders** | Yes (feature-level) | 0 (on cached) | 0 | ~50x cheaper | Yes | Custom (simple) |
| **BERTScore alignment** | **Yes (native)** | 0 (on cached) | 0 | ~50x cheaper | Yes | `bertscore` / trivial |
| **Sinkhorn OT** | **Yes (native)** | 0 (on cached) | 0 | ~50x cheaper | Yes | `pot` (installed) |
| **Hungarian alignment** | **Yes (native)** | 0 (on cached) | 0 | ~50x cheaper | Yes | `scipy` |
| **ColBERT MaxSim** | **Yes (native)** | 0 (on cached) | 0 | ~50x cheaper | Yes | Trivial |

---

## 1. Attention-Based Attribution

### 1.1 Attention Rollout

**Reference:** Abnar & Zuidema, "Quantifying Attention Flow in Transformers," ACL 2020.

**Core mechanism:** Recursively multiplies attention weight matrices across layers, combined with identity matrices (for residual connections), to approximate how much information from each input token reaches a given hidden representation. Result: a single matrix mapping hidden states back to input token contributions.

**Frozen models:** Fully compatible. Only requires `output_attentions=True`. No gradients needed.

**Pairwise:** Single-sequence only. Can be combined with cosine similarity: `rollout_q * cosine_sim * rollout_c`.

**Cost:** 1 forward pass + cheap matrix multiplications. ~50x cheaper than IG.

**Interpretability:** Higher correlation with gradient-based importance than raw attention (Abnar & Zuidema 2020). However, produces overly diffuse attributions because it treats all attention heads equally and ignores MLPs/LayerNorm (Chefer et al. 2021).

### 1.2 Attention Flow / Generalized Attention Flow (GAF)

**References:**
- Abnar & Zuidema, "Quantifying Attention Flow in Transformers," ACL 2020.
- Azarkhalili & Libbrecht, "Generalized Attention Flow," ACL 2025.

**Core mechanism:** Treats the multi-layer attention graph as a flow network. Uses maximum-flow algorithms to compute information flow from hidden nodes to input tokens, capturing bottleneck effects that rollout misses. GAF (2025) adds gradient-based head weighting.

**Frozen models:** Yes. GAF adds 1 backward pass.

**Pairwise:** Single-sequence only.

**Cost:** 1 forward pass + max-flow computation (+ 1 backward for GAF). ~20-50x cheaper than IG.

**Interpretability:** GAF shows consistent improvements over rollout on AOPC and LOdds benchmarks. No human evaluation studies.

### 1.3 Chefer et al. -- Gradient-Weighted Relevance (CVPR/ICCV 2021)

**References:**
- Chefer, Gur & Wolf, "Transformer Interpretability Beyond Attention Visualization," CVPR 2021.
- Chefer, Gur & Wolf, "Generic Attention-Model Explainability for Interpreting Bi-Modal and Encoder-Decoder Transformers," ICCV 2021 (Oral).

**Core mechanism:** Uses gradients to weight attention heads (rather than treating all heads equally), then applies rollout-style aggregation. The ICCV 2021 extension handles encoder-decoder and cross-attention architectures, propagating relevance through cross-attention connections.

**Frozen models:** Yes. 1 forward + 1 backward pass.

**Pairwise:** The ICCV 2021 method handles cross-attention, producing source-to-target token mappings. Directly applicable to T5-based LaTa/PhilTa if query is encoded and candidate is fed as decoder input.

**Cost:** 1 forward + 1 backward. ~25x cheaper than IG.

**Interpretability:** Significant improvement over vanilla rollout. Official implementations available.

**Implementation:** [Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)

### 1.4 ALTI+ (Encoder-Decoder Attribution)

**Reference:** Ferrando et al., "Measuring the Mixing of Contextual Information in the Transformer," EMNLP 2022.

**Core mechanism:** Decomposes the entire attention block (multi-head attention + residual + LayerNorm) into a sum of vectors, measures each token's contribution via vector norms/cosine similarity. ALTI+ extends to encoder-decoder models with cross-attention, producing source-to-target token attribution.

**Frozen models:** Yes, fully post-hoc. No gradient computation needed.

**Pairwise:** Yes for encoder-decoder models (LaTa, PhilTa). Produces source-token to target-token attribution matrix through cross-attention decomposition.

**Cost:** 1 forward pass + linear algebra decomposition. ~50x cheaper than IG.

**Interpretability:** More faithful than gradient-based methods and raw attention on machine translation tasks.

**Implementation:** Meta's `stopes` library includes ALTI+ for fairseq models.

### 1.5 Self-Attention Attribution (ATTATTR)

**Reference:** Hao, Dong, Wei & Xu, "Self-Attention Attribution: Interpreting Information Interactions Inside Transformer," AAAI 2021 (Best Paper Runner-Up).

**Core mechanism:** Applies integrated gradients specifically to self-attention weights (not input embeddings), producing token-to-token interaction scores at each layer. Aggregates into "attribution trees" showing hierarchical information flow.

**Frozen models:** Yes, works on pre-trained models without modification.

**Pairwise:** Within-sequence token-to-token interaction only. Could work on concatenated [query; candidate] input.

**Cost:** ~20-50 forward-backward passes (similar to standard IG). No speed advantage.

**Implementation:** [attattr](https://github.com/YRdddream/attattr)

### 1.6 GlobEnc (Encoder-Only Attribution)

**Reference:** Modarressi et al., "GlobEnc: Quantifying Global Token Attribution by Incorporating the Whole Encoder Layer in Transformers," NAACL 2022.

**Core mechanism:** Incorporates all encoder components (attention, FFN, LayerNorm, residual) into attribution, aggregated across layers. Outperforms previous methods on correlation with gradient-based saliency.

**Frozen models:** Yes.

**Pairwise:** Single-sequence only.

**Cost:** 1 forward pass + decomposition. ~50x cheaper than IG.

**Implementation:** [GlobEnc](https://github.com/mohsenfayyaz/GlobEnc)

---

## 2. SHAP Interaction Values

### 2.1 SHAP Interaction Index

**Reference:** Lundberg & Lee, "A Unified Approach to Interpreting Model Predictions," NeurIPS 2017.

**Core mechanism:** Extends Shapley values to pairwise interactions: `Delta_ij(S) = v(S union {i,j}) - v(S union {i}) - v(S union {j}) + v(S)`, averaged over all coalitions with Shapley weighting. Produces an M x M interaction matrix per prediction.

**Frozen models:** Yes (model-agnostic, black-box).

**Pairwise:** Yes -- the only SHAP variant that natively produces a feature-by-feature interaction matrix. However, exact computation requires 2^n evaluations; sampling-based approximations reduce this.

**Cost:** For 50 tokens total, exact computation is O(2^50). With sampling (SHAP-IQ), budget ~2,000 evaluations minimum. **100-500x more expensive than IG.**

**Interpretability:** Strong theoretical guarantees (linearity, symmetry, dummy, efficiency axioms). Limited human evaluation in NLP settings.

### 2.2 MultiSHAP (Cross-Modal Interactions)

**Reference:** Wang et al., "MultiSHAP," arXiv 2508.00576, 2025.

**Core mechanism:** Computes cross-modal interaction matrix using Shapley Interaction Index. For each element pair (q_i, c_j), measures the synergistic/suppressive contribution beyond individual marginal effects. Originally designed for image-text VQA but directly adaptable to query-candidate token interactions.

**Frozen models:** Yes, fully model-agnostic.

**Pairwise:** YES -- directly produces an m x n matrix analogous to our desired token_q x token_c matrix.

**Cost:** With K=128 Monte Carlo samples: ~80,000 forward passes per pair for 25+25 tokens. **400-2000x more expensive than IG.**

**Interpretability:** Identifies four interaction patterns: beneficial synergy, harmful synergy, helpful suppression, detrimental suppression.

### 2.3 SHAP-IQ / shapiq Library

**Reference:** Muschalik et al., "SHAP-IQ: Unified Approximation of any-order Shapley Interactions," NeurIPS 2024.

**Core mechanism:** Python library for approximating any-order Shapley interactions with theoretical guarantees and variance estimates. Supports k-SII, Faith-Shap (FSII), FBII. Budget is user-specified.

**Frozen models:** Yes.

**Pairwise:** Yes, any-order interactions supported.

**Cost:** User-specified budget (~256-2,000 evaluations demonstrated). Quality depends on budget.

**Implementation:** [shapiq](https://github.com/mmschlk/shapiq) (requires Python 3.12+)

### 2.4 Key Finding: SHAP for Bi-Encoder Retrieval

**Critical limitation:** No published work applies SHAP interactions to bi-encoder retrieval. Your models encode query and candidate independently, so masking a query token only affects the query embedding. The interaction `Delta_ij(S)` reduces to the change in similarity when both tokens are present vs. each alone -- well-defined but may produce sparse/low-magnitude values.

**Masking strategy matters enormously:** For T5, use sentinel tokens. For BERT, use [MASK]. For decoder-only models, masking mid-sequence is unnatural. No consensus on the right baseline for frozen retrieval embeddings.

**Practical verdict:** Theoretically superior to IG-weighted cosine for principled interaction attribution, but 500-1000x more expensive. Feasible only for a handful of illustrative examples (5-10 pairs for a paper figure), not for systematic corpus-level evaluation.

---

## 3. Layer-wise Relevance Propagation (LRP)

### 3.1 Conservative LRP for Transformers (CP-LRP)

**Reference:** Ali, Schnake, Eberle, Montavon, Muller & Wolf, "XAI for Transformers: Better Explanations through Conservative Propagation," ICML 2022.

**Core mechanism:** Identifies attention heads and LayerNorm as main sources of unreliable LRP in transformers. Proposes: attention patterns detached from computation graph and used as redistribution weights; LayerNorm denominators treated as constants. Restores the conservation property that standard LRP violates in transformers.

**Frozen models:** Yes, fully post-hoc. Rules only modify relevance redistribution during backpropagation.

**Pairwise:** No -- single-input attribution only.

**Cost:** 1 backward pass. ~40-50x cheaper than IG.

### 3.2 AttnLRP (State of the Art for Single-Input)

**Reference:** Achtibat et al., "AttnLRP: Attention-Aware Layer-wise Relevance Propagation for Transformers," ICML 2024.

**Core mechanism:** Derives novel LRP rules for non-linear attention within the Deep Taylor Decomposition framework:
- **Softmax rule:** Taylor linearization with bias absorption
- **Matrix multiplication (A*V) rule:** Sequential uniform + epsilon-LRP
- **LayerNorm/RMSNorm rule:** Identity rule (efficient)

**Frozen models:** Yes, via monkey-patching. LXT library patches model classes to redirect gradient computation through LRP rules.

**Pairwise:** No -- single-input attribution. Could serve as inner engine for BiLRP.

**Cost:** 2 forward + 1 backward pass. O(1) relative to forward pass. ~40-50x faster than IG.

**Interpretability:** Substantially outperforms IG on faithfulness. On Llama 2-7b: AttnLRP score 2.50+/-0.05 vs IG -1.23+/-0.05. 46% improvement over CP-LRP on Mixtral 8x7b.

**Models tested:** Flan-T5-XL (T5 architecture supported), Llama 2-7b, Phi-1.5, ViT. BERT support confirmed. Qwen 2 supported; Qwen 3 experimental.

**Implementation:** `pip install lxt` -- [LRP-eXplains-Transformers](https://github.com/rachtibat/LRP-eXplains-Transformers)

### 3.3 BiLRP -- MOST DIRECTLY RELEVANT FOR PAIR MATRIX

**References:**
- Eberle et al., "Building and Interpreting Deep Similarity Models," IEEE TPAMI 2022.
- **Vasileiou & Eberle, "Explaining Text Similarity in Transformer Models," NAACL 2024.**

**Core mechanism:** Decomposes bilinear similarity `y(x, x') = <phi(x), phi(x')>` into second-order relevance scores:

```
R_total = SUM_m [ LRP(phi_m, x) OUTER_PRODUCT LRP(phi_m, x') ]
```

For each embedding dimension m, compute LRP for both inputs, then take outer product. Sum across all dimensions. Result: a **token_q x token_c relevance matrix** showing which token pairs drive the similarity score.

**Frozen models:** Yes, explicitly designed as post-hoc. "The application of these rules does not affect the model's forward predictions."

**Pairwise:** **YES -- this is BiLRP's defining feature.** Natively produces a full token_q x token_c interaction relevance matrix. Validated on grammatical interactions, multilingual semantics, and biomedical text retrieval.

**Cost:** Requires **2 x h backward passes** (h = embedding dimension). For 768-dim BERT: ~1,536 backward passes. Reported ~2 minutes per pair on P100 GPU. **~30-40x MORE expensive than IG.** Scales linearly with embedding dimension.

**Interpretability:** Achieved 0.81 average cosine similarity with ground-truth interactions (vs. 0.62 for baselines). Correlation with human-annotated noun co-occurrence: rho = 0.94. Revealed that non-fine-tuned models use "simple token matching strategy" while fine-tuned models capture deeper semantic interactions.

**Models tested:** BERT, mBERT, SBERT, SGPT (GPT-Neo based). **T5 NOT tested** -- would need extension.

**Implementations:**
- Text similarity: [xai_similarity_transformers](https://github.com/alevas/xai_similarity_transformers) (NAACL 2024)
- Original vision: [BiLRP_explain_similarity](https://github.com/oeberle/BiLRP_explain_similarity)

**Key advantage over current approach:** BiLRP jointly decomposes the similarity score itself (conservation-based), while our current method multiplies two independently computed quantities (`cosine_sim * IG_weight`). BiLRP is theoretically grounded in the actual computation.

### 3.4 PE-Aware LRP (Latest Advance)

**Reference:** Bakish et al., "Revisiting LRP: Positional Attribution as the Missing Ingredient for Transformer Explainability," NeurIPS 2025.

**Core mechanism:** All prior LRP methods ignore positional encodings, violating conservation. Reformulates input as position-token pairs, introduces specialized rules for Rotary PE, Learnable PE, and Absolute PE.

**Frozen models:** Yes, extends LXT framework.

**Cost:** Similar to AttnLRP (single backward pass).

**Implementation:** [PE-AWARE-LRP](https://github.com/YardenBakish/PE-AWARE-LRP)

### 3.5 LRP Conservation Property

LRP's unique advantage over IG: **relevance conservation through layers**. Total relevance remains constant at each layer, meaning explanations always sum to the output score. IG's completeness axiom (attributions sum to f(x) - f(baseline)) is relative to a baseline and doesn't guarantee intermediate conservation.

| Component | LRP Rule | Conservation |
|-----------|----------|:------------:|
| Linear layers | LRP-0 / LRP-epsilon | Exact / Approx |
| Softmax | Taylor linearization | Approximate |
| LayerNorm/RMSNorm | Identity | Exact |
| A*V multiplication | Uniform + epsilon | Exact |
| Residual connections | Relevance splitting | Exact |
| Positional encodings | PE-Aware rules | Exact (new) |

---

## 4. Concept-Based Explanations

### 4.1 TCAV (Testing with Concept Activation Vectors)

**References:**
- Kim et al., "Interpretability Beyond Feature Attribution: Quantitative TCAV," ICML 2018.
- Nejadgholi et al., "Improving Generalizability in Implicitly Abusive Language Detection with CAVs," ACL 2022.

**Core mechanism:** Train a linear classifier to distinguish concept examples vs. random examples at a given layer. The decision boundary normal = Concept Activation Vector (CAV). TCAV score = dot product of CAV with gradient of prediction w.r.t. layer activations, measuring directional sensitivity to each concept.

**Frozen models:** Yes, fully post-hoc. Captum provides `captum.concept.TCAV`.

**Pairwise:** No -- measures global sensitivity of predictions to concepts. Would need a binary classification proxy (same-source vs. different-source) to apply to retrieval.

**Cost:** Training each CAV: seconds (linear classifier on cached activations). TCAV scores: 1 forward + 1 backward per example. ~10x cheaper than IG.

**Interpretability:** High by design -- results are inherently concept-level ("the model is 85% sensitive to legal_terminology at layer 6").

**Latin use case:** Define concept sets: `shared_legal_terms`, `shared_morphology`, `shared_named_entities`, `shared_syntax`. Gather ~50-200 exemplar activations per concept from canon corpus. Measure which layers are most sensitive to which concepts. This tells you "layer 6 SIF+ABTT embeddings are 72% sensitive to legal terminology."

**Limitation:** Global importance only -- tells you "legal terms matter for same-source retrieval" but not which specific legal terms connected passage A to passage B.

### 4.2 Post-hoc Concept Bottleneck Models (CBMs)

**References:**
- Koh et al., "Concept Bottleneck Models," ICML 2020.
- Yuksekgonul et al., "Post-hoc Concept Bottleneck Models," ICLR 2023 (Spotlight).
- Oikarinen et al., "Label-Free Concept Bottleneck Models," ICLR 2023.
- Sun et al., "Concept Bottleneck Large Language Models," ICLR 2025.

**Core mechanism:** Forces predictions through an intermediate "concept layer" where each dimension corresponds to a human-defined concept. Post-hoc CBMs train concept predictors + class predictor on frozen penultimate-layer embeddings. Label-free CBMs use sentence embeddings to auto-generate concept labels.

**Frozen models:** Excellent. Post-hoc CBMs are specifically designed for frozen backbones.

**Pairwise:** Not natively (classification design). But a "pairwise CBM" is natural: compare concept bottleneck vectors of two passages. If A = [legal=0.9, morphology=0.3] and B = [legal=0.8, morphology=0.6], similarity is explained as "driven by shared legal terminology."

**Cost:** Very low. Training: logistic regression on cached embeddings (minutes). Inference: 1 forward pass + lightweight linear predictions. Much cheaper than IG.

**Interpretability:** Strongest advantage -- explanations are inherently concept-level and directly auditable.

**Latin use case:** Define 10-20 Latin-specific concepts (shared_legal_vocabulary, shared_ecclesiastical_terms, shared_verb_morphology, shared_case_system, shared_named_entities). Label a small subset with concept scores (or use LLM auto-labeling). Train concept predictors on frozen SIF+ABTT embeddings. Explain retrieval as weighted combination of concept matches.

### 4.3 Probing Classifiers

**References:**
- Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances," Computational Linguistics 48(1), 2022.
- Bamman & Burns, "Latin BERT: A Contextual Language Model for Classical Philology," arXiv:2009.10053, 2020.

**Core mechanism:** Lightweight classifiers trained on frozen hidden states to predict specific linguistic properties (POS, morphological case, semantic domain). High accuracy = information is encoded in that representation.

**Frozen models:** This is the entire point of probing.

**Pairwise:** Not directly, but pairwise probes could detect shared linguistic properties between passage pairs.

**Cost:** Minimal. 1 forward pass for hidden states, then logistic regression training (seconds).

**Interpretability:** Moderate. Reveals "what information is present" but not "what the model uses." Control tasks (Hewitt & Liang 2019 selectivity) help but don't fully resolve this.

**Latin use case:** Train probes per layer for: morphological case, POS, semantic domain (legal/ecclesiastical/literary), text provenance. Cross-reference with retrieval accuracy per layer to understand the anisotropy dip.

### 4.4 Sparse Autoencoders (SAEs) for Embedding Disentanglement

**References:**
- O'Neill et al., "Disentangling Dense Embeddings with Sparse Autoencoders," arXiv:2408.00657, 2024.
- Anthropic, "Scaling Monosemanticity," Transformer Circuits, 2024.

**Core mechanism:** Train a sparse autoencoder to reconstruct dense embeddings as sparse linear combinations of learned features. Each feature ideally = one interpretable concept. Top-k sparsity ensures only a few features active per embedding.

**Frozen models:** Excellent. Trained entirely on cached embeddings from frozen models.

**Pairwise:** Very promising. Decompose passages into sparse features, then decompose cosine similarity into feature-by-feature contributions. "Passages A and B are similar because they share feature F7 (legal terminology, contribution=0.35) and F12 (ecclesiastical register, contribution=0.22)."

**Cost:** Training: one-time offline (~13K steps on cached embeddings). Inference: one matrix multiplication. Essentially free at explanation time.

**Interpretability:** Features capture diverse concepts: domains, methodologies, abstract patterns. Form hierarchical "feature families" (parent = broad concept, children = specializations).

**Latin use case:** Train SAE on 1,278 cached SIF+ABTT embeddings at optimal layer. Inspect learned features for Latin-meaningful concepts. Decompose retrieval similarity into feature contributions. Requires zero additional forward passes.

### 4.5 Concept Relevance Propagation (CRP)

**Reference:** Achtibat et al., "From Attribution Maps to Human-Understandable Explanations through Concept Relevance Propagation," Nature Machine Intelligence 5(9), 2023.

**Core mechanism:** Extends LRP by conditioning the backward relevance pass on specific concepts (neuron groups/clusters). Produces concept-specific heatmaps answering both "where" and "what" simultaneously.

**Frozen models:** Yes, post-hoc. Mainly demonstrated on CNNs; transformer adaptation requires AttnLRP rules.

**Implementation:** [zennit-crp](https://github.com/rachtibat/zennit-crp)

### 4.6 LACOAT (Latent Concept Attribution)

**Reference:** Yu et al., "Latent Concept-based Explanation of NLP Models," EMNLP 2024.

**Core mechanism:** Four-module pipeline: (1) hierarchical clustering on hidden states discovers latent concepts; (2) IG finds salient input words; (3) logistic regression maps test representations to concept clusters; (4) LLM generates human-readable concept summaries.

**Pairwise:** No -- classification only. But concept discovery could identify meaningful Latin categories automatically.

**Cost:** Higher than IG (uses IG internally with 500 steps + concept discovery overhead + LLM API calls).

**Interpretability:** 89% of annotators found latent concepts relevant. For Latin, could automatically separate "lex" into legal-term vs. general-reading facets.

---

## 5. Cross-Attention Probing & Alignment Methods

### 5.1 BERTScore-Style Greedy Alignment

**Reference:** Zhang et al., "BERTScore: Evaluating Text Generation with BERT," ICLR 2020.

**Core mechanism:** Full cosine similarity matrix between all token embeddings from text A and B. Greedy max-matching for precision/recall.

**Frozen models:** Works entirely on pre-computed hidden states. No model access needed.

**Pairwise:** YES -- the similarity matrix `sim[i,j] = cos(h_q[i], h_c[j])` IS the alignment matrix.

**Cost:** One matrix multiply. Essentially free on cached embeddings.

**Interpretability:** Moderate. Works well at earlier layers. At later layers, heavy contextualization means token embeddings no longer represent just that token. The Vasileiou & Eberle EMNLP 2025 survey warns: "token embedding alignment does not equate to input token alignment for highly contextualized layers."

**Implementation:** Trivial: `sim = q_embs @ c_embs.T / (norms_q * norms_c)`.

### 5.2 ColBERT-Style Late Interaction (MaxSim)

**Reference:** Khattab & Zaharia, "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT," SIGIR 2020.

**Core mechanism:** For each query token, compute maximum cosine similarity against all document tokens. Final score = sum of per-query-token MaxSim values. Asymmetric "soft" alignment.

**Frozen models:** Yes. Pure post-hoc on token embeddings.

**Pairwise:** YES -- |q| x |c| similarity matrix with argmax per row showing alignment.

**Cost:** Same as BERTScore. Essentially free.

**Interpretability:** High for retrieval. Each query token's contribution is explicit and traceable.

### 5.3 Optimal Transport / Sinkhorn Alignment

**References:**
- Kusner et al., "From Word Embeddings to Document Distances," ICML 2015. (Word Mover's Distance)
- Zhao et al., "MoverScore: Text Generation Evaluating with Contextualized Embeddings and Earth Mover Distance," EMNLP 2019.
- Arase et al., "Unbalanced Optimal Transport for Unbalanced Word Alignment," ACL 2023.
- Huang et al., "OTTAWA: Optimal TransporT Adaptive Word Aligner," Findings of ACL 2024.

**Core mechanism:** Models token alignment as optimal transport. Given a cost matrix (cosine distance), finds the transport plan minimizing total cost. Sinkhorn adds entropic regularization for speed. **Unbalanced OT** (Arase 2023) handles null alignment -- tokens with no counterpart -- critical for Latin manuscripts with different content.

**Frozen models:** YES. Takes pre-computed embeddings. No model access needed. `pot` is already installed.

**Pairwise:** YES -- the transport plan `T[i,j]` IS the alignment matrix. Unlike greedy BERTScore, OT produces many-to-many soft alignment.

**Cost:** Sinkhorn: O(L * n * m) where L ~ 50-100 iterations. For <512 tokens: milliseconds. Exact EMD: O(n^3) but sub-second. **Vastly cheaper than IG.**

**Interpretability:** HIGH. The transport plan is directly interpretable. Unbalanced OT naturally handles tokens that shouldn't align, assigning them to a "null" sink instead of forcing spurious alignments.

**Implementation:** `ot.sinkhorn(a, b, M, reg=0.05)` from POT. `a`, `b` = uniform distributions, `M` = cosine distance matrix. Return value = alignment matrix.

### 5.4 Word Rotator's Distance (WRD) / WSMD

**References:**
- Yokoi et al., "Word Rotator's Distance," EMNLP 2020.
- Yamaguchi et al., "Improving Word Mover's Distance by Leveraging Self-Attention Matrix," Findings of EMNLP 2023.

**Core mechanism (WRD):** Decomposes token embeddings into norm (importance) and direction (meaning). Uses norms as mass distribution and angular distance as cost in OT. Separates "what a token means" from "how important it is."

**Core mechanism (WSMD):** Extends WMD with structural information from self-attention via Fused Gromov-Wasserstein distance. +9.5% AUC improvement on PAWS-QQP.

**Frozen models:** Yes. WRD uses only embeddings; WSMD additionally uses attention weights.

**Pairwise:** YES -- both produce transport plans as alignment matrices.

### 5.5 Hungarian Algorithm (Hard 1-to-1 Alignment)

**Core mechanism:** Given cosine distance matrix, finds optimal one-to-one matching maximizing total similarity. Produces hard alignment where each token maps to exactly one counterpart.

**Frozen models:** Yes. Pure post-processing.

**Pairwise:** YES -- binary |q| x |c| matrix with min(|q|, |c|) matches.

**Cost:** O(n^3) via `scipy.optimize.linear_sum_assignment()`. Milliseconds for <512 tokens.

**Interpretability:** VERY HIGH for human inspection. Each token gets exactly one partner. Limitation: forced 1-to-1 may be unrealistic for Latin with inflectional variants mapping to the same concept.

### 5.6 Integrated Jacobians for Siamese Encoders

**References:**
- Moeller, Nikolaev & Pado, "An Attribution Method for Siamese Encoders," EMNLP 2023.
- Moeller, Nikolaev & Pado, "Approximate Attributions for Off-the-Shelf Siamese Transformers," EACL 2024.

**Core mechanism:** Generalizes IG to two-input Siamese models by computing Integrated Jacobians along interpolation paths. Produces feature-pair attributions that reduce to a token-token matrix.

**Frozen models:** The 2024 approximate version works on off-the-shelf models without modification.

**Pairwise:** YES -- |q| x |c| attribution matrix. Found that top 5% of token pairs account for ~77% of prediction. ~38% of attributions are negative (models balance matches against mismatches).

**Cost:** ~50-1000 forward+backward passes depending on layer depth. Comparable to or more expensive than current IG.

### 5.7 Key Survey Reference

**Reference:** Vasileiou & Eberle, "Interpretable Text Embeddings and Text Similarity Explanation: A Survey," EMNLP 2025.

**Taxonomy:**
- **Set-based methods** (BERTScore, ColBERT): Fast, no model access needed, but contextualization limits interpretability at later layers.
- **Attribution-based methods** (IG, BiLRP, Integrated Jacobians): Trace through computation graph, more principled but expensive.
- **Critical caveat:** "Token embedding alignment does not equate to input token alignment, as contextualization steps may obscure actual contributions of input tokens."

---

## Recommendations (Ranked by Feasibility)

### Tier 1: Immediate Implementation (use cached embeddings, zero extra cost)

**1. Sinkhorn OT Alignment** -- Replace IG pair matrix with optimal transport alignment on cached hidden states. You already have `pot` installed. Unbalanced OT handles null alignment naturally. Many-to-many soft matching. Millisecond computation per pair.

```python
import ot
M = 1 - cosine_similarity(q_hidden, c_hidden)  # cost matrix
T = ot.sinkhorn(a, b, M, reg=0.05)  # transport plan = alignment
```

**2. BERTScore/ColBERT-style raw cosine alignment** -- Pure cosine similarity matrix between token embeddings as a baseline. Overlay with MaxSim (ColBERT-style) per-token importance.

**3. Hungarian hard alignment** -- Clean 1-to-1 matching for paper figures showing unambiguous "this token matches that token."

### Tier 2: Low-Cost Model-Aware Methods (1-2 forward/backward passes)

**4. ALTI+ for T5 models (LaTa, PhilTa)** -- Cross-attention decomposition natively produces source-to-target attribution. 1 forward pass. Best attention-based method for encoder-decoder architectures.

**5. AttnLRP via LXT** -- State-of-the-art single-sequence attribution. `pip install lxt`. 2 forward + 1 backward pass. T5 architecture supported (Flan-T5-XL tested). Could replace IG for per-token importance at ~40x lower cost.

**6. Chefer ICCV 2021** -- Gradient-weighted relevance through cross-attention for T5 models. Official PyTorch implementation available.

### Tier 3: Concept-Level Understanding (requires concept definition + training)

**7. Sparse Autoencoders on cached SIF+ABTT embeddings** -- Train SAE on 1,278 cached embeddings. Inspect learned features for Latin-meaningful concepts. Decompose similarity into feature-by-feature contributions. Zero extra forward passes at explanation time. Most novel approach.

**8. TCAV with Latin concept sets** -- Define concept sets (legal, morphological, ecclesiastical, etc.), train CAVs per layer. Global diagnostic: "which concepts matter at which layers." Uses Captum's built-in TCAV.

**9. Post-hoc Concept Bottleneck** -- Train concept predictors on frozen embeddings. Fully interpretable retrieval: "these passages match because concept_legal=0.9."

### Tier 4: High-Quality but Expensive Pairwise Attribution

**10. BiLRP (NAACL 2024)** -- The only method natively designed to explain cosine similarity at the token-pair level. Produces the theoretically principled version of our pair matrix. rho=0.94 with human judgment. But ~30-40x MORE expensive than IG (~1,536 backward passes for 768-dim models). T5 not supported in existing implementation. Best for a small number of showcase examples.

**11. SHAP Interaction Values (via shapiq)** -- Principled game-theoretic interaction matrix. 500-1000x more expensive than IG. Feasible only for 5-10 illustrative pairs.

### Recommended Strategy

**Phase 1 (immediate):** Implement Sinkhorn OT + raw cosine alignment on cached embeddings. Compare qualitatively with current IG pair matrix. If alignment matrices look more linguistically meaningful to a Latin scholar, this is the solution -- no IG needed for pair explanations.

**Phase 2 (low effort):** Add ALTI+ or Chefer ICCV 2021 for T5 models to get attention-based cross-sequence attribution. Compare with OT alignment.

**Phase 3 (concept-level):** Train an SAE on cached embeddings to discover interpretable Latin concept directions. This could be the most novel contribution.

**Phase 4 (validation):** For 5-10 showcase pairs, run BiLRP as ground truth for token-pair attribution quality. Compare OT alignment vs. BiLRP to validate whether the cheap alignment methods capture similar patterns.

---

## Key References

### Attention-Based
- Abnar & Zuidema, ACL 2020 -- Attention Rollout & Flow
- Azarkhalili & Libbrecht, ACL 2025 -- Generalized Attention Flow
- Chefer et al., CVPR 2021 -- Gradient-weighted relevance
- Chefer et al., ICCV 2021 -- Bi-modal/encoder-decoder explainability
- Hao et al., AAAI 2021 -- Self-Attention Attribution (ATTATTR)
- Ferrando et al., EMNLP 2022 -- ALTI/ALTI+
- Modarressi et al., NAACL 2022 -- GlobEnc
- Jain & Wallace, NAACL 2019 -- "Attention is not Explanation"
- Wiegreffe & Pinter, EMNLP 2019 -- "Attention is not not Explanation"

### SHAP
- Lundberg & Lee, NeurIPS 2017 -- SHAP
- Kokalj et al., EACL 2021 -- TransSHAP
- Wang et al., arXiv 2025 -- MultiSHAP
- Wenderoth et al., AAAI 2025 -- InterSHAP
- Muschalik et al., NeurIPS 2024 -- SHAP-IQ/shapiq
- Tsai et al., JMLR 2023 -- Faith-Shap
- Mosca et al., COLING 2022 -- SHAP review for NLP

### LRP
- Ali et al., ICML 2022 -- Conservative LRP for transformers
- Achtibat et al., ICML 2024 -- AttnLRP
- Eberle et al., IEEE TPAMI 2022 -- BiLRP (original)
- **Vasileiou & Eberle, NAACL 2024 -- BiLRP for text similarity**
- Bakish et al., NeurIPS 2025 -- PE-Aware LRP
- Achtibat et al., Nature MI 2023 -- Concept Relevance Propagation

### Concept-Based
- Kim et al., ICML 2018 -- TCAV
- Nejadgholi et al., ACL 2022 -- TCAV for NLP
- Koh et al., ICML 2020 -- Concept Bottleneck Models
- Yuksekgonul et al., ICLR 2023 -- Post-hoc CBMs
- Oikarinen et al., ICLR 2023 -- Label-free CBMs
- Sun et al., ICLR 2025 -- CB-LLMs
- Yu et al., EMNLP 2024 -- LACOAT
- O'Neill et al., arXiv 2024 -- SAE for embeddings
- Belinkov, Computational Linguistics 2022 -- Probing survey
- Bamman & Burns, arXiv 2020 -- Latin BERT

### Alignment / Cross-Attention
- Zhang et al., ICLR 2020 -- BERTScore
- Khattab & Zaharia, SIGIR 2020 -- ColBERT
- Kusner et al., ICML 2015 -- Word Mover's Distance
- Zhao et al., EMNLP 2019 -- MoverScore
- Arase et al., ACL 2023 -- Unbalanced OT for word alignment
- Huang et al., ACL Findings 2024 -- OTTAWA
- Yokoi et al., EMNLP 2020 -- Word Rotator's Distance
- Yamaguchi et al., EMNLP Findings 2023 -- WSMD
- Moeller et al., EMNLP 2023 -- Integrated Jacobians
- Moeller et al., EACL 2024 -- Approximate Integrated Jacobians
- **Vasileiou & Eberle, EMNLP 2025 -- Text similarity explanation survey**

### Implementations
- [LXT (AttnLRP)](https://github.com/rachtibat/LRP-eXplains-Transformers) -- `pip install lxt`
- [BiLRP for text](https://github.com/alevas/xai_similarity_transformers)
- [Chefer Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability)
- [GlobEnc](https://github.com/mohsenfayyaz/GlobEnc)
- [ATTATTR](https://github.com/YRdddream/attattr)
- [shapiq](https://github.com/mmschlk/shapiq)
- [PE-Aware LRP](https://github.com/YardenBakish/PE-AWARE-LRP)
- [zennit-crp](https://github.com/rachtibat/zennit-crp)
- [POT](https://pythonot.github.io/) -- already installed
