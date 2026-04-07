# Lane 4 — Positioning Against Prior Attribution Methods

**Author:** related-work-weaver agent
**Date:** 2026-04-07
**Purpose:** Provide the related-work section with (1) a 2-sentence pitch,
(2) per-rival differentiation lines, and (3) a 150-word positioning paragraph
ready to drop into the resubmission. Sister deliverables in
`docs/research/own_method_research/`.

**Anchor citations (verified by Lane 1, do not re-derive):**
- Brinner and Zarrieß 2023, *Model Interpretability and Rationale Extraction
  by Input Mask Optimization*, Findings of ACL 2023, pp. 13722–13744.
- Brinner and Zarrieß 2024, *Rationalizing Transformer Predictions via
  End-To-End Differentiable Self-Training*, EMNLP 2024 main.
- Vasileiou and Eberle 2024, *Explaining Text Similarity in Transformer
  Models*, NAACL 2024 main, pp. 7859–7873.

---

## 1. Two-sentence pitch

We introduce a learned-mask attribution method for bi-encoder retrieval that
optimises a per-instance soft mask over both the query and the candidate
against pair cosine similarity in an ABTT-cleaned embedding space. Unlike
gradient or alignment baselines, the mask is trained end-to-end with
sparsity and continuity priors, and a counterfactual variant flips the
calibrated Assignment-Accuracy decision rather than merely preserving the
score.

*(57 words, 2 sentences.)*

---

## 2. Per-rival differentiation lines

**Integrated Gradients (Sundararajan et al. 2017).**
We differ from IG by optimising a learnable token mask end-to-end against the
pair-similarity loss, instead of integrating input gradients along a
straight-line baseline path that has no notion of cosine fidelity or
ABTT projection.

**Captum-style perturbation (UNK substitution, occlusion, LIME — Ribeiro
et al. 2016, Kokhlikyan et al. 2020).**
We differ from occlusion- and LIME-style perturbation by jointly fitting a
continuous, smoothness-regularised mask under a single sparsity-controlled
objective, instead of enumerating discrete token deletions or fitting a
linear surrogate over independently sampled perturbations.

**MaRC (Brinner and Zarrieß 2023).**
We differ from MaRC by replacing its frozen-classifier log-likelihood
fidelity loss with ABTT-cleaned bi-encoder cosine similarity and by
optimising a *paired* `(λ_q, λ_c)` jointly, rather than a single mask over a
single classification input.

**Brinner and Zarrieß 2024 (EMNLP, amortised follow-up).**
We differ from the 2024 amortised variant by retaining the per-pair
optimisation surface required for retrieval, where the relevant supervision
signal is a pair score and not a class label, and by adding a
counterfactual objective tied to a calibrated Assignment-Accuracy
threshold rather than to class-rationale alignment.

**BERTScore-greedy (Zhang et al. 2020).**
We differ from BERTScore-greedy by *learning* a soft alignment under a
fidelity loss with sparsity and continuity priors, rather than reading off a
fixed greedy bipartite matching of token cosines that has no optimisation
loop, no notion of removed mass, and no relation to the model's actual
retrieval decision.

**DLA (Direct Logit Attribution).**
We differ from DLA by training a mask whose objective is preserved
(or flipped) similarity in the *post-ABTT* embedding space, rather than
reporting a closed-form, gradient-free, norm-weighted geometric-mean score
that is invariant to which tokens are perturbed.

**Attention-Weighted (`cosine × sqrt(diag attention)`).**
We differ from Attention-Weighted by treating attention as evidence to be
explained rather than as the explanation itself, and by optimising the mask
against pair similarity instead of multiplicatively re-scoring tokens by a
self-attention diagonal that is known to be a poor faithfulness signal
(Jain and Wallace 2019).

**Attention-Standalone (`cosine × sqrt(column-mean attention)`).**
We differ from Attention-Standalone by deriving token salience from a
fidelity-loss optimisation rather than from a column-mean of attention
weights, which mixes incoming attention from unrelated positions and
provides no per-pair supervision signal.

**Optimal Transport / EMD with `|IG|` mass (Cuturi 2013-style Sinkhorn).**
We differ from OT/EMD attribution by learning the marginal token weights
end-to-end against ABTT-cleaned cosine similarity, rather than fixing the
marginals from `|IG|` magnitudes and reading off the resulting transport
plan as the explanation.

**BiLRP for text similarity (Vasileiou and Eberle 2024).**
We differ from BiLRP by optimising a learned, sparsity- and
continuity-regularised mask under a similarity-fidelity loss in an
ABTT-cleaned space, rather than propagating second-order LRP relevance
through the network with no learned parameters and no post-hoc geometry
correction.

**ColBERT-style late-interaction alignment (Khattab and Zaharia 2020;
Santhanam et al. 2022).**
We differ from ColBERT-style alignment by *training* an explanation-time
mask against a frozen mean-pooled bi-encoder, rather than reading off a
MaxSim alignment that is an emergent byproduct of contrastive training and
exists only inside late-interaction architectures.

---

## 3. 150-word positioning paragraph (verbatim for related work)

Our method belongs to the family of **learned-mask, per-instance rationale
extractors** initiated by MaRC (Brinner and Zarrieß 2023) and amortised
for classification by their EMNLP follow-up (Brinner and Zarrieß 2024).
Both target a frozen classifier with a log-likelihood fidelity loss and a
sequential `|i−j|` distance prior; neither addresses sentence-pair
similarity. The closest published work on attribution for transformer-based
text similarity is BiLRP (Vasileiou and Eberle 2024), a gradient-propagation
method with no learned parameters, sparsity prior, or post-hoc geometric
correction. The corresponding corner — a learned, jointly optimised
`(λ_q, λ_c)` mask whose fidelity loss is pair cosine similarity in an
ABTT-cleaned bi-encoder space — has, to our knowledge, not been studied.
We address it with two complementary tiers: PairMask, a cached per-pair
optimiser over ABTT-cleaned cosine similarity, and CounterfactualMask, a
live Gumbel-sigmoid attention mask with bucket-conditional objectives that
flip the calibrated Assignment-Accuracy decision.

*(149 words.)*

---

## Notes for Lane 5 (synthesis) and the methods chair

1. **Cite venues exactly as written above.** MaRC is *Findings of ACL 2023*
   (not "August 2025 input-mask paper"). The 2024 follow-up is *EMNLP 2024
   main*. BiLRP-for-similarity is *NAACL 2024 main*.
2. **Hedge wording.** The paragraph deliberately says "to our knowledge"
   rather than "first" — Lane 1 flagged that ICLR/NeurIPS workshop
   contributions and non-NLP venues were not exhaustively searched. Keep
   the hedge in the camera-ready unless someone runs the targeted SIGIR /
   ECIR / CIKM 2024–2026 grep Lane 1 listed under risks.
3. **Variant names.** "PairMask" and "CounterfactualMask" are Lane 2's
   proposals; if they get renamed in Lane 5, find-and-replace here too.
4. **The Jain and Wallace 2019 citation** in the Attention-Weighted line
   ("Attention is Not Explanation", NAACL 2019) is well-known and safe.
   If the related-work section already cites it elsewhere, keep one
   instance and drop the other.
5. **Self-citations.** None used here. If our prior workshop paper or the
   pre-resubmission version of this paper introduced any of the 6 baseline
   methods (IG-cosine, attention-weighted, DLA, etc.) in our setting, the
   methods section should cite that introduction; the related-work section
   should not.

**Word count:** ~830 words (under the 1,000 cap).
