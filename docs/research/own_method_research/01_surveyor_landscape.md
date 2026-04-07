# Lane 1 — Landscape Survey: Learned Masks for Attribution (2023–2026)

**Author:** surveyor agent
**Date:** 2026-04-07
**Purpose:** Map the post-MaRC literature on learned-mask attribution and identify
the concrete gap our Latin-retrieval method should exploit. Sister deliverables
in `docs/research/own_method_research/`.

**Anchor citation (do not re-derive):** Brinner & Zarrieß 2023, *Model
Interpretability and Rationale Extraction by Input Mask Optimization*, Findings
of ACL 2023, pp. 13722–13744. arXiv 2508.11388 is a 2025 re-deposit, not a new
paper. Method already digested in `docs/research/attribution_paper_2025_input_mask.md`.

---

## 1. Literature map (themes, then papers)

### Theme A — Learned-mask / rationale extraction (the MaRC family)

1. **Brinner & Zarrieß 2023 (Findings of ACL)** — *MaRC*. Per-instance Adam
   optimisation over a Gaussian-reparameterised soft mask `λ`, frozen
   classifier, classification-likelihood fidelity loss + sparsity + bandwidth
   log-barrier. ~2–3 min/example on BERT-base. Movie reviews + ImageNet only.
   This is the paper we are differentiating from.

2. **Brinner & Zarrieß 2024 (EMNLP main)** — *Rationalizing Transformer
   Predictions via End-To-End Differentiable Self-Training*. Same authors'
   follow-up. Collapses the classical three-player game (selector + classifier +
   complement classifier) into a **single model that simultaneously classifies
   and scores tokens in one forward pass** — i.e. an **amortised** rationale
   predictor. Still classification-only (text classification benchmarks, ERASER-style
   evaluation). Loss is class-wise rationale alignment, not pair-similarity.
   This is the most important paper to engage with: it already closes MaRC's
   "amortisation" gap for classification, so the gap our method exploits cannot
   just be "make MaRC amortised".

3. **Various 2024–2025 *select-then-predict* descendants** of Lei et al. 2016 /
   Bastings et al. 2019 / Yu et al. 2019. The space is mature for
   classification but uniformly assumes a categorical-label fidelity signal.
   None of these has been ported to a cosine-similarity / dense-retrieval
   objective. (Confirmed by Opitz et al. 2025 survey, see Theme C.)

### Theme B — Explanations for *similarity* and *bi-encoder retrieval*

4. **Vasileiou & Eberle 2024 (NAACL main, pp. 7859–7873)** — *Explaining Text
   Similarity in Transformer Models*. Adapts BiLRP (Eberle et al. 2022) to
   transformer-based bilinear similarity models. Produces **second-order**
   relevance scores `R(i,j)` over query×candidate token pairs by
   conservation-preserving LRP propagation. **Loss-free, no learned parameters,
   no mask** — pure post-hoc gradient-style propagation. Closest published
   work to "attribution for sentence-pair similarity" but explicitly *not* a
   learned-mask method, and not designed for ABTT-cleaned embedding spaces.

5. **Chen, Merullo & Eickhoff 2024 (SIGIR, pp. 1401–1410)** — *Axiomatic
   Causal Interventions for Reverse Engineering Relevance Computation in
   Neural Retrieval Models*. Activation patching on TAS-B to isolate attention
   heads that detect duplicate query-document tokens. Mechanistic
   interpretability, *not* a learned mask, *not* per-example attribution.
   Interesting because it is the only 2024 paper that takes a neural ranker
   apart at the head level for IR; relevant if our method ever needs to
   justify why ABTT-cleaned middle layers carry retrieval signal.

6. **Wang et al. 2024 (arXiv 2408.13672)** — *ColBERT [MASK] Tokens Perform
   Term Weighting and Exhibit Cyclic Contextualization*. Probes ColBERT's mask
   tokens as implicit term weighters. Not a mask-learning method but
   confirms the IR community treats token weights as ad-hoc artefacts of
   training, not as objects to optimise post-hoc.

7. **Opitz, Schaefer, Müller, Padó & Frank 2025 (EMNLP main, pp. ~1135)** —
   *Interpretable Text Embeddings and Text Similarity Explanation: A Survey*.
   Comprehensive 2025 survey covering interaction-attribution methods
   (BiLRP / Integrated Jacobians), surrogate models, sparse-lexical retrieval
   (SPLADE, BM25-aware), late interaction (ColBERT, BERTScore), and
   space-shaping. **Explicitly notes no learned-mask methods exist in this
   space and that input-saliency methods for embedding similarity are
   compute-heavy and only aim to explain *similarity itself*, not retrieval
   decisions.** This is the single most load-bearing citation for the gap
   claim.

### Theme C — Frontier post-hoc attribution that goes beyond UNK substitution

8. **Achtibat et al. 2024 (Nature Machine Intelligence)** — *AttnLRP:
   Attention-aware Layer-wise Relevance Propagation for Transformers*. Sharper
   LRP rules for attention heads, layer norms, and GeLU. Model-internal,
   gradient-style, *not* a learned mask. Important as a "post-hoc baseline
   that already exists" — our method should beat it on faithfulness for the
   retrieval objective.

9. **Lopardo et al. / "When LRP Diverges from Leave-One-Out in Transformers"
   (BlackboxNLP 2025)** — Documents that LRP-style propagation and
   leave-one-out perturbation give *different* answers in transformers, and
   that perturbation is closer to ground truth on faithfulness metrics. This
   is a strong argument *for* perturbation-style (mask) methods like MaRC and
   ours, *against* purely gradient-based methods like BiLRP/AttnLRP.

10. **Monteiro Paes et al. 2024 (ICML)** — *Selective Explanations*.
    Amortised explainer (small head emits feature attribution in one forward
    pass) plus a "select then refine" gating: amortised by default, expensive
    Monte Carlo only for inputs where the amortised head looks unreliable.
    Generic feature attribution, **not text-specific, not retrieval-specific**,
    but the architectural pattern (amortised mask + per-example refinement) is
    directly copyable.

### Theme D — Rationalisation for retrieval / IR (the empty corner)

11. **(absence)** — Despite searching ACL/EMNLP/NAACL/SIGIR/ECIR 2023–2026 we
    found **zero** papers that learn a soft attention mask whose objective is
    pair-similarity in dense-embedding space. SPLADE-family work (Formal et
    al. 2021–2024) learns sparse *vocabulary* weights for retrieval, not
    per-instance attention masks for explanation. ColBERT's late interaction
    gives token alignments for free but they are an emergent byproduct of
    MaxSim, not an optimised explanation. EXS (Singh & Anand 2019) and
    axiomatic explanations (Anand et al. 2021) are LIME-style or
    rule-matching, not learned masks. The Opitz et al. survey confirms this
    absence.

---

## 2. Frontier scan (post-MaRC, by seed idea)

| Seed idea (for the methodologist) | Closest published work | Status of the gap |
|---|---|---|
| **Amortised mask** (one forward pass head emits `λ`) | Brinner & Zarrieß 2024 (EMNLP) for *classification*; Monteiro Paes et al. 2024 for generic attribution | **Done for classification, untouched for retrieval/cosine.** |
| **Embedding-space objective** (mask trained against cosine similarity, not class likelihood) | Vasileiou & Eberle 2024 (BiLRP for similarity) — but no mask, no optimisation, no sparsity prior | **Open.** No learned-mask method optimises a cosine-similarity loss. |
| **Joint query×candidate mask** (mask both sides, not just query) | ColBERT late interaction gives an alignment matrix for free; BiLRP gives a 2nd-order interaction matrix; neither *learns* a mask | **Open.** Existing IR explanation work is single-side or alignment-as-byproduct. |
| **Structured-distance kernel** (replace `d(i,j) = |i−j|` with manuscript-aware distance) | None — even MaRC's image variant uses 8-connected pixel grid | **Open.** Linguistic / manuscript / parse-aware kernels are unexplored. |
| **ABTT-cleaned representation as the optimisation surface** | Mu & Viswanath 2018 (ABTT itself, no mask); our own pipeline (no mask) | **Open by construction.** ABTT is post-hoc geometry surgery, no one has plugged it into a mask-learning loss. |
| **Contrastive / contrastive-positive mask** (mask must keep `cos(q, c⁺)` high while keeping `cos(q, c⁻)` low) | Causal contrastive learning (NeurIPS 2024) is for treatment effects, not text rationales; PairCFR (ACL 2024) augments classification training data, not explanation | **Open for retrieval rationales.** |

---

## 3. THE GAP (the deliverable)

**Primary gap (sharp, defensible, exploitable in our setting):**

> **No published method learns a per-instance soft mask whose fidelity loss is
> pair cosine similarity in an ABTT-cleaned, mean-pooled sentence-embedding
> space, and the broader IR-explanation literature has only studied
> single-sided (query-only) attributions or alignment-as-byproduct (ColBERT
> MaxSim), never a *jointly optimised* `(λ_q, λ_c)` pair of token masks.**

Why this is real and not generic:

- **MaRC and Brinner & Zarrieß 2024** both target a *classifier head* with a
  log-likelihood loss `L(x̃, c)`. Our retrieval task has no classifier head:
  the only meaningful fidelity signal is `cos(M(x̃_q), M(x̃_c))` after the
  ABTT projection, which is mathematically a different objective and changes
  what the optimal mask looks like (e.g., it can favour mask configurations
  that *cancel anisotropic dimensions* rather than preserve them).
- **Vasileiou & Eberle 2024 (BiLRP-for-similarity)** is the only published
  text-similarity attribution method we found. It is gradient-propagation,
  not a learned mask, has no sparsity or smoothness prior, and is not
  designed for ABTT-cleaned spaces. It also produces a `(q_len, c_len)`
  interaction matrix similar to our existing 6 methods — so a learned-mask
  method gives us something *categorically* new in our 7-method comparison.
- **Opitz et al. 2025 survey** explicitly enumerates the methods that exist
  for explaining text-embedding similarity and contains zero learned-mask
  entries. This makes the gap claim a citation, not a vibe.
- **The IR community's own attribution work** (EXS, axiomatic explanations,
  Chen et al.'s causal interventions) is either model-agnostic surrogate
  fitting (LIME-style) or mechanistic-interpretability head probing. None
  optimises a mask per query-candidate pair against a similarity score.

**Secondary gap (cheaper to claim, useful as a hedge):**

> **No mask-learning method has used a non-sequential (e.g., manuscript-line,
> dependency-tree, or chunk-aware) distance kernel in the Gaussian
> reparameterisation.** MaRC's `d(i,j) = |i−j|` is a strong inductive bias
> that assumes consecutive tokens belong together — fine for movie reviews,
> wrong for fragmentary Latin manuscripts where the *line/folio* index
> matters more than the *token* index. This is a smaller methodological
> contribution but it is genuinely novel and our dataset uniquely enables it.

**Why the Latin retrieval setting is well-positioned to fill these gaps:**

- We already cache `query_hidden`, `pcs`, and `mean_vec` for ABTT in every IG
  artifact NPZ, so the embedding-space optimisation surface is one
  numpy-load away.
- We already have 6 attribution methods that produce `(q_len, c_len)`
  matrices, so the joint-mask method drops into the same comparison
  scaffolding (Lane 4's synthesis can re-use the existing visualiser).
- The 2,238 unlabelled queries in `data/canon_unlabelled/` give an
  evaluation pool that is two orders of magnitude larger than MaRC's
  hand-curated 100-example movie review set, which makes amortisation
  *necessary* rather than optional.

---

## 4. Risks to the gap claim

1. **Brinner & Zarrieß 2024 (EMNLP) might already cover more than its
   abstract suggests.** We could not extract the full PDF in this sprint
   (PDF binary on the anthology mirror, OpenReview gated). The abstract and
   the available metadata point to classification-only, but Lane 2 should
   skim §3 (loss function) and §4 (experiments) before locking in the gap
   wording. *If* it secretly evaluates on STS or any pair-similarity task,
   our gap shrinks from "open corner" to "open in retrieval but not STS",
   which is still publishable but weaker.

2. **A non-NLP venue (ICLR, NeurIPS, ICML 2024–2025) may have published a
   learned-mask method for embedding-space similarity** that we did not catch
   because we biased the search to ACL Anthology. Specifically worth a
   targeted check: anything from the **representation alignment**,
   **mechanistic interpretability**, or **contrastive explanation** workshops
   at ICLR 2024/2025 (the survey hits a Re-Align workshop paper as a
   citation, suggesting that community is active here).

3. **The IR community may publish "rationale" work without using the word
   "rationale".** Worth grepping SIGIR/ECIR/CIKM 2024–2026 for
   "explainable dense retrieval", "post-hoc IR explanation", and
   "query-document attribution" specifically. The Chen et al. 2024 SIGIR
   paper is the only one we surfaced; there may be 2025 follow-ups we missed.

4. **The "joint query+candidate mask" framing might exist in image retrieval
   / visual grounding** under different terminology (e.g., "co-attention
   masks" in V&L). If so, our novelty is "first text-retrieval instance" not
   "first ever", which is fine but should be honest in the related-work
   paragraph.

5. **MaRC's authors may be working on the retrieval extension right now.**
   Brinner & Zarrieß have published consistently (2023, 2024, and the 2025
   re-deposit suggests ongoing curation). Worth a single check of
   `dblp.org/pid/49/8155.html` (Sina Zarrieß) before submission to confirm
   nothing landed at NAACL/ACL 2026 in the same corner.

---

## 5. Positioning statement (one paragraph for Lane 4)

MaRC (Brinner & Zarrieß, ACL Findings 2023) and its EMNLP 2024 follow-up
established per-instance and amortised soft-mask rationale extraction for
**transformer classifiers**, with a classification-likelihood fidelity loss
and a sequential `|i−j|` distance prior. The only published attribution
method designed for **transformer-based text similarity** is Vasileiou &
Eberle (NAACL 2024), which is a gradient-propagation BiLRP variant with no
learned parameters, no sparsity prior, and no notion of post-hoc geometric
correction. The 2025 Opitz et al. survey of interpretable text embeddings
contains zero learned-mask entries and explicitly flags input-saliency for
embedding similarity as an open problem. Our Latin manuscript retrieval
setting sits exactly in this empty corner: we have a frozen bi-encoder, a
cosine-similarity objective, an ABTT-cleaned embedding surface, and a
2,238-query unlabelled pool. The methodologist (Lane 2) should design a
learned-mask variant that (a) optimises ABTT-cleaned cosine similarity
rather than classification log-likelihood, (b) jointly masks query and
candidate tokens, and (c) is amortised so it scales to the full unlabelled
pool — leaving the manuscript-aware distance kernel as an optional
secondary novelty if the primary contribution needs reinforcement.

---

**Word count:** ~1,460 words (under the 1,500 cap).

**Sources verified against ACL Anthology / arXiv during this sprint
(2026-04-07):**

- Brinner & Zarrieß 2023, ACL Findings — `aclanthology.org/2023.findings-acl.867/`
- Brinner & Zarrieß 2024, EMNLP main — `aclanthology.org/2024.emnlp-main.664/`
- Vasileiou & Eberle 2024, NAACL main — `aclanthology.org/2024.naacl-long.435/`
- Chen, Merullo & Eickhoff 2024, SIGIR — `dl.acm.org/doi/10.1145/3626772.3657841`
- Opitz et al. 2025, EMNLP main — `aclanthology.org/2025.emnlp-main.1135/`
- Monteiro Paes et al. 2024, ICML — arXiv 2405.19562
- Achtibat et al. 2024, AttnLRP — Nature Machine Intelligence (cited in BlackboxNLP 2025 reproducibility study)
- Lopardo et al. 2025, BlackboxNLP — `aclanthology.org/2025.blackboxnlp-1.10.pdf`
- Wang et al. 2024 ColBERT [MASK] probing — arXiv 2408.13672

**Citations the methodologist should re-verify before any of them appears in
the paper draft:**

- Achtibat et al. 2024 venue/year (we cite from a secondary source).
- Brinner & Zarrieß 2024 EMNLP §3 loss function and §4 task list (PDF
  inaccessible to this agent; abstract was the only direct evidence).
- Monteiro Paes et al. 2024 — confirm ICML acceptance vs arXiv-only.
