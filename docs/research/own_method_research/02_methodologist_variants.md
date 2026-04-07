# Lane 2 — Learned-Mask Attribution Variants (Methodologist Brainstorm)

**Goal.** Propose 3–5 learned-mask attribution variants that go beyond MaRC (Brinner & Zarrieß, ACL Findings 2023) and fit our retrieval pipeline. Two-tier scope: a *cached* variant (operates on existing NPZ artifacts, no model forward) and a *live* variant (full model in the loop). End with one recommendation per tier.

**Anchor.** MaRC = per-instance Adam over `(w, σ)` defining a Gaussian-kernel soft token mask, optimized against a *classification* log-likelihood (sufficiency + comprehensiveness) with sparsity + bandwidth log-barrier. Distance is linear `|i−j|`. Frozen classifier. ~2–3 min/instance on BERT-base.

We want to keep the *spirit* of "learn the mask" while attacking three weaknesses for our setting: (i) MaRC's loss is wrong for symmetric bi-encoder retrieval; (ii) MaRC ignores the ABTT geometry entirely; (iii) MaRC re-runs Adam per pair, which doesn't scale to our 4-bucket × 6-model × N-pair eval.

---

## Variant 1 — PairMask (cached, ABTT-cosine sufficiency/comprehensiveness)

**Parameterization.** Two soft token masks $\lambda^q,\lambda^c \in [0,1]^{n_q+n_c}$ via independent sigmoid logits, *no* Gaussian smoothing. Optionally a 1-D total-variation penalty for span continuity (cheaper than MaRC's bandwidth field; one extra hyperparameter). The masks reweight tokens *inside the existing pooling*: $\tilde h_q = \mathrm{ABTT}\big(\sum_i \lambda^q_i\, w^{\text{SIF}}_i\, h^q_i / \sum_i \lambda^q_i w^{\text{SIF}}_i\big)$ and similarly for $c$. ABTT here means subtracting `mean_vec` and projecting out `pcs`, both taken from the cached NPZ (no fitting at attribution time).

**Gradients.** Mask logits only. Everything else (hidden states, PCs, mean) is frozen and pre-cached. PyTorch autograd over the pooling+ABTT closed form — no model forward.

**Per-pair vs amortized.** Per-pair, ~50–100 LBFGS or Adam steps on a tiny tensor (≤512 logits per side). Wall-clock target: <1 s per pair on CPU.

**Loss.** Pair-similarity sufficiency/comprehensiveness in **ABTT-cosine** space:
$$\mathcal{L} = -\cos(\tilde h_q(\lambda),\, \tilde h_c(\lambda)) \;+\; \beta \cdot \cos\!\big(\tilde h_q(1-\lambda),\, \tilde h_c(1-\lambda)\big) \;+\; \alpha \big[\tfrac{1}{n}\sum \lambda_i\big]^2 \;+\; \gamma \cdot \mathrm{TV}(\lambda)$$
The first term *maximizes* similarity under the mask (sufficiency); the second *minimizes* similarity under the complement (comprehensiveness); $\alpha,\gamma$ are sparsity / continuity. This is the direct retrieval analog of MaRC's classification loss — and it is what MaRC literally cannot do because it has no class.

**Two-tier mapping.** **Cached only.** Touches `query_hidden`, `candidate_hidden`, `pcs`, `mean_vec` — exactly the NPZ keys that already exist.

**Differentiation from MaRC.** Same "learn a soft mask with sufficiency + comprehensiveness," but the loss is *symmetric pair similarity in ABTT-cleaned cosine space*, not classifier log-likelihood. MaRC has no notion of pair geometry; this is the minimal but principled port.

**Risk.** With masks on both sides and no kernel smoothing, the optimum can be degenerate (e.g., $\lambda$ collapses to a single best-aligned token pair). The TV term and an explicit minimum-mass constraint $\sum \lambda \ge \tau \cdot n$ are required.

---

## Variant 2 — SubspaceMask (cached, mask over ABTT principal-component coordinates)

**Parameterization.** Instead of a token mask, learn a *coordinate* mask $\mu \in [0,1]^K$ over the top-$K$ principal components of the per-token embedding cloud (with $K \gg D$, e.g. $K=64$). Decision function:
$$\tilde h = \mathrm{mean\_pool}(h) - \mathrm{mean\_vec}; \quad \hat h = \tilde h - \sum_{k=1}^K (1-\mu_k)\,(\tilde h^\top v_k)\, v_k$$
i.e., $\mu_k=0$ deletes component $k$ (like ABTT), $\mu_k=1$ keeps it. Recover token-level attribution post hoc by projecting each token's contribution through the kept subspace: $a_i = \big| \langle h^q_i,\, V \mathrm{diag}(\mu) V^\top \cdot h^c_{j^\star(i)} \rangle\big|$.

**Gradients.** Mask $\mu$ only. The component basis $V$ (top-$K$ PCs) is fit once on train hiddens — no per-pair PCA. We already store top-$D$ in `pcs`; one extra cached `pcs_extended` of size $K$ would be needed.

**Per-pair vs amortized.** Per-pair, but the optimization variable is *tiny* ($K\!\sim\!64$ scalars), so this is essentially closed-form (a few Adam steps).

**Loss.** Same ABTT-cosine sufficiency + complement term as Variant 1, plus $\ell_1$ on $(1-\mu)$ to encourage *deleting* few components. Optionally an entropy regularizer to keep $\mu$ near $\{0,1\}$.

**Two-tier mapping.** **Cached.** Needs a one-time extension of the artifact builder to store $K$ PCs instead of $D$.

**Differentiation from MaRC.** MaRC explains *which input tokens matter*. SubspaceMask explains *which dimensions of the learned embedding geometry matter for this pair* — a distinct interpretability question MaRC cannot ask. It is the natural learned-mask analog of ABTT itself.

**Risk.** Coordinate masks are less directly readable to a humanist scholar than token masks, and the back-projection step to recover token attributions is a second optimization whose faithfulness must be checked. Also: the "mask" lives in a model-specific basis, complicating cross-model comparison.

---

## Variant 3 — AmortizedHead (live + cached, one network for all pairs)

**Parameterization.** A small MLP / Transformer "mask head" $g_\phi$ that takes the cached per-token hiddens $(H^q, H^c)$ and emits two soft masks $\lambda^q,\lambda^c$ in one forward pass. Concretely: cross-attention block (one layer, $\sim$2–4 heads, 128-dim) followed by a per-token sigmoid. *No* Gaussian kernel — locality is learned.

**Gradients.** $\phi$ only; the bi-encoder is frozen. Trained corpus-wide on the full training split, *not* per pair.

**Per-pair vs amortized.** Amortized. After training, attribution for any pair is $O(\text{one head forward})$ — milliseconds. This is the single biggest scaling win over MaRC.

**Loss.** A *contrastive* + sufficiency objective evaluated in ABTT cosine space:
$$\mathcal{L}_\phi \;=\; \mathbb{E}_{(q,c^+,c^-)}\Big[ -\cos\big(\tilde h_q(\lambda),\, \tilde h_{c^+}(\lambda)\big) + \cos\big(\tilde h_q(\lambda),\, \tilde h_{c^-}(\lambda)\big) \Big] \;+\; \alpha\,\mathbb{E}\big[\bar\lambda\big]$$
with $c^-$ a hard in-batch negative. The mask must support *discrimination* across the corpus, not just inflate similarity for one pair. This is fundamentally different from MaRC's instance-local objective.

**Two-tier mapping.** **Both.** Cached tier: train $g_\phi$ on cached $(H^q,H^c)$ tensors (no model needed); inference is also cached. Live tier: same architecture, but during training the bi-encoder is *jointly fine-tunable* (optional ablation) to test whether the model "learns to be explainable."

**Differentiation from MaRC.** Amortized — one global mask predictor instead of per-instance Adam. Also: trained contrastively across pairs, so masks are consistent and comparable across the corpus, which MaRC's instance-local optimum is not.

**Risk.** Train/test distribution shift (the head is only as good as the training distribution of pairs); also, "what is the head learning about Latin?" becomes a meta-interpretability question. Needs train/dev split discipline aligned with the leak-free Phase 8+ protocol.

---

## Variant 4 — CounterfactualMask (live, decision-flipping)

**Parameterization.** Two soft token masks parameterized as Gumbel-sigmoid, applied as *attention masks* inside the actual transformer (multiply pre-softmax attention logits by $\log\lambda$). The model genuinely sees a masked input, not a re-pooled hidden state.

**Gradients.** Masks only; model frozen. Gumbel-sigmoid gives a low-variance reparameterized estimator and a hard-mask test-time variant.

**Per-pair vs amortized.** Per-pair, ~30–50 forward+backward passes through the *actual* bi-encoder.

**Loss.** **Decision-flipping**: the mask must move the pair across the *learned assignment threshold*. For a `wrong_similar` pair (model says "same," ground truth "different"):
$$\mathcal{L} \;=\; \mathrm{ReLU}\big(\cos(\tilde h_q(\lambda), \tilde h_c(\lambda)) - \tau + m\big) \;+\; \alpha (1-\bar\lambda)^2$$
i.e., mask the *minimum* token set whose removal pushes similarity below the threshold $\tau$ (read from the calibrated Phase-9 evaluator). For `correct_similar`, do the opposite: find the minimum mask that *preserves* the decision (equivalent to comprehensiveness). The four buckets give four distinct counterfactual objectives, all aligned with our actual evaluation metric.

**Two-tier mapping.** **Live only** — needs the real model forward to honor the masked attention; cached hiddens won't reflect the cascade through later layers. (A degenerate cached-only version exists but loses the point.)

**Differentiation from MaRC.** MaRC explains "which tokens support the predicted label." This explains "which tokens are *responsible for the model being right or wrong* relative to the ground-truth assignment threshold." The objective is calibrated to the deployed decision boundary, not to abstract sufficiency. No prior interpretability paper I am aware of optimizes per-bucket counterfactual masks against a calibrated threshold.

**Risk.** Expensive (per-pair Adam through a real model — back to MaRC's wall clock or worse). Threshold $\tau$ must be frozen from the train split to avoid leakage. Gumbel variance can dominate at $\bar\lambda \approx 0$.

---

## Variant 5 — LowRankAttention (live, rank-$r$ perturbation)

**Parameterization.** Instead of a token mask, parameterize a rank-$r$ perturbation of the final-layer attention matrix: $A'_\ell = \mathrm{softmax}(\mathrm{logits}_\ell + U_\ell V_\ell^\top)$, with $U_\ell, V_\ell \in \mathbb{R}^{n\times r}$, $r \in \{1,2,4\}$. Read out token importance from the row/column norms of $UV^\top$.

**Gradients.** $(U_\ell, V_\ell)$ only; model frozen.

**Per-pair vs amortized.** Per-pair, comparable cost to Variant 4.

**Loss.** ABTT-cosine sufficiency; nuclear-norm regularizer on $UV^\top$.

**Two-tier mapping.** Live only.

**Differentiation from MaRC.** Operates on attention *flow*, not on inputs; produces a directed `(query_token → candidate_token)` attribution structurally similar to the OT transport plan but learned, not computed.

**Risk.** Most exotic of the five; harder to map back to a per-token bar chart for the webapp; risk of becoming more architectural noise than interpretable signal. Strong novelty but the highest "is this even an attribution method?" review risk.

---

## RECOMMENDATION

**Cached tier → Variant 1 (PairMask).**
**Live tier → Variant 4 (CounterfactualMask).**

### Justification

**(i) Novelty vs MaRC and the literature.**
- *PairMask* is the smallest-possible legitimate novelty: replace MaRC's classification log-likelihood with ABTT-cleaned pair cosine. To my knowledge no published method does learned-mask attribution for bi-encoder retrieval in a *post-processed* (ABTT) embedding space — MaRC is classification-only, BERTScore/OT are non-learned, ColBERT-style late interaction is alignment-not-attribution. The loss substitution is small in code, conceptually exact, and directly answers the question "does the learned mask add anything over IG-cosine here." Defensible at ACL: it is a *minimal controlled variation* on MaRC plus a real domain motivation (ABTT).
- *CounterfactualMask* is more ambitious: optimizing a mask against a calibrated decision threshold, *per bucket*, with bucket-conditional objectives. I am not aware of any prior work that ties learned-mask attribution to a calibrated retrieval threshold; the closest is contrastive rationale extraction (Jain & Wallace 2020, FRESH), which uses classification heads. The "explain why the model is *wrong*" framing for the wrong-similar / wrong-not-similar buckets is the strongest novelty pitch — and it directly serves the scholar-review webapp use case (a humanist looking at a `wrong_similar` pair wants to know what tokens to *remove* to fix the model's mistake).

**(ii) Fit to our pipeline.**
- *PairMask* slots in as the seventh builder function in `run_resubmit_ig_comparison.py` next to `build_ig_pair_matrix`, `build_bertscore_matrix`, etc. It uses exactly the NPZ keys that already exist (`query_hidden`, `candidate_hidden`, `pcs`, `mean_vec`) and the same `cosine_matrix` / `clean_tokens` helpers. No new caching pass, no model load, no SLURM job — runs on a laptop. It will produce a `(n_q, n_c)` outer-product matrix `λ^q λ^cᵀ` that drops directly into the existing 6-panel comparison figure and the webapp's `attribution_method` dropdown.
- *CounterfactualMask* needs a real model forward but reuses `EmbeddingCleaner` from `src/sif_abtt.py` and the existing `attribution_targets.py` machinery (the ABTT-norm scalar target is essentially the building block; we just optimize the mask instead of integrating gradients). The four-bucket structure of `phase12f_examples.csv` maps one-to-one onto four loss formulations, so the bucket eval *is* the experimental design. The threshold $\tau$ is already learned by `evaluate_vectors.py` per (model, layer, method) tuple — read it from the existing CSVs.

**(iii) Feasibility.**
- *PairMask*: ~150 LoC (mask logits + closed-form pooling + ABTT in PyTorch, Adam loop, hyperparameter $(\alpha, \beta, \gamma, \tau_{\min})$). One grad student, a few days. Hyperparameters tuned on a held-out subset of the 80 example pairs; main risk is degenerate collapse (mitigated by TV + minimum-mass).
- *CounterfactualMask*: ~400–600 LoC (Gumbel-sigmoid mask layer, hook into attention, per-bucket loss switch, threshold loader, batched Adam). One grad student, ~1–2 weeks. Main feasibility risk is wall-clock on Qwen3-8B; mitigation is to run it only on the 80 pre-selected example pairs (not the full corpus) and only on the 3 smaller models (LaBSE, LaTa, PhilTa) for the headline numbers, with a Qwen-0.6B sanity check.

The two recommendations are *deliberately at different points in the cost/novelty plane*: PairMask is the cheap, defensible, "of course you should try this" variant that makes the cached webapp gallery instantly richer. CounterfactualMask is the swing-for-the-fences pitch that gives the paper a real story beyond "we ported MaRC." Together they cover both tiers without redundancy.

---

**Files inspected for grounding:** `/projects/beto/irowerojas/localLatin/src/sif_abtt.py`, `/projects/beto/irowerojas/localLatin/src/attribution_targets.py`, `/projects/beto/irowerojas/localLatin/scripts/resubmit/run_resubmit_ig_comparison.py` (lines 1–200).
