# Proposal: A Learned-Mask Attribution Method for ABTT-Cleaned Bi-Encoder Retrieval

**Status.** Pre-implementation methodology proposal for discussion at the next meeting. Synthesizes four parallel research lanes (`docs/research/own_method_research/01–04`) into a single recommendation. **Not implemented yet.**

**Authors.** Synthesis by main agent from `surveyor`, `methodologist`, `ablation-architect`, `related-work-weaver` lane outputs. Date: 2026-04-07.

**Anchor citation (the paper we differentiate from).** Brinner & Zarrieß, *Model Interpretability and Rationale Extraction by Input Mask Optimization*, **Findings of ACL 2023**, pp. 13722–13744 (referred to throughout as **MaRC**). The arXiv 2508.11388 deposit is a 2025 re-upload, not a new paper. The authors' EMNLP 2024 follow-up *Rationalizing Transformer Predictions via End-To-End Differentiable Self-Training* amortizes MaRC for **classification only**.

---

## 1. Differentiation hook

Our method is, to our knowledge, the first learned-mask attribution that **jointly optimizes paired query+candidate masks against bi-encoder cosine similarity in an ABTT-cleaned embedding space** — a corner of the design space that the MaRC family (Brinner & Zarrieß 2023, 2024) cannot reach because its fidelity loss targets a classifier head, and that the only published transformer-text-similarity attribution method (BiLRP for text similarity, Vasileiou & Eberle 2024) does not optimize for at all because it has no learned parameters.

## 2. Mathematical specification

We propose a two-tier instantiation. Both tiers share the same fidelity-loss family; they differ in *where the mask is applied* (cached pooled hiddens vs. live attention) and *what supervises it* (sufficiency/comprehensiveness vs. calibrated decision flip).

### Notation

Let a query/candidate pair be `(x_q, x_c)` with cached per-token hiddens `H_q ∈ R^{n_q × d}` and `H_c ∈ R^{n_c × d}` (the `query_hidden`/`candidate_hidden` keys in every NPZ artifact). Let `μ ∈ R^d` and `P ∈ R^{D × d}` be the centring vector and top-D principal components (`mean_vec`/`pcs`), fitted on the train split via `EmbeddingCleaner` in `src/sif_abtt.py`. The ABTT-cleaned mean-pool of a masked side is

$$\tilde h(\lambda; H, m) \;=\; \mathrm{ABTT}\!\Big(\textstyle\frac{\sum_{i:\,m_i=1} \lambda_i\, H_i}{\sum_{i:\,m_i=1} \lambda_i + \varepsilon}\Big), \qquad \mathrm{ABTT}(v) \;=\; (v - \mu) - \sum_{d=1}^{D}\langle v - \mu,\, P_d\rangle\, P_d$$

where `m` is the model's `attention_mask`, `λ ∈ [0,1]^n` is a soft mask, and `ε` prevents zero-division. The pair-similarity score is `s(λ_q, λ_c) := cos(\tilde h(λ_q; H_q, m_q),\, \tilde h(λ_c; H_c, m_c))`. Let `s_full := s(\mathbf 1, \mathbf 1)` and `τ` be the calibrated assignment threshold for the relevant `(model, layer, method)` tuple, read from `runs/active/resubmit/results/*.csv` (column `abtt_tau`, joined via `phase12f_examples.csv`).

### Tier 1 — PairMask (cached, per-pair sufficiency/comprehensiveness)

**Parameterization.** Two independent sigmoid-parameterized soft masks `λ_q = σ(z_q)`, `λ_c = σ(z_c)` with logits `z_q ∈ R^{n_q}`, `z_c ∈ R^{n_c}` initialized at zero. **No Gaussian kernel** — locality is enforced by an explicit total-variation term (cheaper, one fewer hyperparameter than MaRC's bandwidth field).

**Optimization target.**

$$\mathcal{L}_{\text{PairMask}}(z_q, z_c) \;=\; -\,s(\lambda_q, \lambda_c) \;+\; \beta\, s(1-\lambda_q,\, 1-\lambda_c) \;+\; \alpha \big[\bar\lambda\big]^2 \;+\; \gamma \cdot \mathrm{TV}(\lambda)$$

with $\bar\lambda := \tfrac{1}{n_q+n_c}(\sum_i \lambda_{q,i} + \sum_j \lambda_{c,j})$, `TV(λ) := Σ |λ_{i+1} − λ_i|` summed over both sides, and a hard floor `\bar\lambda ≥ τ_{\min}` enforced via barrier (degenerate-collapse guard, per Lane 2's risk note).

**What gets gradients.** Mask logits `(z_q, z_c)` only. The encoder is frozen, the cached hiddens are constants, and the ABTT projection is constant (`μ, P` are pre-fit). PyTorch autograd over the closed-form pooling+ABTT path. Per-pair, ~50–100 Adam steps. **No live model forward.**

**Per-pair vs per-corpus.** Per-pair, ~< 1 s on CPU (the optimization variable is at most ~1024 logits).

### Tier 2 — CounterfactualMask (live, decision-flipping with bucket-conditional objective)

**Parameterization.** Gumbel-sigmoid attention masks injected as additive logits at the **final** transformer layer (with an optional `--mask_layer` ablation, see §5):

$$A'_{q} \;=\; \mathrm{softmax}\!\big(\mathrm{logits}_q + \log \lambda_q\big), \qquad A'_{c} \;=\; \mathrm{softmax}\!\big(\mathrm{logits}_c + \log \lambda_c\big)$$

Hard masks at test time via straight-through estimator.

**Optimization target — bucket-conditional.** Let `b ∈ {correct_similar, correct_not_similar, wrong_similar, wrong_not_similar}` be the pair's bucket label (already in `phase12f_examples.csv`). For each bucket the loss enforces a *decision flip across the calibrated threshold τ*:

$$
\mathcal{L}_b(\lambda) \;=\;
\begin{cases}
\mathrm{ReLU}\big(\,\tau + m - s(\lambda_q,\lambda_c)\,\big) + \alpha (1-\bar\lambda)^2 & b \in \{\text{correct\_similar}\}\\[2pt]
\mathrm{ReLU}\big(\, s(\lambda_q,\lambda_c) - \tau + m\,\big) + \alpha\, \bar\lambda^2 & b \in \{\text{correct\_not\_similar}\}\\[2pt]
\mathrm{ReLU}\big(\, s(\lambda_q,\lambda_c) - \tau + m\,\big) + \alpha (1-\bar\lambda)^2 & b \in \{\text{wrong\_similar}\}\\[2pt]
\mathrm{ReLU}\big(\,\tau + m - s(\lambda_q,\lambda_c)\,\big) + \alpha\, \bar\lambda^2 & b \in \{\text{wrong\_not\_similar}\}
\end{cases}
$$

In plain language: in `correct_*` buckets, find the *minimum* mask whose **removal preserves** the correct decision (high comprehensiveness — explains the supporting evidence). In `wrong_*` buckets, find the *minimum* mask whose **removal flips the wrong decision toward the gold label** (counterfactual — explains the spurious evidence). The four buckets give four distinct counterfactual semantics, all anchored to the calibrated retrieval threshold the paper already reports.

**What gets gradients.** Mask logits only; the bi-encoder is frozen. ~30–50 forward+backward passes per pair.

**Per-pair vs per-corpus.** Per-pair, ~90 s on a single A100 for BERT-base-sized models.

### What we deliberately do *not* propose (and why)

- **Amortization-only as the headline.** Lane 1 documented that Brinner & Zarrieß **EMNLP 2024** already amortized MaRC's mask predictor for classification. "Amortize MaRC" alone is therefore no longer a defensible novelty. We instead pin the novelty on the *task* (retrieval/cosine) and *structure* (joint `(λ_q, λ_c)` paired masks) — and treat amortization as a follow-up *conditional on* PairMask validating the embedding-space objective.
- **Manuscript-aware distance kernels** (the "secondary gap" Lane 1 flagged). Genuinely novel because no prior mask method has used a non-sequential distance kernel, but a smaller methodological contribution that should reinforce the paper *only if* the primary contribution lands. Held in reserve.
- **MaRC-2024 amortized as a head-to-head baseline.** Lane 3's stated reason ("we lack labeled mask targets") is incorrect — MaRC-2024 trains end-to-end on classification labels, no mask labels needed. The *real* reason it can't be a head-to-head is task mismatch: it requires a classification head we do not have. The honest move is to cite it as related work and not re-implement it. (Re-implementing it on a contrastive retrieval objective would essentially recover Lane 2's `AmortizedHead` Variant 3 — that becomes our future-work item.)

## 3. Pseudocode — PairMask for one cached pair

The Tier 1 method uses *only* the keys already saved in every NPZ artifact under `runs/active/ig_examples/artifacts/<model_slug>/example*_pair_example.npz`. Listing is intentionally numpy/torch-only — no model load, no tokenization, no dependency on `transformers`.

```python
import numpy as np, torch

def pairmask_attribute(npz_path, alpha=1.0, beta=1.0, gamma=0.05,
                       tau_min=0.1, n_steps=80, lr=0.5):
    """
    Returns soft masks (lambda_q, lambda_c) and a (n_q, n_c) outer-product
    attribution matrix that drops into run_resubmit_ig_comparison.py's
    `pair_matrix_*` slot.  Cached-only: no live model.
    """
    d = np.load(npz_path)
    H_q  = torch.from_numpy(d["query_hidden"]).float()      # (n_q, dim)
    H_c  = torch.from_numpy(d["candidate_hidden"]).float()  # (n_c, dim)
    m_q  = torch.from_numpy(d["query_attention_mask"]).bool().squeeze()
    m_c  = torch.from_numpy(d["candidate_attention_mask"]).bool().squeeze()
    mu   = torch.from_numpy(d["mean_vec"]).float()          # (dim,)
    P    = torch.from_numpy(d["pcs"]).float()               # (D, dim)

    def abtt_pool(H, lam, mask):                            # closed-form ABTT cosine input
        w   = (lam * mask.float()).unsqueeze(-1)
        h   = (w * H).sum(0) / (w.sum() + 1e-6)
        c   = h - mu
        return c - (P @ c).unsqueeze(0).matmul(P).squeeze(0)

    z_q = torch.zeros(H_q.shape[0], requires_grad=True)
    z_c = torch.zeros(H_c.shape[0], requires_grad=True)
    opt = torch.optim.Adam([z_q, z_c], lr=lr)

    for _ in range(n_steps):
        lam_q, lam_c = torch.sigmoid(z_q), torch.sigmoid(z_c)
        s_keep = torch.cosine_similarity(abtt_pool(H_q, lam_q, m_q),
                                         abtt_pool(H_c, lam_c, m_c), dim=0)
        s_drop = torch.cosine_similarity(abtt_pool(H_q, 1-lam_q, m_q),
                                         abtt_pool(H_c, 1-lam_c, m_c), dim=0)
        bar    = ((lam_q*m_q).sum()+(lam_c*m_c).sum()) / (m_q.sum()+m_c.sum())
        tv     = (lam_q[1:]-lam_q[:-1]).abs().sum() + (lam_c[1:]-lam_c[:-1]).abs().sum()
        floor  = torch.relu(tau_min - bar) ** 2             # barrier guard
        loss   = -s_keep + beta*s_drop + alpha*bar**2 + gamma*tv + 10.0*floor
        opt.zero_grad(); loss.backward(); opt.step()

    lam_q = torch.sigmoid(z_q).detach().numpy() * m_q.numpy()
    lam_c = torch.sigmoid(z_c).detach().numpy() * m_c.numpy()
    pair_matrix = np.outer(lam_q, lam_c)                    # drop-in for pair_matrix_*
    return lam_q, lam_c, pair_matrix
```

Total: ~30 lines including the helper. CounterfactualMask (Tier 2) adds a Gumbel-sigmoid layer and a transformer forward inside the optimization loop, plus a bucket-conditional loss switch — ~150 LoC, deferred to a separate listing in the implementation PR.

## 4. Plug-in plan — adding this as the 7th method to `run_resubmit_ig_comparison.py`

The existing scaffolding makes Tier 1 a near-zero-friction addition. Verified during planning:

### Files to touch (no rewrites; pure additions)

1. **`scripts/resubmit/run_resubmit_ig_comparison.py`** — add a new builder function next to `build_ig_pair_matrix`, `build_bertscore_matrix`, etc.:

   ```python
   def build_pairmask_matrix(query_hidden_clean, candidate_hidden_clean,
                             query_attention_mask, candidate_attention_mask,
                             pcs, mean_vec, **hp):
       """7th attribution method: PairMask (learned soft mask, ABTT-cosine fidelity)."""
       # body = the function in §3 above, but operating on the in-memory tensors
       # returned by the per-pair loader (so we don't re-load the NPZ).
       ...
       return pair_matrix  # (n_q_clean, n_c_clean) numpy array
   ```

   This signature mirrors the other builders. The function returns a `(q_masked_len, c_masked_len)` matrix exactly like the other 6 methods, so all downstream consumers (top-k extraction, webapp gallery, faithfulness eval) get it for free.

2. **`scripts/resubmit/persist_attribution_methods.py`** — register the new method in two places:

   - Append `"pairmask"` to the `MAIN_METHODS` list near the top of the file.
   - Append `("pairmask", build_pairmask_matrix_baseline, build_pairmask_matrix_abtt)` to the `method_builders` tuple list around line 147.

   The per-pair loop already iterates `method_builders` and writes `pair_matrix_pairmask_baseline`, `pair_matrix_pairmask_abtt`, `topk_pairmask_baseline_query`, `topk_pairmask_baseline_candidate`, `topk_pairmask_abtt_query`, `topk_pairmask_abtt_candidate` to each NPZ. **No new IO code is needed.** The CSV `methods_available` column is auto-rebuilt with `pairmask` appended.

3. **No changes to** `src/sif_abtt.py`, `src/attribution_targets.py`, or any extraction CLI. The PairMask builder reads ABTT components straight from the cached NPZ; `EmbeddingCleaner` is reused implicitly because the same `pcs`/`mean_vec` were stored when the cache was built.

### Webapp integration

The frontend's attribution-method dropdown is populated from the `methods_available` column of `phase12f_examples.csv` (no hard-coded list). Once the new method appears in that column, the dropdown auto-includes it and renders the `pair_matrix_pairmask_*` heatmap exactly like the 6 existing methods. **No frontend changes required.**

### What Tier 2 adds to the plug-in plan

CounterfactualMask cannot live in the cached pipeline because it needs a live model forward. Proposed structure:

- New SLURM script `slurm/resubmit/run_counterfactual_mask.sbatch` (parallel to the IG extraction sbatches), one per `(model, layer)` tuple, batched over the 80 example pairs.
- New CLI `scripts/resubmit/run_counterfactual_mask.py` that loads the model, attaches the Gumbel-sigmoid attention-mask hook, runs the bucket-conditional optimizer per pair, and writes results to a *separate* NPZ key namespace `cf_mask_*` so it doesn't collide with the cached `pair_matrix_*`.
- A second `persist_attribution_methods.py` invocation merges the `cf_mask_*` outputs into the canonical artifact NPZs after the live run.

This deliberately keeps Tier 1 in CPU/cache land and Tier 2 in GPU/SLURM land, so the cheap tier can ship without waiting on the expensive one.

## 5. Evaluation plan

Direct port of Lane 3's protocol, condensed to the headline rows. Full table in `docs/research/own_method_research/03_ablation_design.md`.

### 5.1 Baselines (essential set)

| # | Baseline | Variant | Why |
|---|---|---|---|
| B0 | Random mask | — | sanity floor |
| B1 | Uniform / no mask | — | sufficiency ceiling |
| B2 | All-zero mask | — | comprehensiveness floor |
| B3 | IG (×2) | baseline + ABTT | grounds the "ABTT helps attribution" claim |
| B4–B8 | BERTScore-greedy, OT, Attention-Weighted, DLA, Attention-Standalone | ABTT only | the existing 6-method comparison |
| B9 | **Top-IG-as-hard-mask** | ABTT | the critical "any mask" control: distinguishes "learned mask" from "any mask, even a hand-crafted one" |
| B11 | **MaRC-2023 re-implementation** | ABTT | the methodological head-to-head; uses the public github.com/inas-argumentation/Explainability code as a base |
| B12 | MaRC-2024 (amortized for classification) | — | **cite as related work, not re-implemented** (task mismatch, see §2 "what we deliberately do not propose") |

### 5.2 Faithfulness metrics — adapted to bi-encoder cosine

All metrics defined in **ABTT-cleaned cosine space**. `s_full := cos(ABTT(mean_pool(H_q)), ABTT(mean_pool(H_c)))`, `τ` from `phase12f_examples.csv`. For mask `λ`, the masked similarity is computed by zero-ing out non-top-k token rows of `H_q`/`H_c` (not by re-pooling with `Σλ` in the denominator — keeps comparability across `k`).

| Metric | Formula | Direction |
|---|---|---|
| **Sufficiency@k** | `(s(top_k(λ_q), top_k(λ_c)) − τ) / (s_full − τ)` | ↑ = top-k% preserves the decision margin |
| **Comprehensiveness@k** | `(s_full − s(1−top_k(λ_q), 1−top_k(λ_c))) / (s_full − τ)` | ↑ = removing top-k% drops sim to threshold |
| **AOPC** | mean over `k ∈ {5,10,20,30,50}` | single-number summary for headline table |
| **Decision-flip rate @ Comp@20** | fraction of pairs where masking the top-20% flips the prediction across τ | bi-encoder analogue of classification flip-rate; this is the metric that maps faithfulness back to **Assignment Accuracy**, our paper's primary metric |
| **Asymmetric sufficiency** | `Suff(k)` with one side fully unmasked | catches one-sided spurious mass |
| **Log-odds Δ** | `z_train(s_masked) − z_train(s_full)` | classification-log-odds analogue, computed by z-scoring against the train cosine distribution |

### 5.3 Per-component ablations (PairMask)

Run on M1 (PairMask) only unless marked. Priority labels are P0 = must, P1 = strong, P2 = nice. Full bucket-level expectations and falsifiable mechanisms in Lane 3 §D.

| Ablation | Setting | Targets |
|---|---|---|
| A1: sparsity off | `α = 0` | confirms the L1 prior matters (Comp drops) — **P0** |
| A2: continuity off | `γ = 0` | confirms TV produces real spans (T5 sentencepiece most affected) — **P0** |
| A3: ABTT-in-loss off | optimize raw cosine | the **key retrieval-aware claim** — anisotropy dip should return on decoder models — **P0** |
| A4: sufficiency-only | `β = 0` | targets `wrong_similar` bucket — **P0** |
| A6: pair-sim vs class-proxy | replace cosine target with DLA logit target | **falsifies** "need pair objective" — **P0** |
| A7: independent vs joint | tie `λ_q ← λ_c` | tests whether jointness adds value or just doubles parameters — **P1** |
| A11 [Tier 2]: bucket-conditional vs uniform | 4-head loss vs one | **the Tier-2 novelty claim**; without this, Tier 2 collapses to "MaRC with a cosine loss" — **P0** |

### 5.4 Statistical rigor

- **Pairing.** 80 pairs, every method on every pair → **Wilcoxon signed-rank** (paired) for headline comparisons, with rank-biserial effect size.
- **Bootstrap CIs.** 10,000 resamples over pairs, BCa intervals on AOPC and FlipRate.
- **Multiple comparisons.** Primary tests = ours vs {MaRC-2023, IG-ABTT, Top-IG-as-mask, BERTScore} × {AOPC_suff, AOPC_comp, FlipRate} = 12 tests. **Holm–Bonferroni** within each metric column. Bucket-level breakdowns are exploratory and reported unadjusted (with the note made visible).
- **Per-model.** Pool in headline with model as a Wilcoxon blocking factor; per-model breakdown in supplement, mandatory for Qwen3-0.6B (only decoder, distinct anisotropy profile).
- **Power.** 80 paired samples gives ~80% power at α=0.004 for an effect of `d ≈ 0.32` → PairMask must beat MaRC-2023 by ≥ 0.04 AOPC in absolute terms. **Pre-register a contingency scale-up to 200 pairs** (balanced across buckets) if the gap is borderline.

### 5.5 Falsification gates (in cheapest-first order)

1. PairMask vs Top-IG-as-hard-mask: AOPC_suff improvement < 0.02 on ≥3 of 4 models → cached tier is "IG with extra steps"; abandon Tier 1.
2. PairMask vs Uniform mask: `Suff@20 < 1.1 × Uniform@20` → method isn't concentrating signal; kill.
3. **A3 ablation (ABTT-in-loss):** if removing ABTT makes it *better* on any model → the retrieval-aware framing is wrong; pivot the paper.
4. Spearman ρ(PairMask, IG-ABTT) > 0.85 → we're rediscovering IG; reframe as "distilled IG", not "new signal".
5. **A11 ablation (bucket-conditional, Tier 2 only):** if uniform-objective ≈ bucket-conditional within noise → drop the Tier-2 novelty claim entirely.
6. MaRC-2023 wins on ≥3 of 4 buckets on AOPC_comp → honest pivot to "MaRC transfers to retrieval", still publishable.

### 5.6 Compute estimate

| Component | Cost |
|---|---|
| **Tier 1 headline** (PairMask + cached baselines + faithfulness eval) | **~15 min on CPU** |
| Falsification gates 1–4 | trivial after Tier 1 |
| MaRC-2023 re-implementation × 4 models × 80 pairs | **~12 GPU-hr overnight (1 × A100)** |
| PairMask ablations A1–A6 | ~30 min (CPU, ~5 min per ablation) |
| **Tier 2** (CounterfactualMask + A11–A13) | **~8 GPU-hr** |
| **Tier 1 + Tier 2 + MaRC-2023** | **~20 GPU-hr** |

**Hard gate:** reconvene after Tier 1 + falsification gates 1–4 (~30 min wall) before spending the 12 GPU-hr on MaRC-2023. If PairMask falsifies at gate 1 or gate 2, MaRC re-implementation is unnecessary.

## 6. Positioning paragraph (verbatim — drop into related work)

Our method belongs to the family of **learned-mask, per-instance rationale extractors** initiated by MaRC (Brinner and Zarrieß 2023) and amortised for classification by their EMNLP follow-up (Brinner and Zarrieß 2024). Both target a frozen classifier with a log-likelihood fidelity loss and a sequential `|i−j|` distance prior; neither addresses sentence-pair similarity. The closest published work on attribution for transformer-based text similarity is BiLRP (Vasileiou and Eberle 2024), a gradient-propagation method with no learned parameters, sparsity prior, or post-hoc geometric correction. The corresponding corner — a learned, jointly optimised `(λ_q, λ_c)` mask whose fidelity loss is pair cosine similarity in an ABTT-cleaned bi-encoder space — has, to our knowledge, not been studied. We address it with two complementary tiers: PairMask, a cached per-pair optimiser over ABTT-cleaned cosine similarity, and CounterfactualMask, a live Gumbel-sigmoid attention mask with bucket-conditional objectives that flip the calibrated Assignment-Accuracy decision.

*(149 words. Verified by Lane 4 against ACL-style author-year format. The "to our knowledge" hedge is deliberate — Lane 1 flagged that ICLR/NeurIPS workshop venues and SIGIR/ECIR/CIKM 2024–2026 were not exhaustively searched.)*

## 7. Risks and open questions

### R1. Lane 1's literature hedge is real

We could not extract the full text of **Brinner & Zarrieß EMNLP 2024** during the surveyor sprint (PDF binary on the anthology mirror, OpenReview gated). The abstract and metadata point to *classification-only*, but a careful reading of §3 (loss function) and §4 (experiments) is required before submission. **If** that paper secretly evaluates on STS or any pair-similarity benchmark, our gap shrinks from "open corner" to "open in retrieval but not STS" — still publishable, but the related-work paragraph needs a sharper hedge.

### R2. "Is the cached tier *real* enough?"

A reviewer will reasonably ask whether PairMask, which never runs the model, is a "real" learned-mask attribution method or a glorified embedding-space reweighting. The honest answer is that it is the **embedding-space analog** of the input-mask family: it reweights what the encoder *already produced*, rather than re-running the encoder under a counterfactual input. This is a meaningful interpretability question (which dimensions of the cached representation drive the cosine score?) but it is **not** equivalent to MaRC's question (what would happen if I re-ran the encoder with these tokens removed?). The mitigation is to **always pair PairMask with CounterfactualMask** in the headline table — Tier 2 answers the "real model" question, and Tier 1's role is to be the cheap-and-defensible counterpart.

### R3. ABTT-mask interaction risk

ABTT removes the top-D principal components from the *unmasked* pooled embedding. PairMask then optimizes in ABTT-cleaned space. There is a degenerate solution where the mask discovers the directions that ABTT *already removed* and concentrates mass there (i.e., it "rediscovers anisotropy" that ABTT supposedly cleaned). The A3 ablation (ABTT-in-loss off) is the diagnostic, but the falsification gate 3 needs to be watched closely: a PairMask that beats no-ABTT on average could still be exploiting a degenerate corner per-pair. **Open question:** should we add an orthogonality regularizer that penalizes mask configurations whose pooled embedding lies near the removed PC subspace?

### R4. Sample size

80 pairs is small. The power calculation in §5.4 says we need a ≥ 0.04 AOPC gap vs MaRC-2023 to clear Holm-corrected α. **Pre-register** the 200-pair contingency scale-up in the proposal so we are not seen to be p-hacking the bucket sizes after looking at preliminary results. The 200-pair pool is feasible by extending the existing `phase12f` selection script with `--n_per_bucket 50`.

### R5. Variant naming and terminology lock-in

"PairMask" and "CounterfactualMask" come from Lane 2 and have already propagated to Lane 4's positioning paragraph. If the meeting picks different names, **find-and-replace across all 5 files at once** to avoid drift. The acronym **PMA** (PairMask Attribution) is one possible compact form for the paper.

### R6. The amortization fork is still on the table

Lane 1 was strongest on amortization (the un-amortized version exists in MaRC-2023; the amortized version exists for classification in MaRC-2024; the amortized-for-retrieval corner is empty). Lane 2 deliberately picked per-pair variants for the proposal, but Variant 3 in `02_methodologist_variants.md` (`AmortizedHead`) is a fully-specified alternative we should treat as **the natural follow-up paper** if PairMask validates the embedding-space objective. The mention of it in §2 is intentional — we want it visible to the advisor as the next step, not buried.

### R7. MaRC-2023 re-implementation cost

12 GPU-hours overnight is fine, but only if the public github.com/inas-argumentation/Explainability code transfers cleanly to our model checkpoints. A pre-flight 30-min smoke test on LaBSE before committing to the full sweep is mandatory. **Open question:** if the MaRC code is BERT-only and doesn't trivially transfer to T5/Qwen, do we (a) only run MaRC on LaBSE and report it as a single-model spot-check, or (b) spend a week porting? The proposal currently assumes (a); the meeting should explicitly approve.

### R8. Webapp consequences

PairMask producing an `outer(λ_q, λ_c)` matrix may render very *flat* in the existing 6-panel heatmap because the mask is sparse. This is a UX question, not a science one, but the existing webapp gallery viewer should be tested with a synthetic sparse matrix before the real PairMask outputs land — otherwise reviewers may see "broken-looking" visualizations on first inspection.

---

## Sister documents (lane outputs)

- `docs/research/own_method_research/01_surveyor_landscape.md` — literature map, frontier scan, gap statement, citation list (1,460 words)
- `docs/research/own_method_research/02_methodologist_variants.md` — five variant specs + recommendation (1,490 words)
- `docs/research/own_method_research/03_ablation_design.md` — full evaluation protocol with ablation table, statistical rigor, falsification gates, compute estimate (~2,100 words)
- `docs/research/own_method_research/04_positioning.md` — 2-sentence pitch, 11 per-rival differentiation lines, 150-word positioning paragraph (~830 words)

This synthesis is **not** a verbatim concatenation. Where lanes disagreed (notably on amortization vs per-pair, and on the feasibility of MaRC-2024 as a head-to-head baseline), §2 picks a side and documents the reasoning.
