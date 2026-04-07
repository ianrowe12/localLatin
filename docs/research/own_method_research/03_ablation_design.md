# 03 — Ablation & Evaluation Protocol (Lane 3)

**Purpose.** Spec the experiment that would let the advisor say "yes, that would convince me" for a learned-mask attribution method on Latin retrieval. Everything is per-pair faithfulness; Assignment Accuracy is only used to set the decision threshold `τ` inside the counterfactual loss.

**Evaluation surface.** 80 pairs in `runs/active/ig_examples/phase12f_examples.csv` (4 models × 4 buckets × 5 pairs). Each pair has its model's per-layer `abtt_tau` and `D`. All faithfulness metrics are computed in **ABTT-cleaned cosine space** using the `pcs` / `mean_vec` already cached in every NPZ.

---

## A. Baselines — what gets compared

Let `M` be the method that produces a per-token importance vector `s_q ∈ R^{n_q}`, `s_c ∈ R^{n_c}` (for pair-matrix methods we take row/col marginals, which is already what `topk_*_query`/`_candidate` keys do).

| # | Baseline | Variant | Tier | Essential? |
|---|---|---|---|---|
| B0 | Random mask | — | trivial | **must** (sanity floor) |
| B1 | Uniform mask (no masking) | — | trivial | **must** (upper anchor for sufficiency) |
| B2 | All-zero mask | — | trivial | **must** (lower anchor for comprehensiveness) |
| B3 | IG | baseline + ABTT | existing | **must** (6 methods × 2 variants = 12 total, see note) |
| B4 | BERTScore-greedy | ABTT only | existing | **must** |
| B5 | Optimal Transport | ABTT only | existing | **must** |
| B6 | Attention-Weighted (Ditto) | ABTT only | existing | **must** |
| B7 | DLA | ABTT only | existing | **must** |
| B8 | Attention-Standalone | ABTT only | existing | nice |
| B9 | Top-IG-as-hard-mask | ABTT | "any mask" control | **must** |
| B10 | Top-attention-as-hard-mask | ABTT | "any mask" control | nice |
| B11 | MaRC-2023 (re-implemented) | ABTT | new | **must** |
| B12 | MaRC-2024 EMNLP (amortized) | ABTT | new | **nice** |
| M1 | **PairMask** (Tier 1, cached) | ABTT | ours | **must** |
| M2 | **CounterfactualMask** (Tier 2, live) | ABTT | ours | stretch |

**On "baseline vs ABTT" for the 6 existing methods:** run both variants only for IG (B3) so we can ground the "ABTT helps attribution, not just retrieval" claim. For B4–B8 report ABTT-only. That gives **12→7 method-variant rows** on the existing-baseline side and keeps the headline table under control.

**On MaRC-2024 (amortized) feasibility.** The 2023 paper is the methodological competitor — per-instance optimization, same structural assumption as PairMask. It *must* be re-implemented (it's ~200 LoC adapted from github.com/inas-argumentation/Explainability). The 2024 follow-up amortizes via a trained head; we would need a labeled train set of mask targets which we do not have. **Recommendation:** cite 2024 as related, only re-implement if Lane 2 chooses an amortized variant. Otherwise the 2023 re-implementation satisfies the "fair head-to-head with MaRC family" bar.

**MaRC-2023 hyperparameters.** Keep the paper's defaults (Gaussian kernel σ=1, Adam lr=0.1, 100 steps, λ_sparsity from appendix). Live-model required; batch over the 80 pairs on a single A100 (one model at a time to avoid VRAM swaps).

---

## B. Faithfulness metrics — adapted to bi-encoder cosine

All metrics operate on **ABTT-cleaned cosine**. Define:

```
sim(x_q, x_c) := cos( ABTT(mean_pool(x_q)), ABTT(mean_pool(x_c)) )
ABTT(v)       := (v − μ) − Σ_{d=1..D} ⟨v−μ, p_d⟩ p_d      # μ, p_d, D from NPZ
```

For a mask `λ ∈ [0,1]^n`, "masked hidden" is `λ ⊙ H` (elementwise along the token axis). `H` is `query_hidden`/`candidate_hidden` from the NPZ; `λ = 0` sends that token's hidden to zero, so the mean-pool denominator uses `attention_mask` not `sum(λ)` (keeps comparability to the unmasked case).

Let `s_full = sim(H_q, H_c)`, `s_zero = sim(0·H_q, 0·H_c)`, `τ = abtt_tau` (from the CSV row).

**B.1 Sufficiency@k.** Keep top-k% of tokens by mask score on each side, zero the rest:

```
Suff(k) = ( sim(H_q ⊙ top_k(λ_q), H_c ⊙ top_k(λ_c)) − τ )
        / ( s_full − τ )
```

Higher is better; 1.0 = top-k% preserves the decision margin perfectly, 0 = collapses to threshold, <0 = flipped below τ.

**B.2 Comprehensiveness@k.** Zero the top-k%, keep the rest:

```
Comp(k) = ( s_full − sim(H_q ⊙ (1−top_k(λ_q)), H_c ⊙ (1−top_k(λ_c))) )
        / ( s_full − τ )
```

Higher is better; 1.0 = removing top-k% drops sim all the way to τ.

**B.3 AOPC.** Sweep `k ∈ {5, 10, 20, 30, 50}` percent, report `AOPC_suff = mean_k Suff(k)` and `AOPC_comp = mean_k Comp(k)`. Single-number summary for the headline table.

**B.4 Log-odds analogue.** Classification log-odds is `log P(c|x̃) − log P(c|x)`. For cosine, the natural analogue is the **margin-to-threshold ratio in z-space**. We z-score cosine against the train-set distribution for that `(model, layer)` (use `runs/active/resubmit/results/*.csv` or recompute from train bases). Define:

```
LogOddsΔ(k) = z(s_masked) − z(s_full)                       (should be ≈ 0 for Suff, very negative for Comp)
```

Report on comprehensiveness only — that's where the decision flips.

**B.5 Decision-flip rate (new, bi-encoder specific).** Fraction of pairs where `Comp@20` flips the prediction across `τ` (matches/doesn't match the bucket's gold label). This is the metric that maps attribution quality to **Assignment-Accuracy** faithfulness. Per-bucket breakdown — in `correct_*` buckets we want a flip (shows the mask found the decision-relevant tokens); in `wrong_*` buckets we want to measure whether the mask identifies the tokens driving the *wrong* call.

**B.6 Asymmetric sufficiency.** Compute `Suff(k)` on the query side only (mask `λ_q`, unmasked `c`) and candidate side only. Tests whether the method is picking one-sided spurious tokens.

---

## C. Sparsity & continuity diagnostics

| Diagnostic | Definition | Purpose |
|---|---|---|
| **Density** | `(1/n) Σ_i λ_i` after top-k thresholding | Controls for "cheap" methods that just keep everything |
| **Span coherence** | mean run-length of contiguous kept tokens / random-baseline expectation | Does our continuity regularizer actually produce spans? |
| **IG correlation** | Spearman ρ between our mask and IG-ABTT token importance, per pair, then averaged | Are we just rediscovering IG? |
| **Cross-method agreement** | 7×7 Spearman ρ matrix (mean over pairs), plus per-bucket matrices | Shows which methods cluster; isolates outliers |
| **Content-token focus** | fraction of top-k mass on non-punctuation / non-stopword tokens | Leverages existing `is_content_token` helper in `run_resubmit_ig_comparison.py` |

Success signal: our method has low-to-moderate IG correlation (ρ ≈ 0.3–0.6) — high enough to be plausible, low enough to be adding something.

---

## D. Per-component ablation table (our method)

Rows below are run **only on PairMask (M1)** unless marked [Tier2]. Metric columns: `AOPC_suff`, `AOPC_comp`, `FlipRate`, `Density`, `SpanCoh`. Priority P0=must, P1=strong, P2=nice.

| # | Ablation | Setting | Expected effect (buckets) | Mechanism | Prio |
|---|---|---|---|---|---|
| A0 | Full method | all on | best AOPC, moderate density | — | P0 |
| A1 | Sparsity λ=0 | no L1 on mask | Density↑↑, AOPC_suff≈flat, AOPC_comp↓ | without sparsity pressure, Suff is cheap (keep everything) but Comp loses because removing top-k removes less distinctive mass | P0 |
| A2 | Continuity λ=0 | no TV reg | SpanCoh→random, AOPC similar on LaBSE, AOPC↓ on T5 | T5 sentencepiece fragments need continuity more than wordpiece; `correct_*` buckets most affected | P0 |
| A3 | ABTT-in-loss OFF | optimize raw cosine | FlipRate↓ in `correct_similar`, AOPC_suff↓ on decoder models | anisotropy dip returns; decoder models (Qwen) hit hardest — this is the key "retrieval-aware" claim | P0 |
| A4 | Suff-only loss | no comp term | AOPC_comp↓↓, AOPC_suff≈ | targets the `wrong_similar` bucket: without comp, mask cannot distinguish "enough to keep similar" from "drives false match" | P0 |
| A5 | Comp-only loss | no suff term | AOPC_suff↓↓, AOPC_comp≈ | symmetric control to A4 | P1 |
| A6 | Pair-sim vs class-proxy | replace sim loss with DLA logit | AOPC drops on `correct_not_similar` bucket (no cross-token signal) | DLA is single-encoder; loses cross-pair info — falsifies "need pair objective" | P0 |
| A7 | Independent q/c vs joint | tie `λ_q ← λ_c` via attention | worse on `wrong_similar`, better density | forces the mask to discover asymmetric evidence; targets buckets where q and c disagree in length | P1 |
| A8 | Mask initialization | uniform 0.5 vs IG-init vs attn-init | IG-init → faster convergence, same endpoint if well-specified | sanity check on optimization landscape | P2 |
| A9 | Num Adam steps | {10, 50, 150, 500} | elbow around 50 | compute/quality tradeoff | P1 |
| A10 | Temperature of sigmoid | {0.5, 1, 2} | sharper → better Comp, worse Suff | regularizes hard vs soft | P2 |
| A11 [Tier2] | Bucket-conditional vs uniform | 4-head loss vs one | uniform → `wrong_*` buckets regress | tests whether conditioning on predicted bucket helps — this is the Tier-2 novelty claim | P0 |
| A12 [Tier2] | Gumbel vs continuous sigmoid | τ∈{0.1, 0.5, 1.0} straight-through | Gumbel → better FlipRate, worse SpanCoh | tests whether hard masking is needed for the counterfactual objective | P1 |
| A13 [Tier2] | Live mask layer | {final, final−2, mid-anisotropy} | anisotropy-layer best on LaBSE | validates that the method benefits from acting inside the dip | P1 |

**Why bucket-level "expected effect"?** Buckets are where the science lives: `correct_similar` tests the method's ability to find supporting evidence, `correct_not_similar` tests the ability to find *distinguishing* evidence (the hard case for IG), `wrong_similar` tests whether the method can reveal spurious matches, `wrong_not_similar` tests mask behavior under low-confidence correct rejections.

---

## E. Statistical rigor

- **Sample.** 80 pairs is small but *paired* (every method runs on the same pairs). Use **Wilcoxon signed-rank** per-pair for every pairwise method comparison on AOPC_suff, AOPC_comp, FlipRate. Report effect size (rank-biserial).
- **CIs.** Bootstrap (B=10 000) over pairs for each method's AOPC. Report 95% BCa intervals in the headline table.
- **Multiple comparisons.** Primary comparison = our method vs each of {MaRC-2023, IG-ABTT, Top-IG-as-mask, BERTScore}. That's 4 tests × 3 metrics = 12 p-values. Apply **Holm–Bonferroni** within each metric column. The bucket-level tests (4× more) are exploratory — report unadjusted + a visible note.
- **Per-model.** Report per-model in the ablation supplement (4 mini-tables), **pool in the headline table** with model as a blocking factor in the Wilcoxon. If pooled effect is driven by one model, flag it and show the breakdown. Non-negotiable: per-model breakdown for Qwen3-0.6B, because it's the only decoder and its anisotropy profile differs.
- **Power check.** With 80 paired samples, Wilcoxon has ~80% power to detect an effect of d≈0.32 at α=0.004 (Holm-corrected). Translate: our method must beat MaRC by ≥0.04 AOPC in absolute terms to be publishable. If the gap is smaller, we need more pairs — *pre-register* a scale-up to 200 pairs (balanced across buckets) as a contingency.

---

## F. Falsification — specific shut-down criteria

Run these checks in order; stop if any fire.

1. **PairMask vs Top-IG-as-hard-mask.** If PairMask's AOPC_suff improvement is `< 0.02` on ≥3 of 4 models → the cached tier is "just IG with extra steps". Drop PairMask, move effort to Tier 2 only.
2. **PairMask vs Uniform mask.** If `Suff@20` for PairMask is `< 1.1 × Uniform@20`, the method is not concentrating signal; kill.
3. **ABTT-in-loss ablation (A3).** If removing ABTT from the loss makes things *better* on any model, we've mis-stated the retrieval-aware claim; revisit the framing entirely.
4. **IG correlation too high.** If Spearman ρ(PairMask, IG-ABTT) > 0.85 on average → we're rediscovering IG. Not fatal, but must be surfaced and the positioning must shift from "new signal" to "distilled IG".
5. **Bucket-conditional adds nothing (A11, Tier 2 only).** If A11 uniform-objective matches bucket-conditional within noise, drop the bucket-conditional claim — it's the primary Tier-2 novelty, and without it Tier 2 collapses to "MaRC with a cosine loss".
6. **MaRC-2023 wins outright.** If MaRC beats ours on ≥3 of 4 buckets on AOPC_comp, pivot the paper to "MaRC transfers to retrieval" rather than "new method". Honest fallback, still publishable.

---

## G. Compute estimate

| Component | Per pair | 80 pairs × 4 models | Total |
|---|---|---|---|
| PairMask optimization (150 Adam steps, embedding-space) | ~0.5 s | 160 s | **~3 min** (CPU fine) |
| All 6 existing methods (cached) | instant | instant | **0** (already done) |
| MaRC-2023 re-impl (live, 100 steps, BERT-base-sized) | ~2 min | ~160 min/model | **~11 hr** (A100, 4 models, serial) |
| MaRC-2023 on Qwen3-0.6B & PhilTa (larger) | ~4 min | ~320 min each | **+~11 hr** (included above if we run 4 models) |
| Faithfulness eval suite (sweep k, recompute ABTT-cosine) | ~0.1 s | ~30 s total | **~1 min** |
| Bootstrap CIs (B=10 000 × ~15 methods × 3 metrics) | — | — | **~5 min** (numpy) |
| **Tier 1 total** (PairMask + all baselines + MaRC) | — | — | **~12 GPU-hr + ~10 CPU-min** |
| CounterfactualMask (Tier 2, live, ~500 fwd/back) | ~90 s | ~120 min/model | **~8 GPU-hr** |
| MaRC-2024 amortized (if pursued, training head) | — | ~4 hr train + 10 min eval | **~4 GPU-hr/model = 16 GPU-hr** |
| **Tier 1 + Tier 2 total** | — | — | **~20 GPU-hr** |
| **All (incl. MaRC-2024)** | — | — | **~36 GPU-hr** |

**Recommended execution order** (cheapest/most informative first):

1. PairMask full + all cached baselines + faithfulness eval (~15 min wall). **This alone produces the headline table.**
2. Falsification checks 1–4 (trivial once step 1 is done).
3. MaRC-2023 re-implementation + eval (~12 GPU-hr overnight).
4. Ablations A0–A6 on PairMask (~30 min; each ablation is another PairMask run).
5. If still promising: Tier 2 (CounterfactualMask + A11–A13).
6. MaRC-2024 only if reviewer requests.

**Gate:** after step 2, reconvene before spending the 12 GPU-hr on MaRC-2023. If PairMask already falsifies at step 2, we're done and MaRC re-impl is unnecessary.

---

## Artifact checklist for the evaluator

Before running anything, confirm from an NPZ (pick `artifacts/sentence-transformers_LaBSE/example001_pair_example.npz`) that these keys exist: `query_hidden`, `candidate_hidden`, `query_attention_mask`, `candidate_attention_mask`, `pcs`, `mean_vec`, `query_ig_abtt`, `candidate_ig_abtt`, `topk_ig_abtt_query`, `topk_ig_abtt_candidate`. Pull `abtt_tau` from the matching row in `phase12f_examples.csv`. That plus `numpy` is sufficient for everything in sections A–C and most of D. Tier 2 additionally needs live model loading (use `scripts/resubmit/` extraction code paths).

---

**Relevant files**

- `/projects/beto/irowerojas/localLatin/runs/active/ig_examples/phase12f_examples.csv` — 80-pair bucket index with per-pair `abtt_tau`, `D`, bucket label
- `/projects/beto/irowerojas/localLatin/runs/active/ig_examples/artifacts/<model_slug>/example*_pair_example.npz` — cached hiddens, PCs, IG, topk
- `/projects/beto/irowerojas/localLatin/scripts/resubmit/run_resubmit_ig_comparison.py` — reference for `clean_tokens` (per-token ABTT), `sparsity_ratio`, `content_focus`, `topk_precision` — reuse these
- `/projects/beto/irowerojas/localLatin/scripts/resubmit/evaluate_vectors.py` — reference for `compute_assignment_acc` and threshold sweep (Tier 2 counterfactual objective)
- `/projects/beto/irowerojas/localLatin/src/sif_abtt.py` — `EmbeddingCleaner` class (needed only if Tier 2 re-fits ABTT post-mask)
