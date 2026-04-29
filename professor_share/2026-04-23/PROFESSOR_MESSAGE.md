# Draft message to professor

Subject: **MaRC + IG attribution numbers — mixed, with one robust signal**

---

Hi Professor,

Per your ask at our walkthrough, I finished computing the quantitative attribution-quality metrics for MaRC and IG with and without ABTT on the curated 20-pair-per-model set (LaTa, PhilTa, LaBSE, Qwen3-0.6B). Attaching:

1. **`attribution_main_standalone.pdf`** — the main-text candidate table: LaTa + PhilTa × {IG, MaRC} × {baseline, ABTT} × {Suff@25%, Comp@25%, Cmpct@0.8, ρ_LOO}. Better-of-baseline-vs-ABTT is bolded in each cell.
2. **`chart_rho_loo_t5.png`** — ρ_LOO (rank correlation between attribution importance and leave-one-out impact on the cosine). ABTT beats baseline on all 4 cells.
3. **`chart_suff25_t5.png`** — Sufficiency@25% on the same 4 cells. Baseline beats ABTT on 3 of 4.
4. **`chart_rho_loo_all.png`** — ρ_LOO across LaTa, PhilTa, LaBSE. ABTT improves on every one of the 6 cells where the metric is defined.

**The short version: the clean "ABTT uniformly improves attribution quality" claim does not hold on this subset.** Only PhilTa × MaRC clears the ≥3/4-metrics bar. For IG on both LaTa and PhilTa, three of the four faithfulness metrics actually go the wrong direction under ABTT. ρ_LOO is the one metric that behaves as hoped: it improves on **every** (model, method) cell where it is defined (6/6 across LaTa, PhilTa, LaBSE).

I think this gives us three honest framings for the paper, in order of decreasing confidence:

1. **Narrowed claim**: "ABTT improves ρ_LOO across every model and attribution method we evaluated, while other faithfulness metrics show encoder- and method-dependent effects." The main table supports this read; the appendix table shows the full mixed picture.
2. **Descriptive**: report the numbers without a directional ABTT claim and let the reader infer. The current main-table caption is already neutral.
3. **Reposition attribution as diagnostic**: drop "ABTT improves explanation" from the contribution list and keep the attribution section as a diagnostic showing where existing methods succeed vs fail on ABTT-processed encoders.

A cleaner experiment to actually support a stronger claim would be to run these metrics on a random sample from the test set (say 200 pairs/model) rather than the curated phase12f visualization set — with only 20 pairs per cell the numbers are descriptive, not statistically powered. Happy to do that if you think it's worth the compute, but it's not strictly necessary for the three framings above.

Let me know which framing you want to go with and I'll lock it in for the main-text attribution section. I'm holding off on the final paper rewrite (orchestra Run 3) until we decide.

Best,
Ian

---

## Quick numbers (for your reference, also in the attached table)

| cell                 | Suff@25% | Comp@25% | Cmpct | ρ_LOO | ABTT wins |
|----------------------|----------|----------|-------|-------|-----------|
| LaTa × IG            | base     | base     | base  | ABTT  | 1/4       |
| LaTa × MaRC          | base     | base     | base  | ABTT  | 1/4       |
| PhilTa × IG          | base     | base     | base  | ABTT  | 1/4       |
| PhilTa × MaRC        | **ABTT** | **ABTT** | base  | ABTT  | **3/4**   |
| LaBSE × IG (appendix)   | base     | ABTT     | base  | ABTT  | 2/4       |
| LaBSE × MaRC (appendix) | base     | ABTT     | base  | ABTT  | 2/4       |
| Qwen3-0.6B × IG (appendix) | ABTT | ABTT  | base  | ABTT  | 3/4       |

ρ_LOO column: ABTT wins 7/7 (decoder cell on Qwen3-0.6B × MaRC excluded — metric undefined there).

---

## Update (200 random test pairs/model)

Reran the attribution metrics on **200 random pairs per model** (100 positive + 100 negative, winnable test set) for LaTa and PhilTa, per our discussion. Standard errors shrank by roughly $\sqrt{10} \approx 3.2\times$ relative to the 20-curated baseline, so deltas of ~0.05 are now meaningful.

New artifacts in this directory:

- `attribution_main_200pair_standalone.pdf` — one-page table, same 2-model x 2-method x 4-metric shape, cross-variant bolding
- `chart_rho_loo_t5_200pair.{pdf,png}` — $\rho_{\text{LOO}}$ on T5 cells
- `chart_suff25_t5_200pair.{pdf,png}` — Suff@25% on T5 cells

**Headline vs 20-curated:** the picture got *softer*, not sharper.

| Cell | 20 pairs | 200 pairs |
|------|----------|-----------|
| LaTa x IG     | 1/4 | **1/4** |
| LaTa x MaRC   | 1/4 | **2/4** (comp flipped ABTT) |
| PhilTa x IG   | 1/4 | **1/4** |
| PhilTa x MaRC | 3/4 | **2/4** ($\rho_{\text{LOO}}$ flipped baseline) |

$\rho_{\text{LOO}}$ specifically: at 20 pairs it improved universally (4/4 T5 cells, 8/8 across all 4 models). At 200 pairs it now improves on **3/4** T5 cells — PhilTa x MaRC regresses from $-0.036$ to $-0.120$, a drop of $\sim 4\sigma$ that is not a sampling artifact.

**So the ``universal $\rho_{\text{LOO}}$ lift'' framing I proposed last round no longer survives the sample.** The narrowest defensible signal is now:
- **For IG specifically, ABTT lifts $\rho_{\text{LOO}}$ on both T5 models** (LaTa: $-0.014 \to +0.106$; PhilTa: $+0.042 \to +0.120$). That is a 2-cell story, not 4-cell, and it is a claim about IG rather than attribution in general.

Given this, my updated recommendation is **option 3 from the last message: reposition the attribution section as a diagnostic exhibit** rather than a contribution. The 200-pair numbers don't give us evidence that ABTT uniformly improves attribution quality; they give us a fine-grained mixed picture of where existing attribution methods succeed or fail on ABTT-processed encoders. That's still a useful thing to show, but the paper shouldn't claim a directional improvement.

If you still want to argue ABTT improves *something* attribution-related, the most defensible version is: "on IG specifically, ABTT tokens rank better under leave-one-out probing on both Latin T5 models." That's small but it holds.

Let me know how you'd like to frame it in the paper and I'll lock it in for Run 3 (narrative rewrite).
