# READY.md — pre-meeting brief, 2026-04-27

One-page elevator read for the 200-positive-pair attribution-metrics
bundle. Verified end-to-end on 2026-04-27 against
`runs/active/ig_examples_200pos/attribution_metrics/summary.csv`.

## What's in the bundle

Send these from `professor_share/`:

- `MEETING_PREP_2026_04_27.html` — primary walkthrough. Charts embedded
  as base64 PNGs (no broken paths). Single external dep is MathJax v3
  from cdn.jsdelivr.net; opens fine offline if MathJax has been cached
  before, otherwise needs internet on first open.
- `attribution_main_200pos_standalone.pdf` (+ `.tex`) — single-page
  results table, recompiles clean with `latexmk -pdf` (verified 15:10
  today, no LaTeX warnings).
- `chart_rho_loo_t5_200pos.{png,pdf}`, `chart_suff25_t5_200pos.{png,pdf}`
  — standalone charts for handouts.
- `QA_PREP_2026_04_27.md` — likely-question crib sheet.
- `make_charts_200pos.py` — receipt for how the charts were generated
  from `summary.csv`.

The earlier `attribution_main_200pair_standalone.*` and
`make_charts_200pair.py` are the older 200-random-pair bundle that had
the cosine-inflation bug. Don't send those to the professor — the
200pos versions supersede them.

## What fix landed

Commit `58b1b86` (merged via `f824025`, 2026-04-26 ~15:38 UTC, before
the 200pos run at ~19:15 UTC). PCs at
`runs/phase12_release/pcs/<slug>/` had been fit on token-keep-filtered
embeddings while `run_attribution_metrics.py` pools without that
filter; stale mean_vec / PC1 were near-orthogonal to the actual
distribution, so ABTT was *adding* anisotropy instead of removing it.
`scripts/ig/refit_pcs_for_attribution.py` refits mean_vec + D=10 PCs
through the same mean-pool path as `forward_pooled`. All
`FEATURED_MODELS` D-values are now pinned to 10 in
`scripts/ig/sample_{positive,random}_test_pairs.py`.

## Story to lead with

**ABTT improves leave-one-out ρ on positives, T5-only.** ρ_LOO
(baseline → ABTT):

- LaTa IG  −0.055 → +0.460  (+0.515)
- LaTa MaRC −0.124 → +0.455  (+0.580)
- PhilTa IG +0.058 → +0.245  (+0.188)
- PhilTa MaRC −0.080 → +0.295  (+0.375)

Range of improvement: +0.19 to +0.58 across the four cells, all four
positive. This is the headline result; lead with the ρ_LOO chart.

Compactness@0.80 is the second-tier finding and it's mixed. Use as
nuance, not headline:

- PhilTa IG improves    0.132 → 0.501  (good for ABTT)
- LaTa MaRC regresses  0.905 → 0.393  (bad for ABTT)
- LaTa IG essentially flat 0.359 / 0.363
- PhilTa MaRC modest gain 0.648 → 0.618 (slight regression actually)

Cosine sanity check: post-fix ABTT full_cos is **below** baseline as
expected (LaTa 0.564 < 0.934, PhilTa 0.574 < 0.954). Pre-fix this
inequality was reversed for LaTa.

## Still uncertain — flag as such

1. **LaTa MaRC compactness regresses sharply** (0.91 → 0.39). The
   "ABTT helps" story does not generalise to compactness on this
   model/method combo. No clean explanation yet.
2. **Cosine magnitudes are "less wrong", not vetted-correct.** The
   investigation predicted ABTT full_cos ≈ 0.30 on a mixed
   100-pos/100-neg sample. The 200pos run is positives-only and shows
   ≈ 0.56, which is plausibly higher because positives have inherently
   higher cosine, but no apples-to-apples comparison has been run to
   confirm. We know it's no longer broken; we don't know it's *right*.
3. **Only T5 (LaTa, PhilTa) in this run.** LaBSE / Qwen3-0.6B / KaLM
   were affected by the same stale-PC bug (esp. LaBSE, which had D=1)
   but were **not re-run** after the fix. Do not generalise to "all
   encoders" from these two.
4. **Effective ρ_LOO sample is ~83/200** in the LaTa rows
   (`loo_n_used_mean` ≈ 83). Many pairs got dropped from the LOO stat
   for whatever the dropping criterion is — flag the small effective
   N if the prof asks how robust the +0.5 rho gain is.

## Do NOT overclaim

- Don't say "ABTT solves the cosine-inflation problem." ABTT bringing
  cosines down to plausible is a correctness check that the **fix**
  works, not a finding about ABTT.
- Don't say "ABTT improves attribution quality" without qualification.
  It improves ρ_LOO across the board; compactness is mixed; sufficiency
  / comprehensiveness ratios in the table are not all monotone in the
  same direction.
- Don't generalise beyond LaTa + PhilTa. The bundle is T5-only.
- Don't claim the cosine-fix run is the final word — non-T5 models
  still need re-running before that claim is honest.

## Pre-meeting checklist

- [x] PDF rebuilt clean today (`latexmk -pdf attribution_main_200pos_standalone.tex`, no warnings, 1 page, 190585 bytes).
- [x] HTML opens locally; MathJax CDN URL is current; 9/9 internal anchors resolve.
- [x] Every numeric claim in HTML, PDF, and Q&A traces to a row in `summary.csv` (verified at 3 d.p.).
- [x] Cosine fix is on main as `58b1b86` and PC files at
      `runs/phase12_release/pcs/{bowphs_LaTa/layer4_pcs.npz,bowphs_PhilTa/layer6_pcs.npz}` are shape `(10, 768)`.
- [ ] Confirm laptop has internet (or pre-load the HTML once) so MathJax renders if you screen-share.
