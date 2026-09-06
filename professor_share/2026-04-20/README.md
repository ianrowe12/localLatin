# MaRC + IG attribution metrics — shareable bundle

Generated 2026-04-20, post-orchestra Run 1 Agent 1.4.

## Files to send the professor

1. `attribution_main_standalone.pdf` — one-page compiled view of the trimmed main table: LaTa + PhilTa × {IG, MaRC} × {baseline, ABTT} × 4 faithfulness metrics, with a reading guide.
2. `chart_rho_loo_t5.pdf` / `.png` — grouped bar chart of ρ_LOO on the 2 T5 models (main-text scope). Robust signal: ABTT > baseline on all 4 cells.
3. `chart_suff25_t5.pdf` / `.png` — grouped bar chart of Sufficiency@25% on the same 2 T5 models. Mixed signal: baseline > ABTT on 3 of 4 cells.
4. `chart_rho_loo_all.pdf` / `.png` — wider view across LaTa, PhilTa, LaBSE × {IG, MaRC}. ρ_LOO improves on every one of the 6 cells where the metric is defined. (Qwen3-0.6B × MaRC is excluded because the decoder-only fallback makes ρ_LOO undefined for that cell.)
5. `PROFESSOR_MESSAGE.md` — drafted email/Slack body. Edit before sending.

Suggested attach order for email: PDF table first (headline), then the two T5 charts (ρ_LOO success + Suff@25% caveat), then the wider ρ_LOO chart as supporting evidence.

## What the numbers say (short)

- **Original sanity-check threshold**: "ABTT improves ≥3/4 metrics for both MaRC and IG on at least one T5 model." **Not met.** Only PhilTa × MaRC clears 3/4.
- **One robust signal**: ρ_LOO (rank correlation between attribution importance and true LOO impact) improves under ABTT on every (model, method) cell where the metric is defined (6/6).
- **Other three metrics** (sufficiency, comprehensiveness, compactness) go the other way for IG on both T5 models; MaRC on PhilTa improves 3/4; MaRC on LaTa 1/4.

## Sources

- Summary CSV: `runs/active/ig_examples/attribution_metrics/summary.csv`
- Full paper-ready tables:
  - `overleaf_drafts/tables/attribution_metrics.tex` (full auto-generated)
  - `overleaf_drafts/tables/attribution_metrics_main.tex` (LaTa+PhilTa × IG+MaRC, main text)
  - `overleaf_drafts/tables/attribution_metrics_appendix.tex` (4 models × 9 methods)
- Honest findings write-up: `docs/analyses/FINDINGS_attribution.md`
- Chart script: `professor_share/make_charts.py`
- Standalone table source: `professor_share/attribution_main_standalone.tex`

## Paths are symlink-safe

Everything here lives under `/projects/beto/irowerojas/localLatin/professor_share/` (also reachable via `/u/irowerojas/localLatin/professor_share/`).
