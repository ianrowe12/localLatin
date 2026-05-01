# Run 4 Main Attribution Artifacts

Generated: 2026-04-30

## Source

Main-text artifacts read the Run 3 operational attribution summary:

```text
runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv
```

The table uses the predeclared operational attribution layers from
`docs/runs/run2_attribution_layer_contract.md`: LaTa L7, PhilTa L1, and
mT5-base L1.

## Artifacts

- `overleaf_drafts/tables/attribution_metrics_main.tex`
- `overleaf_drafts/figures/fig_attribution_rho_loo_main.tex`
- `overleaf_drafts/figures/fig_attribution_rho_loo_main.pdf`
- `overleaf_drafts/figures/fig_attribution_rho_loo_main.png`

The table is scoped to the paper's two candidate-attribution methods, IG and
retrieval-adapted MaRC, across LaTa, PhilTa, and mT5-base. It foregrounds
`rho_LOO` and reports ERASER-style metrics as baseline-to-ABTT values so
disagreements remain visible.

## Threshold Choice

The main table keeps the conventional global thresholds: Sufficiency and
Comprehensiveness at 25 percent of query tokens, and MinFrac at recovery
threshold 0.80. These are the midpoint headline settings in the Run 3 sweep
grid and are used globally across all models and methods. No model-specific or
method-specific threshold was selected.

This choice preserves continuity with the prior attribution artifacts while
the appendix carries the broader 10/25/50 percent and 0.70/0.80/0.90/0.95
sweep. The main table is therefore not optimized for any individual cell:
`rho_LOO` improves in all 6/6 model-method cells, while the ERASER-style
metrics improve under ABTT in only 6/18 comparisons.

## Rebuild

```bash
python scripts/ig/build_main_attribution_artifacts.py
python -m py_compile scripts/ig/build_main_attribution_artifacts.py
python -m src.attribution_metrics
```

Focused LaTeX check used during generation:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error /tmp/run4_main_attribution_check.tex
```
