# Run 4 Agent 2: Appendix Attribution Sweep Package

Generated: 2026-05-01

## Outputs

- `overleaf_drafts/tables/attribution_metrics_sweep_main_methods.tex`
  separates the two paper-facing attribution views, IG and retrieval-adapted
  MaRC, across LaTa, PhilTa, and mT5-base.
- `overleaf_drafts/tables/attribution_metrics_sweep_supplemental_methods.tex`
  keeps additional methods and diagnostic baselines out of the main-method
  appendix table.
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary_sweep_long.csv`
  is the machine-readable long sweep grid.
- `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/appendix_sweep_completeness.json`
  records source-data completeness.

## Completeness Check

The current Run 3 source summary contains all three main-paper models and all
model/method/variant cells, and it contains Sufficiency and Comprehensiveness
at 10%, 25%, and 50%. It only contains the legacy MinFrac/Compactness threshold
at 0.80. Exact MinFrac@0.70, MinFrac@0.90, and MinFrac@0.95 are therefore marked
as missing in the generated appendix tables and as `metric_n=0` in the long CSV.

The metric computation script now supports the full compactness grid, but the
checked-in Run 3 `summary.csv` was generated before that sweep was rerun. To
make the appendix numerically complete, rerun the attribution metric stage from
the cached NPZ artifacts with:

```bash
python scripts/ig/run_attribution_metrics.py \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --out_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --tex_out runs/active/ig_examples_200pos_run3_operational/attribution_metrics.tex \
  --sweep_tex_out overleaf_drafts/tables/attribution_metrics_200pos_sweep_appendix.tex \
  --compactness_thresholds 0.70,0.80,0.90,0.95 \
  --trust_remote_code
```

The cached NPZ artifact directories are not present in this checkout, so this
branch packages the appendix structure and makes the missing threshold data
auditable rather than fabricating values.

## Generation Commands

```bash
python -m py_compile scripts/ig/package_attribution_sweep_appendix.py
python scripts/ig/package_attribution_sweep_appendix.py
python scripts/ig/package_attribution_sweep_appendix.py --strict
```

The strict check currently fails by design until the exact MinFrac@0.70,
MinFrac@0.90, and MinFrac@0.95 columns exist in the source summary.
