# Run 5 Agent 2 Reproducibility Audit

Generated: 2026-05-01

## Scope

Audited the paper-facing outputs before the rewrite for source traceability,
Run 2 attribution-layer consistency, metric directionality, mT5 coverage, and
avoidance of superseded 200-random or two-model-only attribution artifacts.

## Pass / Fail Items

| Item | Status | Evidence |
|---|---|---|
| Run 2 operational attribution layers | Pass | `scripts/ig/attribution_model_config.py` and `runs/active/ig_examples_200pos_run3_operational/artifact_inventory.csv` agree on LaTa L7, PhilTa L1, mT5-base L1, D=10. |
| Main attribution table traceability | Pass | `overleaf_drafts/tables/attribution_metrics_main.tex` rebuilds from `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv` via `scripts/ig/build_main_attribution_artifacts.py`. |
| Main attribution model coverage | Pass | Source summary has 54 rows and includes LaTa, PhilTa, and mT5-base; the main IG/MaRC slice has all 12 model-method-variant rows with n=200 per row. |
| Attribution metric directionality | Pass | Caption states rho_LOO, Suff@25, and Comp@25 are higher-is-better, while MinFrac@0.80 is lower-is-better; bolding follows those directions. Verified rho_LOO improves in all 6/6 model-method cells. |
| Attribution appendix completeness | Fail, documented | `python scripts/ig/package_attribution_sweep_appendix.py --strict` fails because `compactness@0.70_mean`, `compactness@0.90_mean`, and `compactness@0.95_mean` are absent from the checked-in Run 3 summary. The non-strict appendix marks these as missing. |
| Cached attribution artifact receipt | Fail in this checkout | `build_attribution_run_manifest.py --require_complete` fails because the cached PC/NPZ/per-pair JSON artifacts are not materialized in this worktree. The checked-in manifest records them as complete, but the actual files are absent, likely because `git-lfs` was unavailable during worktree checkout. |
| Task B cumulative top-K table traceability | Pass after label fix | The inline top-K table in `overleaf_drafts/acl_latex.tex` traces to `overleaf_drafts/figures/taskb_mseed_selected_configs.csv`; the values match after multiplying `dir_acc_at_K_mean` by 100 and rounding. |
| Task B cumulative top-K method label | Fixed | The table values come from SIF+ABTT multi-seed configs, not pure ABTT. Updated the caption to say this is the variance-aware SIF+ABTT table and points pure-ABTT ranking to the single-seed per-layer tables. |
| Per-layer Task A / Task B table source traceability | Fail | `scripts/resubmit/build_per_layer_tables.py` still defaults to `runs/active/resubmit/results/phase_resubmit_results.csv` and `runs/active/resubmit/taskb_mseed/aggregated_results.csv`, but those source CSVs are not present. The generated per-layer TeX tables are checked in, but their source/audit CSVs are not available in this checkout. |
| Cluster-geometry headline range | Fail | Figure captions report silhouette gains of 0.09--0.75, but no checked-in cluster metric CSV/JSON was found to trace that range. |
| mT5 naming and inclusion | Fixed / Pass | Standardized displayed labels from `mt5-base` to `mT5-base` in the paper text and generated Task B table snippets; mT5-base remains included in the main attribution and retrieval tables. |
| Superseded attribution artifacts in current paper | Pass | `overleaf_drafts/acl_latex.tex` does not contain `200-random`, `200pair`, `two-model`, `LaTa L4`, or `PhilTa L6`. Historical mentions remain only in run notes where explicitly labeled superseded or diagnostic. |

## Verification Commands

```bash
python -m py_compile \
  scripts/ig/build_main_attribution_artifacts.py \
  scripts/ig/package_attribution_sweep_appendix.py \
  scripts/resubmit/build_per_layer_tables.py \
  scripts/resubmit/visualize_taskb_mseed.py \
  scripts/resubmit/visualize_resubmit.py

python scripts/ig/build_main_attribution_artifacts.py

python scripts/ig/package_attribution_sweep_appendix.py

python scripts/ig/package_attribution_sweep_appendix.py --strict

python scripts/ig/build_attribution_run_manifest.py \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --pc_root runs/active/ig_examples_200pos_run3_operational/pcs \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --retrieval_mark_root runs/active/ig_examples_200pos_run3_operational/retrieval_mark/artifacts \
  --metrics_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --out_json /tmp/run5_manifest_check.json \
  --out_inventory_csv /tmp/run5_inventory_check.csv \
  --require_complete

python scripts/resubmit/build_per_layer_tables.py
```

Expected failures:

- `package_attribution_sweep_appendix.py --strict` fails until the missing
  compactness thresholds are recomputed.
- `build_attribution_run_manifest.py --require_complete` fails in this checkout
  because cached artifact files are absent.
- `build_per_layer_tables.py` fails because the source CSV paths named by the
  builder are absent.

## Remaining Risks For Rewrite

- Do not treat the per-layer Task A/Task B tables as fully reproducible until
  their exact source CSVs or audit CSVs are restored and checked against the TeX.
- Do not claim the Task B multi-seed top-K table is pure ABTT; it is SIF+ABTT
  according to its selected-config CSV.
- Do not present compactness thresholds 0.70, 0.90, or 0.95 as measured; they
  are placeholders marked missing until metrics are rerun.
- Recheck cluster-geometry captions or regenerate a source CSV for the
  silhouette range before keeping the 0.09--0.75 headline.
