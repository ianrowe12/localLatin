# Run 5 Agent 2 Reproducibility Audit

Updated: 2026-05-02 01:33 CDT

## Scope

Audited the paper-facing outputs before the rewrite for source traceability,
Run 2 attribution-layer consistency, metric directionality, mT5 coverage, and
avoidance of superseded 200-random or two-model-only attribution artifacts.

## Pass / External-Limitation Items

| Item | Status | Evidence |
|---|---|---|
| Run 2 operational attribution layers | Pass | `scripts/ig/attribution_model_config.py`, `runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv`, and `runs/active/ig_examples_200pos_run3_operational/artifact_inventory.csv` agree on LaTa L7, PhilTa L1, mT5-base L1, D=10. |
| Main attribution table traceability | Pass | `overleaf_drafts/tables/attribution_metrics_main.tex` rebuilds from `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv` via `scripts/ig/build_main_attribution_artifacts.py`. |
| Main attribution model coverage | Pass | Source summary has 54 rows and includes LaTa, PhilTa, and mT5-base; the main IG/MaRC slice has all 12 model-method-variant rows with n=200 per row. |
| Attribution metric directionality | Pass | Caption states rho_LOO, Suff@25, and Comp@25 are higher-is-better, while MinFrac@0.80 is lower-is-better; bolding follows those directions. Verified rho_LOO improves in all 6/6 model-method cells. |
| Attribution compactness sweep | Pass | Regenerated the active Run 3 bundle and rebuilt `runs/active/ig_examples_200pos_run3_operational/attribution_metrics/summary.csv`, `summary_sweep_long.csv`, and `appendix_sweep_completeness.json`. The summary now includes `compactness@0.70`, `@0.80`, `@0.90`, and `@0.95`; the sweep has 594 rows and `complete: true`. |
| Cached attribution artifact receipt | Pass | Slurm job `18001329` materialized the active Run 3 generated artifacts locally: 600 canonical attribution NPZs, 600 MaRC sidecars, 600 per-pair metric JSONs, plus the three PC files. `build_attribution_run_manifest.py --require_complete` now passes against those generated paths. |
| Task B cumulative top-K table traceability | Pass | The inline top-K table in `overleaf_drafts/acl_latex.tex` traces to `overleaf_drafts/figures/taskb_mseed_selected_configs.csv`; values match after multiplying `dir_acc_at_K_mean` by 100 and rounding. |
| Task B cumulative top-K method label | Pass | The table values come from SIF+ABTT multi-seed configs, not pure ABTT. The caption labels it as the variance-aware SIF+ABTT table and points pure-ABTT ranking to the single-seed per-layer tables. |
| Per-layer Task A / Task B table source traceability | Pass | Restored the ignored source CSVs `runs/active/resubmit/results/phase_resubmit_results.csv` and `runs/active/resubmit/taskb_mseed/aggregated_results.csv`, plus regenerated audit CSVs under `runs/active/resubmit/results/perlayer_tables/`. `python scripts/resubmit/build_per_layer_tables.py` rebuilds the tracked TeX tables without changing headline numbers. |
| Cluster-geometry headline range | Pass | Restored trace CSVs `runs/active/resubmit/cluster_viz/cluster_silhouette_main.csv`, `cluster_silhouette_appendix.csv`, and `cluster_silhouette.csv`. The main-caption range 0.09--0.75 traces to main-model baseline-to-ABTT silhouette deltas: minimum PhilTa t-SNE delta 0.0875, maximum mT5-base t-SNE delta 0.7470. |
| mT5 naming and inclusion | Pass | Displayed labels use `mT5-base`; mT5-base remains included in the main attribution and retrieval tables. |
| Superseded attribution artifacts in current paper | Pass | `overleaf_drafts/acl_latex.tex` does not contain `200-random`, `200pair`, `two-model`, `LaTa L4`, or `PhilTa L6`. Historical mentions remain only in run notes where explicitly labeled superseded or diagnostic. |

## Attribution Artifact Reproduction

The active Run 3 compactness sweep was completed by regenerating the active
generated artifacts. No full active Run 3 generated artifacts were found in this
checkout, nearby `/projects/beto/irowerojas/` worktrees, or
`/projects/beto/irowerojas/localLatin_archive/` before regeneration; only
checked-in aggregate receipts and smoke-run artifacts were materialized locally.
The full GPU regeneration was submitted from this checkout:

```bash
REPO_ROOT=/projects/beto/irowerojas/localLatin \
sbatch slurm/ig/run_attribution_200pos_run3_operational.sbatch
```

Slurm job: `18001329` (`attr_200pos_r3`). It ran on `gpua067` from
2026-05-01 21:03:30 CDT to 2026-05-02 01:20:53 CDT and completed successfully
with exit code `0:0`.

Materialized generated artifacts:

```text
600 canonical attribution NPZs
600 MaRC sidecar NPZs
600 per-pair metric JSONs
3 PC files: LaTa L7, PhilTa L1, mT5-base L1
```

After the GPU job completed, the metric/reporting outputs were rebuilt from the
cached per-pair JSONs with `--require_artifacts --render_only`, using the full
compactness threshold set `0.70,0.80,0.90,0.95`. This avoids repeating the GPU
forward passes while still requiring the materialized artifacts and cached
per-pair metrics.

Expected materialized generated paths:

```text
runs/active/ig_examples_200pos_run3_operational/pcs/
runs/active/ig_examples_200pos_run3_operational/artifacts/
runs/active/ig_examples_200pos_run3_operational/retrieval_mark/artifacts/
runs/active/ig_examples_200pos_run3_operational/attribution_metrics/*/*.json
```

GPU reproduction entry point:

```bash
REPO_ROOT=/projects/beto/irowerojas/localLatin \
sbatch slurm/ig/run_attribution_200pos_run3_operational.sbatch
```

Metric and reporting rebuild command:

```bash
python scripts/ig/run_attribution_metrics.py \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --out_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --tex_out runs/active/ig_examples_200pos_run3_operational/attribution_metrics.tex \
  --sweep_tex_out overleaf_drafts/tables/attribution_metrics_200pos_sweep_appendix.tex \
  --compactness_thresholds 0.70,0.80,0.90,0.95 \
  --trust_remote_code \
  --require_artifacts \
  --render_only

python scripts/ig/package_attribution_sweep_appendix.py --strict
python scripts/ig/build_main_attribution_artifacts.py
```

Manifest validation after artifacts are materialized:

```bash
python scripts/ig/build_attribution_run_manifest.py \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --pc_root runs/active/ig_examples_200pos_run3_operational/pcs \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --retrieval_mark_root runs/active/ig_examples_200pos_run3_operational/retrieval_mark/artifacts \
  --metrics_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --out_json /tmp/run5_manifest_check.json \
  --out_inventory_csv /tmp/run5_inventory_check.csv \
  --require_complete
```

## Verification Commands

```bash
python -m py_compile \
  scripts/ig/run_attribution_metrics.py \
  scripts/ig/build_main_attribution_artifacts.py \
  scripts/ig/package_attribution_sweep_appendix.py \
  scripts/ig/build_attribution_run_manifest.py

python scripts/ig/build_main_attribution_artifacts.py
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
```

Observed result: all verification commands above pass after materializing the
active Run 3 generated artifacts.

## Remaining Risks For Rewrite

- The active Run 3 heavy attribution artifacts are intentionally tracked as of
  commit `e57b714`; a clean checkout of `main` should contain the PC, NPZ, MaRC
  sidecar, and per-pair metric JSON artifacts needed for
  `build_attribution_run_manifest.py --require_complete`.
- Do not use older 20-pair, 200-random, 200-positive two-model, LaTa L4, or
  PhilTa L6 artifacts as current headline evidence.
- Keep rho_LOO as the primary faithfulness result; ERASER-style sufficiency,
  comprehensiveness, and MinFrac compactness metrics remain complementary.
