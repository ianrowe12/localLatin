# Run 5 Agent 2 Reproducibility Audit

Updated: 2026-05-01

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
| Attribution compactness sweep | Intentional external-artifact limitation | The metric code supports `compactness@0.70`, `@0.80`, `@0.90`, and `@0.95`, but this checkout lacks the active Run 3 NPZ artifacts and per-pair metric JSONs needed to recompute thresholds other than 0.80. The checked-in `summary.csv` is aggregate-only; the missing thresholds cannot be reconstructed from it without fabricating values. |
| Cached attribution artifact receipt | Intentional external-artifact limitation | `build_attribution_run_manifest.py --require_complete` fails in this checkout because `runs/active/ig_examples_200pos_run3_operational/{pcs,artifacts,retrieval_mark/artifacts}` are ignored heavy generated artifacts and are not materialized. The checked-in `manifest.json`/`artifact_inventory.csv` record the completed GPU run receipt, but a fresh checkout must rerun or restore those generated artifacts before `--require_complete` can pass. |
| Task B cumulative top-K table traceability | Pass | The inline top-K table in `overleaf_drafts/acl_latex.tex` traces to `overleaf_drafts/figures/taskb_mseed_selected_configs.csv`; values match after multiplying `dir_acc_at_K_mean` by 100 and rounding. |
| Task B cumulative top-K method label | Pass | The table values come from SIF+ABTT multi-seed configs, not pure ABTT. The caption labels it as the variance-aware SIF+ABTT table and points pure-ABTT ranking to the single-seed per-layer tables. |
| Per-layer Task A / Task B table source traceability | Pass | Restored the ignored source CSVs `runs/active/resubmit/results/phase_resubmit_results.csv` and `runs/active/resubmit/taskb_mseed/aggregated_results.csv`, plus regenerated audit CSVs under `runs/active/resubmit/results/perlayer_tables/`. `python scripts/resubmit/build_per_layer_tables.py` rebuilds the tracked TeX tables without changing headline numbers. |
| Cluster-geometry headline range | Pass | Restored trace CSVs `runs/active/resubmit/cluster_viz/cluster_silhouette_main.csv`, `cluster_silhouette_appendix.csv`, and `cluster_silhouette.csv`. The main-caption range 0.09--0.75 traces to main-model baseline-to-ABTT silhouette deltas: minimum PhilTa t-SNE delta 0.0875, maximum mT5-base t-SNE delta 0.7470. |
| mT5 naming and inclusion | Pass | Displayed labels use `mT5-base`; mT5-base remains included in the main attribution and retrieval tables. |
| Superseded attribution artifacts in current paper | Pass | `overleaf_drafts/acl_latex.tex` does not contain `200-random`, `200pair`, `two-model`, `LaTa L4`, or `PhilTa L6`. Historical mentions remain only in run notes where explicitly labeled superseded or diagnostic. |

## Attribution Artifact Reproduction

The active Run 3 compactness sweep can be completed only after restoring or
regenerating the active generated artifacts. Do not substitute the older
`runs/active/ig_examples_200pos/` bundle: it is two-model and uses superseded
LaTa/PhilTa attribution layers.

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

If the NPZ/sidecar artifacts already exist, rerun only the metric and reporting
stages:

```bash
python scripts/ig/run_attribution_metrics.py \
  --examples_csv runs/active/ig_examples_200pos_run3_operational/positive200_examples.csv \
  --artifacts_root runs/active/ig_examples_200pos_run3_operational/artifacts \
  --out_root runs/active/ig_examples_200pos_run3_operational/attribution_metrics \
  --tex_out runs/active/ig_examples_200pos_run3_operational/attribution_metrics.tex \
  --sweep_tex_out overleaf_drafts/tables/attribution_metrics_200pos_sweep_appendix.tex \
  --compactness_thresholds 0.70,0.80,0.90,0.95 \
  --trust_remote_code \
  --require_artifacts

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
  scripts/ig/build_main_attribution_artifacts.py \
  scripts/ig/package_attribution_sweep_appendix.py \
  scripts/resubmit/build_per_layer_tables.py \
  scripts/resubmit/visualize_taskb_mseed.py \
  scripts/resubmit/visualize_resubmit.py

python scripts/resubmit/build_per_layer_tables.py
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
```

Observed expected failures in this checkout:

- `package_attribution_sweep_appendix.py --strict` fails until active Run 3
  per-pair metrics are recomputed from materialized NPZ artifacts.
- `build_attribution_run_manifest.py --require_complete` fails until active
  Run 3 generated PC, NPZ, MaRC sidecar, and per-pair metric JSON artifacts are
  restored or regenerated.

## Remaining Risks For Rewrite

- Do not present `compactness@0.70`, `compactness@0.90`, or
  `compactness@0.95` as measured for the active three-model Run 3 bundle until
  the GPU metric stage above is rerun.
- Do not treat a clean source checkout as containing the active Run 3 heavy
  attribution artifacts; it contains checked-in receipts plus documented
  reproduction commands.
- Do not use older 20-pair, 200-random, 200-positive two-model, LaTa L4, or
  PhilTa L6 artifacts as current headline evidence.
