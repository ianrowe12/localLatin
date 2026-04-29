# Layer Geometry Diagnostics and Layer Selection

Generated: 2026-04-29

## Purpose

This note answers the layer-selection question behind the attribution work: why inspect intermediate layers, and can readers choose a layer without already knowing the final test metric?

The accompanying machine-readable evidence is in `runs/active/resubmit/layer_diagnostics/`:

| File | Contents |
|---|---|
| `geometry_per_layer.csv` | Label-free geometry diagnostics for train/test, raw and ABTT-D10 views. |
| `geometry_retrieval_join_main.csv` | Main-paper model join: LaTa, PhilTa, mT5-base. |
| `geometry_retrieval_join_appendix.csv` | Cached appendix model join: LaBSE, Qwen3-0.6B, KaLM-mini. |
| `geometry_correlation_summary.csv` | Pearson/Spearman associations between diagnostics and retrieval outcomes. |
| `layer_rule_candidates.csv` | Candidate operational and diagnostic layers under a predeclared rule. |
| `manifest.json`, `evidence_manifest.json` | Input paths, script provenance, and output inventory. |

All geometry is computed from cached `hidden_mean_tokempty` embeddings in `runs/active/resubmit_bases/phase9_bases/` and aligned to `runs/active/resubmit/data/phase_resubmit_split.csv`. ABTT PCs are fit on train rows only, then applied to train/test rows before diagnostics are reported.

## What The Diagnostics Show

Intermediate layers are worth inspecting because they reveal representation collapse that final-layer or best-layer tables hide. For the main-paper models, the strongest collapse under top-PC dominance and effective rank occurs at intermediate layers:

| Model | Operational train-selected layer | Max-PC1 diagnostic layer | PC1 share at diagnostic layer | Effective rank there | After ABTT-D10: PC1 share | After ABTT-D10: effective rank |
|---|---:|---:|---:|---:|---:|---:|
| LaTa | 7 | 8 | 0.956 | 1.34 | 0.036 | 168.40 |
| PhilTa | 1 | 6 | 0.862 | 1.83 | 0.039 | 155.02 |
| mT5-base | 1 | 5 | 1.000 | 1.00 | 0.044 | 151.74 |

This is the defensible reason to inspect intermediate layers: the geometry makes the middle-layer failure mode visible. ABTT-D10 largely reverses the collapse in those layers, reducing PC1 dominance to roughly 0.04 and lifting effective rank above 150 for the three main models.

The diagnostics are also useful for explaining where ABTT helps most. In the main models, PC1 dominance and low effective rank are strongly associated with ABTT gains over baseline. For example, across the 36 main-model layers, `pc1_variance_ratio` has Pearson 0.869 with ABTT AUROC gain and 0.728 with ABTT DirAcc@1 gain. Conversely, `effective_rank_entropy` has Pearson -0.882 with ABTT AUROC gain and -0.818 with ABTT DirAcc@1 gain.

## What They Do Not Show

The intrinsic diagnostics do not cleanly select the best retrieval layer. They identify collapse/recovery regimes, not the operational optimum.

For example, LaTa's strongest PC1 collapse is layer 8, but the train-only retrieval rule selects layer 7 and the held-out single-seed DirAcc@1 fallback would select layer 1. PhilTa and mT5-base also have diagnostic collapse layers in the middle, while the train-selected operational layer is layer 1. This mismatch is not a failure of the diagnostics; it means the claim should not be "unsupervised diagnostics choose the best retrieval layer."

The right claim is narrower:

> Intrinsic layer geometry diagnoses collapse and explains why ABTT can recover retrieval signal, but it is not a reliable standalone layer-selection oracle on this corpus.

## Recommended Layer Rule

Use a two-stage rule.

1. Use intrinsic geometry as a label-free diagnostic pass. Report top-PC dominance, effective rank, cosine concentration/spread, and ABTT-D10 geometry changes to show whether a layer is collapsed and whether ABTT repairs the space.
2. Choose operational retrieval/attribution layers by a predeclared train-only retrieval criterion. The rule used in `layer_rule_candidates.csv` is: choose the earliest layer within 0.5 percentage points of the best `train_dir_acc_at_1__abtt_optimal`.

Under that rule, the recommended operational layers are:

| Model | Operational layer | Diagnostic collapse layer by max PC1 | Held-out DirAcc@1 fallback layer |
|---|---:|---:|---:|
| LaTa | 7 | 8 | 1 |
| PhilTa | 1 | 6 | 1 |
| mT5-base | 1 | 5 | 1 |

This is the most defensible recommendation for attribution: explain the retrieval-selected operational layer, and treat recovered-collapse layers as mechanism checks in the appendix. If a later paper decision prefers continuity with current held-out main-table selections, use the fallback layer column explicitly and label it as held-out selected, not intrinsic selected.

## Suggested Paper Wording

Short version:

> We inspect intermediate layers because the failure mode is geometric rather than monotonic with depth: several middle layers collapse into a near one-dimensional cone, while ABTT-D10 restores rank and cosine spread. These diagnostics are label-free and explain where ABTT should help, but they do not by themselves identify the best retrieval layer. We therefore use them as diagnostic evidence and select operational attribution layers with a predeclared train-only retrieval rule.

Fallback wording if reviewers ask for a selector:

> Intrinsic geometry diagnostics help identify representation collapse, but they do not reliably select the best retrieval layer on this corpus. We treat them as descriptive checks, not as a layer-selection oracle. For operational retrieval and attribution, we choose the layer by a predeclared training-only retrieval criterion, lock that choice before attribution metrics are computed, and report the full layer sweep so readers can see when the selected layer is part of a broad performance plateau.

## Reproduction

Smoke run:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/resubmit/run_layer_geometry_diagnostics.py \
  --models bowphs/LaTa,bowphs/PhilTa,google/mt5-base \
  --layers 1,6,12 \
  --split_csv /projects/beto/irowerojas/localLatin/runs/active/resubmit/data/phase_resubmit_split.csv \
  --runs_root /projects/beto/irowerojas/localLatin/runs/active/resubmit_bases \
  --out_dir /tmp/locallatin_layer_diagnostics_smoke
```

Full run:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/resubmit/run_layer_geometry_diagnostics.py \
  --models all \
  --split_csv /projects/beto/irowerojas/localLatin/runs/active/resubmit/data/phase_resubmit_split.csv \
  --runs_root /projects/beto/irowerojas/localLatin/runs/active/resubmit_bases \
  --out_dir runs/active/resubmit/layer_diagnostics
```

Join retrieval outcomes:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python scripts/resubmit/build_layer_geometry_evidence.py \
  --diagnostics_csv runs/active/resubmit/layer_diagnostics/geometry_per_layer.csv \
  --retrieval_csv /projects/beto/irowerojas/localLatin/runs/active/resubmit/results/phase_resubmit_results.csv \
  --taskb_csv /projects/beto/irowerojas/localLatin/runs/active/resubmit/taskb_mseed/aggregated_results.csv \
  --out_dir runs/active/resubmit/layer_diagnostics
```

Validation checks performed:

- `geometry_per_layer.csv`: 400 rows.
- Main join: 36 rows.
- Appendix join: 64 rows.
- Correlation summary: 1287 rows.
- Skipped inputs: 0 rows.
- Required diagnostics have no NaNs.
- Cosine quantiles are ordered.
- Effective rank is within `[1, dim]`.
