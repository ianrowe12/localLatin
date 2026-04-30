# Run 3 Agent 1 Three-Model Attribution Artifacts

Generated: 2026-04-29

## Contract

This run follows `PRD-layer-attribution.md`: produce evidence for LaTa, PhilTa,
and mT5-base only; use predeclared layer choices; treat IG and retrieval-adapted
MaRC as parallel attribution views; and do not present older 200-random or
old-layer bundles as current headline evidence.

Run 3 uses the Run 2 operational attribution layers:

| Model | Layer | D | Role |
|---|---:|---:|---|
| LaTa | 7 | 10 | train-selected retrieval layer |
| PhilTa | 1 | 10 | train-selected retrieval layer |
| mT5-base | 1 | 10 | train-selected retrieval layer |

The active output directory for this branch is:

```text
runs/active/ig_examples_200pos_run3_operational/
```

This intentionally does not overwrite the earlier two-model meeting bundle at
`runs/active/ig_examples_200pos/`, whose LaTa L4 and PhilTa L6 artifacts are
superseded for Run 3 main-table reporting.

## Entry Point

Submit the full smoke-plus-run job from this worktree:

```bash
sbatch slurm/ig/run_attribution_200pos_run3_operational.sbatch
```

The job refits PCs into the active run directory, samples 200 positive pairs per
model, generates IG artifacts, runs retrieval-adapted MaRC, persists pair
matrices, computes attribution metrics, and writes `manifest.json` plus
`artifact_inventory.csv`.

## Pooling Decision

Run 3 uses `TOKEN_FILTER=all` for IG and MaRC generation. This aligns artifact
targets with `scripts/ig/run_attribution_metrics.py`, whose forward path uses
plain attention-mask mean pooling. PCs are refit from `hidden_mean` caches
through the same no-filter pooling regime before ABTT artifacts are trusted.

## Expected Outputs

- `positive200_examples.csv`: 600 rows, 200 per model, all `gold_similar=1`.
- `pcs/<slug>/layer<L>_pcs.npz`: shape `(10, 768)` for the selected layer.
- `artifacts/<slug>/example*_pair_example.npz`: canonical IG and method NPZs.
- `retrieval_mark/artifacts/<slug>/example*_pair_example.npz`: MaRC sidecars.
- `attribution_metrics/summary.csv`: all three models with IG, MaRC, random, and inverse rows under baseline and ABTT.
- `manifest.json` and `artifact_inventory.csv`: validation receipts for later reporting agents.
