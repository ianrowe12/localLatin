# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP research project studying **Latin manuscript text retrieval** using internal representations of pre-trained language models. The core question: can embeddings from intermediate transformer layers retrieve related Latin manuscript fragments, and can post-processing (SIF weighting + ABTT/PC removal) fix the "anisotropy dip" that collapses retrieval in middle layers?

The dataset is 1,278 `.txt` files in 538 directories under `data/canon/`, where each directory represents one original text. The labelled retrieval corpus is 1,705 files in 840 directories under `data/canon_labelled/`, and the unlabelled query set is 2,238 files under `data/canon_unlabelled/`. Active work is paper resubmission — see GAMEPLAN.md. Primary metric: Assignment Accuracy.

## Environment & Running Code

**HPC**: NCSA Delta GPU cluster, `beto-delta-gpu` account, `gpuA100x4` partition.

```bash
# Environment activation (interactive)
module load miniforge3-python
conda activate localLatin

# In sbatch scripts, always use:
conda run -n localLatin python ...
```

**Python**: 3.10, conda env `localLatin`. Dependencies in `requirements.txt`. `pot` (Python Optimal Transport) is installed ad-hoc in some sbatch scripts.

**All scripts assume cwd is the repo root** (`/u/irowerojas/localLatin`). The `src/` directory is added to `sys.path` by scripts that import from it.

## Architecture

### Core library (`src/`)

- **`canon_retrieval.py`**: Dataset indexing, similarity matrices, acc@k, threshold sweeps. Imported by nearly everything.
- **`sif_abtt.py`**: SIF weights, ABTT/PC removal (`EmbeddingCleaner` class), token probabilities. The key post-processing module.
- **`pair_evaluation.py`**: Spearman, AUC-ROC, cosine similarity for sentence pairs.
- **`cli_utils.py`**: `parse_layers()` for `--layers "0-12"` or `"0,6,12"` syntax.

### Extraction CLIs (`src/`)

- **`extract_ff1_cli.py`** / **`extract_hidden_cli.py`**: T5 models (LaTa, PhilTa). FF1 uses forward pre-hooks on `wo` projection.
- **`extract_encoder_cli.py`**: BERT (LaBSE) and decoder-only models (Qwen, KaLM). Detects architecture via attribute inspection. Supports `--half_precision` and `--trust_remote_code`.

### Pipeline scripts (`scripts/`)

Grouped by purpose:
- **`scripts/resubmit/`**: Active paper resubmission pipeline (`run_resubmit_data_prep.py`, `run_taskb_mseed.py`, `visualize_taskb_mseed.py`, `run_resubmit_ig_comparison.py`, `run_leiden_examples.py`, `evaluate_vectors.py`, `visualize_resubmit.py`, `index_unlabelled.py`, `run_resubmit_unlabelled_retrieval.py`).
- **`scripts/ig/`**: IG artifact regeneration for the webapp (`run_ig_examples_pipeline.sh`, `run_phase12f_select_pair_examples.py`, `run_phase12f_visualize.py`).
- **`scripts/webapp/`**: Webapp data export (`export_webapp_data.sh`).
- **`scripts/common/`**: Shared helpers (`create_canon_split.py`, `data_prep.py`).
- **`scripts/_archive/`**: Stale phase 3-12 pipelines preserved for reproducibility.

Key active script: **`evaluate_vectors.py`** (Phase 9+ evaluator). Applies 5 methods (baseline, sif_only, sif_abtt_fixed, sif_abtt_optimal, whitening), learns threshold on train, evaluates on test.

### SLURM jobs (`slurm/`)

Grouped by purpose: **`slurm/resubmit/`** (active resubmission jobs), **`slurm/ig/`** (IG artifact regen), **`slurm/_archive/`** (stale phase 3-12 sbatch).

## Data Flow

```
data/canon_labelled/ (1,705 .txt files, 840 dirs)
  → scripts/common/data_prep.py → runs/active/resubmit/data/phase_resubmit_split.csv (50/50 train/test)
  → src/extract_*.py → runs/active/encoder_bases/<model_slug>/<repr>_<pooling>/*.npy
  → scripts/resubmit/evaluate_vectors.py → runs/active/resubmit/results/*.csv
  → scripts/resubmit/visualize_resubmit.py → overleaf_drafts/figures/
```

## Key Conventions

**Leak-free protocol (Phase 8+)**: Always pass `--split_csv` so SIF token probabilities use train files only. `EmbeddingCleaner` must be fit on train embeddings, then applied to all.

**Embedding file naming**: `{repr}_layer{N}_embeddings{suffix}.npy` where suffix is empty (mean), `_sif`, or `_lasttok`. Normalized files end `_norm.npy`.

**Layer indexing**: Hidden states 0 = embedding output, 1..N = transformer blocks. FF1 is 1-indexed (1..N). `--layers` accepts ranges like `0-12`.

**Encoder bases path**: `runs/active/encoder_bases/<model_slug>/<repr>_<pooling>/` where `model_slug = model_name.replace("/", "_")`. This is the canonical 6-model embedding cache.

**"Winnable" queries**: Only files whose directory has >= 2 members can have a correct retrieval answer. `is_winnable` in meta.csv tracks this.

**Assignment Accuracy** (Phase 10 primary metric): Measures whether a file is correctly routed to its existing directory or flagged as new. Optimized via D parameter in ABTT (replacing AUCROC optimization from Phase 9).

## Models

| Short Name | HuggingFace ID | Type | Layers |
|-----------|---------------|------|--------|
| LaTa | `bowphs/LaTa` | T5 Seq2Seq | 12 |
| PhilTa | `bowphs/PhilTa` | T5 Seq2Seq | 12 |
| LaBSE | `sentence-transformers/LaBSE` | BERT encoder | 12 |
| Qwen3-0.6B | `Qwen/Qwen3-Embedding-0.6B` | Decoder | 28 |
| KaLM-mini | `KaLM-Embedding/KaLM-embedding-multilingual-mini-instruct-v2.5` | Decoder | 24 |
| Qwen3-8B | `Qwen/Qwen3-Embedding-8B` | Decoder | 50 |

## Post-Processing Methods

| Method | Description |
|--------|-------------|
| `baseline` | Mean-pool, no correction |
| `sif_only` | SIF-weighted pooling (down-weights frequent tokens) |
| `sif_abtt_fixed` | SIF + remove top D=10 principal components |
| `sif_abtt_optimal` | SIF + ABTT with D tuned per layer on train set |
| `whitening` | PCA whitening fitted on train (consistently fails on this task) |

## Paths

- **Repo root**: `/projects/beto/irowerojas/localLatin` (symlinked at `/u/irowerojas/localLatin`)
- **Legacy in-repo symlink**: `./localLatin` is a self-referential symlink (`localLatin -> /projects/beto/irowerojas/localLatin`) left over from early setup. It is gitignored and NOT in the GitHub repo — it only exists on the HPC working copy. Do not depend on it; use `/u/irowerojas/localLatin` (the home-dir convenience symlink) or the repo root path directly.
- **Datasets**: `data/canon/` (1,278 raw files), `data/canon_labelled/` (1,705 labeled candidates in 840 dirs), `data/canon_unlabelled/` (2,238 unlabeled queries)
- **Active experiment outputs**: `runs/active/resubmit/`, `runs/active/encoder_bases/`, `runs/active/ig_examples/`, `runs/active/resubmit_bases/`
- **Off-repo archive** (old phases, reproducibility-sensitive): `/projects/beto/irowerojas/localLatin_archive/`
- **Paper drafts**: `overleaf_drafts/`
- **Project docs**: `docs/meetings/`, `docs/analyses/`, `docs/research/`
- **Stale code preserved for grep-ability**: `scripts/_archive/`, `slurm/_archive/`, `src/_archive/`

## Webapp (Git Subtree)

The scholar review webapp lives in a separate repo ([localLatin-webapp](https://github.com/ianrowe12/localLatin-webapp)) embedded as a git subtree at `web/`. Zero code imports cross the boundary — the webapp reads research data via configurable paths.

### Running the webapp

```bash
cd web && cp config.yaml.example config.yaml  # set data_root to ".."
cd .. && python -m web                         # from repo root
```

Frontend: `cd web/frontend && npm install && npm run dev` (or `npm run dev:mock` for mock data).

### Deployment rule

After pushing webapp, deployment, or frontend changes to `main`, verify the
`CI and Deploy` GitHub Actions run for the pushed commit finishes green before
calling the work done. If it fails, inspect the failed job logs, fix the cause
or rerun after clearing transient runner state, and report the final workflow
status. Stop local dev servers before triggering production deploys so `npm ci`
can clean `web/frontend/node_modules` without NFS `EBUSY` file-handle errors.

### Subtree workflow

```bash
# Pull latest webapp changes from the webapp repo
git subtree pull --prefix=web webapp main --squash

# Push web/ changes made in this repo back to the webapp repo
git subtree push --prefix=web webapp main
```

### Data contract

The webapp reads these files (relative to `data_root` in `web/config.yaml`):
- `data/canon_unlabelled/` — 2,238 query .txt files
- `data/canon_labelled/` — 840 directories of candidate .txt files
- `runs/active/resubmit/unlabelled/unlabelled_predictions.csv` — model predictions
- `runs/active/ig_examples/` — IG visualization artifacts (CSV + NPZ)
- `runs/active/resubmit/webapp/feedback.db` — auto-created SQLite

Run `bash scripts/webapp/export_webapp_data.sh` to verify all required data files are present.
