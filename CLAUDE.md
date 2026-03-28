# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NLP research project studying **Latin manuscript text retrieval** using internal representations of pre-trained language models. The core question: can embeddings from intermediate transformer layers retrieve related Latin manuscript fragments, and can post-processing (SIF weighting + ABTT/PC removal) fix the "anisotropy dip" that collapses retrieval in middle layers?

The dataset is 1,278 `.txt` files in 538 directories under `canon/`, where each directory represents one original text. The project progresses through numbered phases (1-12), each addressing a specific experimental question. Primary metric: Assignment Accuracy.

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

- **`evaluate_vectors.py`**: Phase 9+ evaluator. Applies 5 methods (baseline, sif_only, sif_abtt_fixed, sif_abtt_optimal, whitening), learns threshold on train, evaluates on test.
- **`run_phase8_canon_sweep.py`**: Phase 8 STS canon sweep with leak-free split.

### SLURM jobs (`slurm/`)

Each phase has corresponding `.sbatch` files.

## Data Flow

```
canon/ (1,278 .txt files, 538 dirs)
  → src/index_canon_cli.py → meta.csv
  → scripts/data_prep.py → phase9_split.csv (50/50 train/test)
  → src/extract_*.py → runs/phase9_bases/<model_slug>/<repr>_<pooling>/*.npy
  → scripts/evaluate_vectors.py → results.csv (AUCROC, Acc@1, Assignment Acc, etc.)
  → scripts/visualize_*.py → figures/
```

## Key Conventions

**Leak-free protocol (Phase 8+)**: Always pass `--split_csv` so SIF token probabilities use train files only. `EmbeddingCleaner` must be fit on train embeddings, then applied to all.

**Embedding file naming**: `{repr}_layer{N}_embeddings{suffix}.npy` where suffix is empty (mean), `_sif`, or `_lasttok`. Normalized files end `_norm.npy`.

**Layer indexing**: Hidden states 0 = embedding output, 1..N = transformer blocks. FF1 is 1-indexed (1..N). `--layers` accepts ranges like `0-12`.

**Phase 9 bases path**: `runs/phase9_bases/<model_slug>/<repr>_<pooling>/` where `model_slug = model_name.replace("/", "_")`.

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
- **Dataset**: `canon/` (1,278 files)
- **Experiment outputs**: `runs/` (not in git)
- **Analysis docs**: `runs/phase9/EXPERIMENT_1_ANALYSIS.md`, `runs/phase10/experiment1/ANALYSIS_*.md`
- **Paper drafts**: `overleaf_drafts/`
