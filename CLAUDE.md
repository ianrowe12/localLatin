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
- **`scripts/resubmit/`**: Active paper resubmission pipeline (`run_resubmit_data_prep.py`, `run_taskb_mseed.py`, `visualize_taskb_mseed.py`, `run_resubmit_ig_comparison.py`, `run_leiden_examples.py`, `evaluate_vectors.py`, `visualize_resubmit.py`, `index_unlabelled.py`, `run_resubmit_unlabelled_retrieval.py`, `build_qq_matrices.py`).
- **`scripts/ig/`**: IG artifact regeneration for the webapp (`run_ig_examples_pipeline.sh`, `run_phase12f_select_pair_examples.py`, `run_phase12f_visualize.py`).
- **`scripts/webapp/`**: Webapp data export (`export_webapp_data.sh`), deploy data packaging (`make_data_release.sh`), deployment smoke checks (`smoke_reviewer_pilot.py`).
- **`scripts/common/`**: Shared helpers (`create_canon_split.py`, `data_prep.py`).
- **`scripts/paper/`**: Overleaf mirror sync (`sync_paper_repo.sh`).
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
- **Paper drafts**: `overleaf_drafts/` — any agent editing prose here MUST follow `docs/research/paper_writing_guidelines.md` (skimmability, crisp contributions, grouped related work, self-contained captions, no em-dashes).
- **Project docs**: `docs/meetings/`, `docs/analyses/`, `docs/research/`
- **Stale code preserved for grep-ability**: `scripts/_archive/`, `slurm/_archive/`, `src/_archive/`

## PR & Merge Workflow

Every implementation change goes through a pull request. No direct commits to `main`.

**Branch naming**: `issue-<N>-<slug>`, e.g. `issue-47-taskb-rerun`. One branch per issue, unless
several issues touch the same files (docs bundles), in which case name it after all of them
(`issue-35-55-57-docs`).

**Pull requests**: always target `main`. The PR body must contain `Closes #N` for every issue the
branch resolves, so merging closes them automatically.

**Merge gate**: a PR may only be merged once both of these hold.

1. **CI is green** on the head commit (workflow added in issue #54).
2. **An independent code-review agent has passed the PR** — a fresh-context reviewer, never the
   agent that wrote the code. The implementer marks the PR ready; the reviewer approves or
   requests changes. Enable auto-merge only after both conditions hold.

**Commit messages**: NEVER add a `Co-Authored-By:` trailer (standing preference from Ian). This
applies to every commit in this repo, agent-authored or not. Use conventional-commit style
subjects (`feat:`, `fix:`, `docs:`, `chore:`) and keep the body focused on what changed and why.

## GPU Budget

Allocation balances as of **2026-08-23**:

| Account | Partition | Balance |
|---------|-----------|---------|
| `beto-delta-gpu` | `gpuA100x4` | **54 h** |
| `beto-delta-cpu` | `cpu` | **338 h** |

Rules:

- **Every sbatch must set a realistic `--time`.** SLURM charges against the *reserved* wall time,
  not the elapsed run time, so an oversized `--time` burns budget even when the job finishes in
  minutes. Estimate from a previous run of the same job and add a modest margin.
- **Prefer the CPU partition** (`--partition=cpu`, `--account=beto-delta-cpu`) whenever the job
  does not need a GPU. In particular, anything that only reads existing embeddings from
  `runs/active/` (evaluation, threshold sweeps, clustering, figures, split verification) is CPU
  work and must not request `--gpus-per-node`.
- **GPU submissions for the current push are limited to issue #47.** Any other GPU job needs
  explicit approval from Ian before submission.

## Paper Repo (Overleaf Sync)

The paper is edited in this repo at `overleaf_drafts/` and mirrored to
[localLatin-paper](https://github.com/ianrowe12/localLatin-paper), which the Overleaf
project syncs with (Overleaf's GitHub menu: "Pull/Push GitHub changes").

```bash
# After paper PRs merge to main: snapshot overleaf_drafts/ to the paper repo
bash scripts/paper/sync_paper_repo.sh push

# After edits arrive from Overleaf: bring them back as a review branch
bash scripts/paper/sync_paper_repo.sh pull
```

`push` commits a clean snapshot to localLatin-paper main; Ian then pulls in Overleaf.
`pull` creates a `paper-sync-*` branch here with the Overleaf-side edits for a normal PR.
Never edit localLatin-paper directly from this side; it is a snapshot mirror, not a git subtree
(subtree was the original #33 plan; this git lacks the subtree command and a mirror is simpler).
`push` refuses to overwrite un-pulled Overleaf edits unless `SYNC_FORCE=1`; `pull` is
deletion-aware and refuses to run over local `overleaf_drafts/` changes.

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
- `runs/active/resubmit/unlabelled/unlabelled_predictions_<variant>.csv` — model predictions,
  one file per post-processing variant (`raw`, `abtt`, `sif`, `sif_abtt`; `sif_abtt` is the
  default). The pre-variant `unlabelled_predictions.csv` is stale and is never served.
- `runs/active/resubmit/unlabelled/qq_sim_<model_slug>.npz` — query-query cosine matrix per
  model (2,238 x 2,238 float16, ~10 MB each), built by
  `scripts/resubmit/build_qq_matrices.py` at exactly the deployed `sif_abtt` layer/cleaner.
  Reviewer-created directories are scored from these. Optional: a model without one simply
  serves no reviewer-directory candidates.
- `runs/active/ig_examples/` — IG visualization artifacts (CSV + NPZ)
- `runs/active/resubmit/webapp/feedback.db` — auto-created SQLite

Run `bash scripts/webapp/export_webapp_data.sh` to verify all required data files are present
(add `--strict` to exit non-zero on anything missing).

Everything under `runs/` in that list is gitignored, so it never reaches the deploy host through
`git pull`. Package it with `bash scripts/webapp/make_data_release.sh --tag data-YYYYMMDD`, publish
the tarball plus its `.sha256` as a GitHub Release, and set the repo variable `DATA_RELEASE_TAG`;
`deploy/deploy.sh` then downloads, verifies and installs it. Every archive member is confined to
`runs/active/`, which is what keeps the reviewer feedback DB at `data/feedback.db` untouchable by a
deploy. See `deploy/REVIEWER_PILOT_READINESS.md`.
