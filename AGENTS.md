# Repository Guidelines

## Project Structure & Module Organization
`src/` contains reusable Python modules and CLI entrypoints (`*_cli.py`) for extraction, retrieval, and post-processing (SIF/ABTT, attribution targets). `scripts/` contains phase-specific pipelines and analysis/plot scripts (for example `scripts/run_phase11_evaluate.py`, `scripts/run_phase12_attribution.py`). `slurm/` holds Delta batch jobs and launchers (`phase*.sbatch`, `phase12_launch_all.sh`). Research outputs go in `runs/` and `results/`; source data lives under `canon/` (do not commit generated data or logs). Paper drafts are in `overleaf_drafts/`.

## Build, Test, and Development Commands
Use the Delta/Miniforge workflow and the shared `localLatin` conda env.

```bash
module load miniforge3-python
conda run -n localLatin python -m pip install -r requirements.txt
```

Common local checks:

```bash
conda run -n localLatin python -m py_compile src/*.py scripts/*.py
conda run -n localLatin python scripts/run_phase12_attribution.py --help
```

Typical HPC execution:

```bash
sbatch slurm/phase12_prepare.sbatch
sbatch slurm/phase12_launch_all.sh
```

Always use `conda run -n localLatin ...` in scripts and sbatch jobs; do not run `pip install` inside batch jobs.

## Coding Style & Naming Conventions
Python code uses 4-space indentation, `snake_case`, and short docstrings for non-obvious logic. Prefer type hints on public helpers and `pathlib.Path` for paths. Keep CLI args explicit and reproducible (seed, layer list, input/output paths). Naming patterns matter: phase scripts use `run_phase*.py`, SLURM jobs use matching `phase*.sbatch`, and model output directories use slugified names like `Qwen_Qwen3-Embedding-0.6B` (`model_name.replace("/", "_")`).

## Testing Guidelines
There is no formal `tests/` suite yet. Treat validation as reproducible smoke tests:
- run script `--help` and `py_compile`
- test on a small sample CSV before full jobs
- verify output shapes/counts in `runs/.../*.npz` and CSV schemas
- compare key metrics against prior phase outputs (for example `runs/phase11/phase11_results.csv`)

## Commit & Pull Request Guidelines
History favors short, imperative summaries with phase context (examples: `Phase 9 halfway`, `analyses corrections`). Prefer `Phase N: <change>` when possible. PRs should include: purpose, affected phases/models/layers, exact commands run (`python`/`sbatch`), output locations under `runs/`, and any changed figures/tables. Do not include large generated artifacts, `canon/` data, or `slurm-*.out/.err` logs.

## Delta / HPC Notes
On Delta, use `gpuA100x4` even for CPU-light jobs in this project setup. Keep train/test fitting leak-free (fit PCs/SIF stats on train split only), and rely on checkpointable scripts for long attribution runs.
