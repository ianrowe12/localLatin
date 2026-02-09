# localLatin: Canon FF1 retrieval

This repo contains the notebooks and helper code for canon folder-retrieval experiments using LaTa FF1 post-activation representations.

## Colab quickstart (recommended)

Important: `drive.mount()` is supported in **Colab (browser notebooks)**. If you run notebooks from VS Code attached to a remote kernel, you generally cannot use `drive.mount()` and would need other approaches (e.g. Drive API). Since we’re running in Colab browser, we use `drive.mount()` for persistence.

1. Open Colab (browser) and run the bootstrap notebook:
   - `notebooks/00_gitClone.ipynb`

This does:
- clone `https://github.com/ianrowe12/localLatin.git`
- `pip install -r requirements.txt`
- `drive.mount('/content/drive')`
- set `REPO_ROOT`, `CANON_ROOT`, `RUNS_ROOT`

2. Put your dataset on Drive:
- Data: `/content/drive/MyDrive/localLatin_data/canon/`
- Outputs (auto-created): `/content/drive/MyDrive/localLatin_runs/ff1_lata_postact/`

3. Run notebooks in order:
- `notebooks/01_index_canon.ipynb`
- `notebooks/02_extract_ff1_lata.ipynb`
- `notebooks/03_eval_retrieval_ff1.ipynb`

## Drive layout (recommended)

```
MyDrive/
  localLatin_data/
    canon/                     # dataset (1278 .txt files)
  localLatin_runs/
    ff1_lata_postact/           # outputs (meta.csv, embeddings, curves)
```

## Notes
- `canon/` and `runs/` are excluded from git (see `.gitignore`).
- You can override paths with env vars:
  - `REPO_ROOT`, `CANON_ROOT`, `RUNS_ROOT`

## Delta (SSH) runbook

This is the currently working setup on Delta using `miniforge3-python`.

### 0) Recommended directory layout
```bash
# Home layout (example)
~/localLatin           # repo
~/localLatin/canon     # dataset root (folder tree of .txt files)
~/localLatin/runs      # outputs (created automatically)
```

### 1) One-time environment setup
```bash
module load miniforge3-python
conda create -y -n localLatin python=3.10
conda activate localLatin
pip install -r /u/irowerojas/localLatin/requirements.txt
```

**Important (Delta / non-interactive shells):** ALWAYS install
packages via `conda run` so they land in the correct env:
```bash
module load miniforge3-python
conda run -n localLatin python -m pip install -r /u/irowerojas/localLatin/requirements.txt
conda run -n localLatin python -c "import sklearn; print(sklearn.__version__)"
```

### 2) Submit the full pipeline run
```bash
sbatch --account=beto-delta-gpu --partition=gpuA100x4 --gpus=1 --cpus-per-task=4 --mem=64g --time=06:00:00 \
  --job-name=canon_runall \
  --output=slurm-%j.out --error=slurm-%j.err \
  --wrap="bash -lc 'module load miniforge3-python && export REPO_ROOT=/u/irowerojas/localLatin CANON_ROOT=/u/irowerojas/localLatin/canon RUNS_ROOT=/u/irowerojas/localLatin/runs/ff1_lata_postact && conda run -n localLatin bash /u/irowerojas/localLatin/scripts/run_all.sh'"
```

### 3) Verify outputs
```bash
ls -la /u/irowerojas/localLatin/runs/ff1_lata_postact
```

### 4) Phase 3 (hidden layer 12, mean pooling)
This phase uses the existing Phase 2 run as the baseline and runs training-free
filtering/projection on **hidden layer 12 mean-pooled embeddings**. No notebooks.

1) Run GAP screening sweeps (CPU):
```bash
sbatch /u/irowerojas/localLatin/slurm/phase3_screen_hidden12_mean.sbatch
```

2) After the screening job finishes, promote top configs to full eval (CPU):
```bash
sbatch /u/irowerojas/localLatin/slurm/phase3_full_eval_hidden12_mean.sbatch
```

3) Update the placeholder run dir before full eval:
- Edit `PHASE3_RUN_DIR` in `slurm/phase3_full_eval_hidden12_mean.sbatch` to the new
  `runs/phase3_hidden12_mean/run_YYYYMMDD_HHMMSS` folder created by the screening job.

### Phase 3 one-shot (GPU account, same style as Phase 2)
If your runs live under `/u/irowerojas/runs` (outside the repo), set paths explicitly:
```bash
sbatch --account=beto-delta-gpu --partition=gpuA100x4 --gpus=1 --cpus-per-task=4 --mem=64g --time=06:00:00 \
  --job-name=phase3_all --output=slurm-%j.out --error=slurm-%j.err \
  --wrap="bash -lc 'module load miniforge3-python && export REPO_ROOT=/u/irowerojas/localLatin RUNS_ROOT=/u/irowerojas/runs BASE_RUN_DIR=/u/irowerojas/runs/ff1_lata_postact/run_20260125_154356 PHASE3_ROOT=/u/irowerojas/runs/phase3_hidden12_mean && conda run -n localLatin bash /u/irowerojas/localLatin/scripts/run_phase3.sh'"
```

Outputs live under:
```
~/runs/phase3_hidden12_mean/run_YYYYMMDD_HHMMSS/
  phase3_baseline_summary.json
  phase3_screening_scoreboard.csv
  phase3_full_eval_scoreboard.csv
  derived/...
```

### 5) Phase 4 (hard-query recovery + margin diagnostics)
This phase analyzes where baseline retrieval fails and whether variance filtering
recovers those failures (or regresses prior hits). It runs on CPU and writes a new
Phase 4 run folder under `$WORK/runs/phase4_hard_queries`.

1) Update the placeholder run dir in the sbatch script:
- Edit `PHASE3_RUN_DIR` in `slurm/phase4_hard_query_analysis.sbatch` to the
  `runs/phase3_hidden12_mean/run_YYYYMMDD_HHMMSS` folder you want to analyze.

2) Submit the Phase 4 analysis job:
```bash
sbatch /u/irowerojas/localLatin/slurm/phase4_hard_query_analysis.sbatch
```

Outputs live under:
```
~/runs/phase4_hard_queries/run_YYYYMMDD_HHMMSS/
  phase4_config.json
  hard_queries.csv
  recovery_regression_summary.csv
  recovery_regression_pivot.csv
  margin_summary.csv
  figures/ (if enabled)
```

### 6) Phase 5 (training-free transforms + retrieval screening)
This phase extends the Phase 3 pipeline with anisotropy correction and retrieval-based
screening (acc@k_winnable). It runs on CPU and writes a new Phase 5 run folder under
`$WORK/runs/phase5_hidden12_mean`.

1) Run Phase 5 screening sweeps (CPU):
```bash
sbatch /u/irowerojas/localLatin/slurm/phase5_screen_hidden12_mean.sbatch
```

2) After the screening job finishes, promote top configs to full eval (CPU):
```bash
sbatch /u/irowerojas/localLatin/slurm/phase5_full_eval_hidden12_mean.sbatch
```

3) Update the placeholder run dir before full eval:
- Edit `PHASE5_RUN_DIR` in `slurm/phase5_full_eval_hidden12_mean.sbatch` to the new
  `runs/phase5_hidden12_mean/run_YYYYMMDD_HHMMSS` folder created by the screening job.

Outputs live under:
```
~/runs/phase5_hidden12_mean/run_YYYYMMDD_HHMMSS/
  phase5_baseline_summary.json
  phase5_screening_scoreboard.csv
  phase5_full_eval_scoreboard.csv
  derived/...
```

### 7) Phase 6 (generalization checks across models/layers)
Phase 6 applies the same training-free transforms across multiple layers and models.
It uses **extract-if-missing** for base embeddings, then evaluates:
- baseline
- variance drop 25%
- best Phase 5 candidate (highest acc@5_winnable)

1) Update the Phase 5 run pointer and submit:
```bash
sbatch /u/irowerojas/localLatin/slurm/phase6_generalize.sbatch
```

2) Before running, edit `PHASE5_RUN_DIR` in:
- `slurm/phase6_generalize.sbatch`

Outputs live under:
```
~/runs/phase6_generalization/run_YYYYMMDD_HHMMSS/
  phase6_config.json
  phase6_scoreboard.csv
  derived/...
```

### 8) Phase 7 (SIF/ABTT + all-layers sweep + MUSTS)
Phase 7 applies official SIF weighting and ABTT (PC removal) across **all layers**
for LaTa/PhilTa, then evaluates generalization on MUSTS (Spearman correlation).

Run the full pipeline:
```bash
bash /u/irowerojas/localLatin/scripts/run_phase7_pipeline.sh
```

Key outputs live under:
```
~/runs/phase7_results/run_YYYYMMDD_HHMMSS/
  phase7_layer_sweep.csv
  phase7_musts_results.csv
  derived/...
```

Useful overrides:
```bash
PHASE7_ROOT=~/runs/phase7_results \
MODELS="bowphs/LaTa,bowphs/PhilTa" \
REPRS="hidden,ff1" \
LAYERS_HIDDEN="0-12" \
LAYERS_FF1="1-12" \
SIF_A=0.001 \
ABTT_D=10 \
MUSTS_LANGS="sinhala,tamil,english,french" \
bash /u/irowerojas/localLatin/scripts/run_phase7_pipeline.sh
```

### Phase 4 comparisons for Phase 5 candidates
To analyze hard-query recovery/regression for a specific Phase 5 derived run, call
the Phase 4 CLI directly and point it at the derived run directory:
```bash
python src/phase4_hard_query_analysis_cli.py \
  --phase3_run_dir "$WORK/runs/phase5_hidden12_mean/run_YYYYMMDD_HHMMSS" \
  --compare_run_dir "$WORK/runs/phase5_hidden12_mean/run_YYYYMMDD_HHMMSS/derived/<derived_run_dir>" \
  --compare_label "phase5_candidate" \
  --layer 12 \
  --k_list "1,3,5"
```

Phase 4 outputs are written under:
```
$WORK/runs/phase4_hard_queries/run_YYYYMMDD_HHMMSS/
```

## SSH + Drive API persistence (headless)

If you work on the Colab A100 VM over SSH (no browser notebooks), use a **service account** to sync outputs to Drive.

### One-time setup
- Enable Google Drive API in your GCP project
- Create a service account + JSON key
- Create Drive folder `localLatin_runs` and share it with the service account email as **Editor**

### Per runtime
1. Start a Colab runtime and mount Drive once in a browser tab:
   - `drive.mount('/content/drive')`
2. Copy the service account key from Drive to the VM:
   - from `/content/drive/MyDrive/localLatin_runs/sa_drive_key.json`
   - to `/content/sa_drive_key.json`
3. SSH into the VM and run extraction/eval
4. Sync outputs to Drive:
```bash
python -m src.drive_sync --local_run_dir /content/localLatin/runs/ff1_lata_postact/run_YYYYMMDD_HHMMSS
```

You can override the key path:
```bash
DRIVE_SA_KEY_PATH=/content/sa_drive_key.json python -m src.drive_sync --local_run_dir /content/localLatin/runs/ff1_lata_postact/run_YYYYMMDD_HHMMSS
```
