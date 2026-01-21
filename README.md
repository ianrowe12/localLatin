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
