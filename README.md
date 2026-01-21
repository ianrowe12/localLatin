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
