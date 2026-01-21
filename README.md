# localLatin: Canon FF1 retrieval

This repo contains the notebooks and helper code for canon folder-retrieval experiments using LaTa FF1 post-activation representations.

## Colab quickstart (Drive + GitHub)

1. **Clone the repo in Colab**
```bash
!git clone https://github.com/<your-username>/localLatin.git
```

2. **Mount Drive and set paths** (first cell in each notebook already does this)
- Data: `/content/drive/MyDrive/localLatin_data/canon/`
- Outputs: `/content/drive/MyDrive/localLatin_runs/ff1_lata_postact/`

3. **Run notebooks in order**
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
