# localLatin

NLP research project on **Latin manuscript text retrieval** using internal representations of pre-trained language models.

The core question: can embeddings from intermediate transformer layers retrieve related Latin manuscript fragments, and can post-processing (SIF weighting + ABTT/PC removal) fix the "anisotropy dip" that collapses retrieval in middle layers?

## Where things live

| Path | Contents |
|---|---|
| `src/` | Core Python library: `canon_retrieval.py`, `sif_abtt.py`, `canon_split_v2.py`, `direct_logit_attribution.py`, extraction CLIs |
| `scripts/resubmit/` | Active resubmission pipeline (data prep, Task B m-seed, IG comparison, Leiden examples) |
| `scripts/ig/` | IG visualization artifact regeneration for the webapp |
| `scripts/webapp/` | Webapp data export utilities |
| `scripts/common/` | Shared helpers (canon split, data prep) |
| `slurm/resubmit/` | Active SLURM jobs for the resubmission pipeline |
| `slurm/ig/` | SLURM jobs for IG artifact regeneration |
| `runs/active/resubmit/` | Active resubmission outputs (predictions, results, distributions, M-seed Task B) |
| `runs/active/encoder_bases/` | 6-model embedding bases for the dataset |
| `runs/active/resubmit_bases/` | Curated subset embeddings for resubmission |
| `runs/active/ig_examples/` | NPZ artifacts for webapp token-attribution visualization |
| `data/canon/` | Source manuscript fragments (538 directories) |
| `data/canon_labelled/` | Labeled retrieval candidates (840 directories, 1705 files) |
| `data/canon_unlabelled/` | Unlabeled query manuscripts (2238 files) |
| `web/` | FastAPI + React webapp for scholar review (git subtree) |
| `deploy/` | Deployment configs (systemd, nginx, deploy.sh) |
| `overleaf_drafts/` | LaTeX paper draft + figures |
| `docs/` | Project documentation: meeting summaries, analyses, research surveys |
| `third_party/SIF/` | Reference SIF implementation (read-only) |
| `*/_archive/` | Stale code preserved for reproducibility (do not depend on) |

## Documentation

- **Project context, environment, conventions**: `CLAUDE.md`
- **Current sprint plan**: `GAMEPLAN.md`
- **Recent meeting summaries**: `docs/meetings/`
- **Historical phase analyses**: `docs/analyses/`
- **Research surveys (XAI methods, geometry, interpretability)**: `docs/research/`

## Common workflows

```bash
# Activate environment (HPC)
module load miniforge3-python
conda activate localLatin

# Run the resubmission pipeline (extracts, retrieves, evaluates, plots)
sbatch slurm/resubmit/resubmit_launch_all.sh

# Run M-seed Task B evaluation
sbatch slurm/resubmit/resubmit_taskb_mseed.sbatch

# Regenerate IG visualization artifacts
sbatch slurm/ig/ig_examples.sbatch

# Run the webapp (local development)
cd web && cp config.yaml.example config.yaml  # set data_root to ".."
cd .. && python -m web

# Deploy webapp to production path (ai.csr.uky.edu/locallatin/)
bash deploy/deploy.sh
```

## Repo conventions

- **Python**: 3.10, conda env `localLatin`. See `requirements.txt`.
- **HPC**: NCSA Delta, `beto-delta-gpu` account, `gpuA100x4` partition.
- **Working directory**: scripts assume cwd is the repo root.
- **Path discipline**: never hardcode `runs/phase_resubmit/...` — use `runs/active/resubmit/...`.
- **Webapp data contract**: see `CLAUDE.md` and `web/config.production.yaml`.
