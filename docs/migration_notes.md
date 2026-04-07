# Migration Notes: Repo Reorganization 2026-04-06

## Summary

On 2026-04-06 the repo was reorganized for clarity and to free disk space. ~80 GB of stale `runs/` data was removed; active runs were renamed under `runs/active/`; scripts/sbatch were grouped by purpose.

## Backups

**Stale runs tarball** (full backup of every deleted/archived run folder):
```
/work/hdd/beto/irowerojas/locallatin_backup_2026-04-06/stale_runs.tar
```
This contains: `runs/phase{3,4,5,6,7,8,9,10,11,12}*`, `runs/_smoke_*`, `runs/_resubmit_phase12e_check*`, `runs/ff1_lata_postact`, `runs/redo_results`, `runs/paper_assets`, `runs/paper_release_assets`, `runs/paper_track_handoff`, and stray `runs/slurm-15640239.{err,out}`.

**Off-repo archive** (reproducibility-sensitive artifacts kept on `/projects/beto/`):
```
/projects/beto/irowerojas/localLatin_archive/
├── redo_results/        (295M — referenced by analysis_redo_exp1_phase8.md)
├── phase8_results/      (3.2M — anisotropy dip discovery figures)
├── phase11/             (20M — PHASE11_ANALYSIS.md + tracked figures/CSVs)
├── phase12/             (363M — original IG/FA token attribution)
├── phase12b/            (5M — deep layer + token delta analysis)
└── phase12d/            (280K — token filtering sanity check)
```

## What moved where

### Top-level docs → `docs/`
- `MEETING_SUMMARY_*` → `docs/meetings/`
- `analysis_*.md` → `docs/analyses/`
- `research/run1_*.md` → `docs/research/`

### Active run folders renamed under `runs/active/`
- `runs/phase_resubmit/`       → `runs/active/resubmit/`
- `runs/phase_resubmit_bases/` → `runs/active/resubmit_bases/`
- `runs/phase9_bases/`         → `runs/active/encoder_bases/`
- `runs/phase12f_examples/`    → `runs/active/ig_examples/`

Symlinks at the old paths preserve backward compatibility for one week, then will be removed in a final cleanup pass (Phase 5).

### Scripts grouped by purpose
- `scripts/resubmit/` — active resubmission pipeline
- `scripts/ig/` — IG artifact regeneration
- `scripts/webapp/` — webapp data export
- `scripts/common/` — shared helpers
- `scripts/_archive/` — stale phase 3-12 pipelines (preserved in git for grep-ability)

### SLURM jobs grouped by purpose
- `slurm/resubmit/` — active resubmission jobs
- `slurm/ig/` — IG artifact jobs
- `slurm/_archive/` — stale phase 3-12 sbatch files

### Datasets moved into `data/`
- `canon/`            → `data/canon/`
- `canon_labelled/`   → `data/canon_labelled/`
- `canon_unlabelled/` → `data/canon_unlabelled/`

### Other moves
- `SIF/` → `third_party/SIF/` (untracked, plain `mv`)
- Stale `src/` CLIs → `src/_archive/`

## What was deleted

**Hard-deleted from disk** (gitignored — no git history to restore from; tarball backup at `/work/hdd/beto/irowerojas/locallatin_backup_2026-04-06/`):
- `runs/phase{3,4,5,6,7,8}*` — early phases superseded by later work
- `runs/phase9/experiment2/` (16G cache), `runs/phase10/experiment2/` (46G cache)
- `runs/phase11_filtered`, `phase11_release`, `phase12_filtered`, `phase12_release`, `phase12c*`, `phase12d_filtered`, `phase12e_release`
- `runs/ff1_lata_postact/`
- `runs/_smoke_*`, `_resubmit_phase12e_check*`
- `runs/paper_assets`, `paper_release_assets`, `paper_track_handoff`
- Stray `runs/slurm-15640239.{err,out}` log files
- `results/` (single ancient run dir from Jan 2026)

**Removed via `git rm`** (recoverable from git history):
- `AGENTS.md` (superseded by CLAUDE.md + GAMEPLAN.md)
- `latinLocal.ipynb` (456-byte stub)
- `README.md` (replaced with new minimal entry-point version)
- `phase_0_notebooks/` (4 Colab quickstart notebooks from January)
- `dataset/` (legacy Phase 9 TSVs, no longer referenced)

## Disk impact

- `runs/` shrunk from ~80 GB to ~13 GB
- `/projects/beto/` quota usage reduced by ~70 GB
- Tarball backup at `/work/hdd/beto/irowerojas/`: ~70 GB (uncompressed)
- Off-repo archive at `/projects/beto/irowerojas/localLatin_archive/`: ~700 MB

## How to restore

If you need an archived folder back:
```bash
# From off-repo archive
cp -r /projects/beto/irowerojas/localLatin_archive/phase11 runs/_recovered_phase11/

# From tarball
cd /tmp
tar -xf /work/hdd/beto/irowerojas/locallatin_backup_2026-04-06/stale_runs.tar runs/phase10/experiment1
mv runs/phase10/experiment1 /projects/beto/irowerojas/localLatin/runs/_recovered_phase10_experiment1/
```

If you need an archived script back:
```bash
git mv scripts/_archive/run_phase12c_retrieval_attribution.py scripts/
# or restore from history
git log --diff-filter=D -- scripts/run_phase12c_retrieval_attribution.py
```
