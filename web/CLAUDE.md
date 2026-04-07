# CLAUDE.md — localLatin Webapp

## Overview

Scholar review web application for the localLatin Latin manuscript retrieval project. Reviewers use this to manually verify model predictions (which text directory does a manuscript fragment belong to?).

## Architecture

- **Backend**: FastAPI app (`app.py`), runs with `python -m web` from the parent directory
- **Frontend**: React 18 + TypeScript + Vite + Tailwind at `frontend/`
- **Data**: All data loaded into memory at startup from configurable paths (`config.py` → `PathsConfig`)
- **Feedback**: SQLite via aiosqlite (`services/feedback_db.py`)
- **No ML dependencies**: Only fastapi, pandas, numpy. Optional `transformers` for token decoding.

## Configuration

`config.yaml` (gitignored) controls all paths. Key field:
- `data_root`: Points to the directory containing `canon_unlabelled/`, `canon_labelled/`, `runs/`
- Default: `"."` (overridable via `LOCALLATIN_DATA_ROOT` env var)
- Subtree mode: set to `".."` (parent = research repo root)
- Standalone: set to path of research repo

## Key Files

- `config.py` — Settings with `PathsConfig.resolve()` for path resolution
- `services/data_store.py` — `build_store()` loads all data at startup. The `DataStore` dataclass is the central data cache.
- `services/token_map_svc.py` — Loads NPZ artifacts, computes cosine similarity matrices, optional HuggingFace tokenizer decoding
- `services/text_tokenizer.py` — Mirrors `src/token_filtering.classify_token()` from the research repo without torch dependency
- `services/feedback_db.py` — SQLite CRUD for reviewer feedback
- `routers/` — One file per API domain (queries, predictions, token_map, feedback, stats)
- `models.py` — All Pydantic request/response models

## Data Contract

The webapp reads these from `data_root`:
- `canon_unlabelled/` — flat directory of 2,238 .txt query files
- `canon_labelled/` — 859 subdirectories, each with .txt candidate files
- `runs/active/resubmit/unlabelled/unlabelled_predictions.csv` — CSV with columns: model, file_id, filename, rank1_dir, rank1_score, ..., rank10_dir, rank10_score, layer, pooling
- `runs/active/ig_examples/phase12f_examples.csv` — IG example metadata
- `runs/active/ig_examples/artifacts/<model_slug>/` — NPZ files with keys: query_embeddings, candidate_embeddings, query_ig_baseline, query_ig_abtt, candidate_ig_baseline, candidate_ig_abtt, query_input_ids, candidate_input_ids, layer, D

## Frontend

- Mock mode: `npm run dev:mock` — uses synthetic data from `src/mock/`
- API types in `src/api/` mirror backend `models.py`
- Token classification duplicated in `src/utils/tokens.ts` (matches `services/text_tokenizer.py`)

## Running

```bash
# Backend (from research repo root or parent of this repo)
python -m web

# Frontend dev
cd frontend && npm install && npm run dev

# Frontend mock (no backend needed)
cd frontend && npm run dev:mock
```

## Git Subtree

This repo is embedded at `web/` in the research repo via git subtree. The flat layout (package contents at repo root) means the directory name becomes the Python package name.
