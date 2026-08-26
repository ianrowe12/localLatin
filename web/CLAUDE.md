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
- `services/token_map_svc.py` — Loads NPZ artifacts, computes cosine similarity matrices, optional HuggingFace tokenizer decoding. The token-map endpoints take `?method=&variant=`; without them the response carries every persisted method x variant matrix (7 x 4 dense grids), so the UI always sends both. `available_methods` / `available_variants` always report the artifact's full contents regardless of the filters.
- `services/text_tokenizer.py` — Mirrors `src/token_filtering.classify_token()` from the research repo without torch dependency
- `services/feedback_db.py` — SQLite CRUD for reviewer feedback, plus the append-only
  `reviewer_dirs` / `reviewer_dir_members` tables. Both are created in `_migrate()` with
  `CREATE TABLE IF NOT EXISTS`, so opening a pre-#95 database adds them and touches no
  existing row. The feedback log stays append-only: nothing on the reviewer-directory path
  updates or deletes a feedback row.
- `services/qq_matrix.py` — reads `qq_sim_<slug>.npz` and answers "how similar is this query
  to this set of member queries" (max over members, self excluded).
- `services/reviewer_dirs.py` — scoring and shaping of reviewer directories, shared by the
  predictions, reviewer_dirs and feedback routers. Status is **derived**, never stored: a
  directory is `matched` once any non-member query reaches `NO_MATCH_BAND` against it.
- `bands.py` — the 0.5 / 0.7 confidence thresholds. **The backend owns these numbers** because
  it decides directory status with them; `GET /api/models` serves them as `confidence_bands`
  and the frontend's `src/api/bands.ts` reads them rather than hardcoding a copy.
- `routers/` — One file per API domain (queries, predictions, token_map, feedback,
  reviewer_dirs, stats)
- `models.py` — All Pydantic request/response models

## Reviewer directories (issue #95)

`POST /api/reviewer_dirs {query_file_id, label?}` -> 201 `ReviewerDir`. Any signed-in, approved
reviewer may call it; there is no extra role gate, since reviewers are exactly who the feature
is for. The response carries the *computed* status, which is `awaiting_match` unless some other
query already scores at or above the band against the seed.

Created directories merge into every subsequent predictions response as extra candidates with
`source: 'reviewer'`, scored live from the q-q matrix (max over member documents). They are
**appended after** the model's ten, never interleaved, so a model candidate's rank still means
what it always did and every historical feedback row keeps pointing at the candidate its
reviewer actually chose. `MAX_CANDIDATE_RANK` in `models.py` is what lets feedback record a
rank past 10.

Confirming a reviewer directory as the correct answer appends that query to the directory's
members (idempotent, `INSERT OR IGNORE`), so later queries are scored against the whole group.

## Data Contract

The webapp reads these from `data_root`:
- `canon_unlabelled/` — flat directory of 2,238 .txt query files
- `canon_labelled/` — 859 subdirectories, each with .txt candidate files
- `runs/active/resubmit/unlabelled/unlabelled_predictions_<variant>.csv` — one CSV per
  post-processing variant (`raw`, `abtt`, `sif`, `sif_abtt`), with columns: model, variant,
  file_id, filename, rank1_dir, rank1_score, ..., rank10_dir, rank10_score, layer, pooling.
  The path pattern, the served variant list, and the default (`sif_abtt`) come from
  `PathsConfig`. Only the default variant is read at startup; the others load on first
  request. The pre-variant `unlabelled_predictions.csv` is stale and is never a fallback.
- `runs/active/resubmit/unlabelled/qq_sim_<model_slug>.npz` — query-query cosine matrix per
  model (2,238 x 2,238 float16, ~10 MB), keys: `sim`, `file_ids`, `excluded`, `meta`. Built at
  exactly the deployed `sif_abtt` configuration by
  `scripts/resubmit/build_qq_matrices.py`, which verifies itself against
  `unlabelled_predictions_sif_abtt.csv` before writing. Reviewer-created directories are scored
  from these. Path pattern is `PathsConfig.qq_matrix_pattern`; matrices are discovered at
  startup and loaded lazily on first use. A model with no matrix serves no reviewer-directory
  candidates and reports `supports_reviewer_dirs: false` on `/api/models`.
- `runs/active/ig_examples/phase12f_examples.csv` — IG example metadata
- `runs/active/ig_examples/artifacts/<model_slug>/` — NPZ files with keys: query_embeddings, candidate_embeddings, query_ig_baseline, query_ig_abtt, candidate_ig_baseline, candidate_ig_abtt, query_input_ids, candidate_input_ids, layer, D

## Frontend

- Mock mode: `npm run dev:mock` — uses synthetic data from `src/mock/`
- API types in `src/api/` mirror backend `models.py`
- Unit tests: `npm test` (vitest, jsdom + Testing Library; setup in `src/test/setup.ts`)
- Post-processing variant: one app-wide choice in `AppContext.activeVariant`, set by
  `components/predictions/VariantSelector.tsx` and persisted under `locallatin-variant`.
  It drives predictions, feedback drafts and the token highlights. The attribution
  artifacts name the uncorrected variant `baseline` where the prediction CSVs name it
  `raw`; `toAttributionVariant` in `src/api/variants.ts` is the only place that bridges
  the two vocabularies.
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
