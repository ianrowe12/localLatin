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
- `services/feedback_db.py` — SQLite CRUD for reviewer feedback and accounts
- `services/rate_limit.py` — in-process sliding-window limiter on `app.state.rate_limiter`, used by
  sign-in and the password endpoints. Per-process, which is correct only because
  `deploy/locallatin.service` runs `--workers 1`; a service restart wipes every open window.
- Passwords: `POST /api/auth/change_password` (self-serve, keeps the calling session and revokes
  the account's others) and `POST /api/auth/accounts/{id}/reset_password` (PI/admin only, returns a
  temporary password once, revokes every session, sets `accounts.must_change_password`). While that
  flag is set, `get_current_user` answers 403 on every route outside
  `dependencies.PASSWORD_CHANGE_EXEMPT_PATHS`, so the forced change is backend-enforced. The one
  authenticated-looking exception is `/api/models`, which has no auth dependency at all (pre-existing)
  and therefore still answers during a forced change.
- Lockout policy (there is no email recovery, so every path must stay recoverable):
  self-reset is refused with 400 — an admin changes their own password, and a second PI/admin resets
  one who is locked out. Sign-in and change-password verify the password *before* consulting the
  rate-limit window and record a hit only on a failed verification, so a correct password always
  gets through and a stranger cannot lock a reviewer out. 429s carry `Retry-After`.
- Sign-in throttle keying: `(client address, username)`, where the address comes from `X-Real-IP`,
  falling back to the **rightmost** `X-Forwarded-For` hop and then the socket peer. `deploy/nginx.conf`
  sets `X-Real-IP $remote_addr` (overwritten per request) but `X-Forwarded-For $proxy_add_x_forwarded_for`,
  which *appends* the peer to whatever the client sent — so XFF's leftmost hops are attacker-chosen
  and only the rightmost is proxy-written. Trusting the left hop would let one attacker rotate a fake
  address per request, never fill a window, and grow the limiter's key space without bound. This is
  safe only because uvicorn binds `127.0.0.1` and is reachable solely through that proxy; exposing it
  directly would make both headers forgeable.
- `routers/` — One file per API domain (queries, predictions, token_map, feedback, stats)
- `models.py` — All Pydantic request/response models

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
