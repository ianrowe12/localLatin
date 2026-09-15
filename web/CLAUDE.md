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
- `services/feedback_db.py` — SQLite CRUD for reviewer feedback and accounts, plus the
  append-only `reviewer_dirs` / `reviewer_dir_members` tables (issue #95), created in
  `_migrate()` with `CREATE TABLE IF NOT EXISTS` so a pre-#95 database gains them without
  any existing row being touched. The feedback log is
  **append-only**: never UPDATE or DELETE a feedback row. `web/tests/test_top10_feedback.py`
  installs SQLite triggers that enforce this during the save path; they are test-only and must not
  move into `_SCHEMA`, because two of `_migrate()`'s normalization UPDATEs re-fire on every boot.
- Shared notes (issue #96): `GET /api/feedback/latest` returns a **merged prefill view**, not a
  stored row. The note, its timestamp and its attribution (`reviewer`, `reviewer_username`, which
  usually name somebody else) come from the newest row by ANY reviewer that carries a non-empty
  note, so an answer saved without prose cannot blank a colleague's reasoning. The decision
  (`outcome`, `correct_rank`, `correct_dir`, `selected_ranks`) always comes from the CALLER's own
  newest row, so a rank pressed by somebody else is never prefilled. When nobody has written a note
  at all, the caller's own row carries the response so their selection is still restored.
- `services/qq_matrix.py` — reads `qq_sim_<slug>.npz` and answers "how similar is this
  query to this set of member queries" (max over members, self excluded).
- `services/reviewer_dirs.py` — scoring and shaping of reviewer directories, shared by the
  predictions, reviewer_dirs, feedback and packets routers. Status is **derived**, never
  stored: a directory is `matched` once a human has filed a second document into it
  (`len(members) > 1`), never from similarity alone.
- `bands.py` — the 0.5 / 0.7 confidence thresholds. **The backend owns these numbers**;
  `GET /api/models` serves them as `confidence_bands`, and the frontend's
  `utils/confidenceBands.ts` reads them via `bandsFrom()` rather than keeping a second
  copy. That file still owns the band function, copy and styling built on top of them.
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
- `routers/` — One file per API domain (queries, predictions, token_map, feedback,
  reviewer_dirs, packets, stats)
- `models.py` — All Pydantic request/response models

## Reviewer directories (issue #95)

`POST /api/reviewer_dirs {query_file_id, label?}` -> 201 `ReviewerDir`. Any signed-in, approved
reviewer may call it; there is no extra role gate, since reviewers are exactly who the feature
is for. The 201 always reports `awaiting_match`.

Creation is refused with 409 (the query already seeds one), 422 (the seed is a guard-excluded
document, so it could never be matched), 429 (`MAX_REVIEWER_DIRS_PER_ACCOUNT`) or 400 (unknown
model). All of these exist because **nothing can ever remove a directory** — both tables are
append-only — so a permanent artefact must not be creatable by a double-click or a typo.

Created directories are served in their own **unranked** field,
`PredictionResponse.reviewer_dir_candidates`, scored live from the q-q matrix (max over member
documents) and capped at `MAX_REVIEWER_CANDIDATES` best-first. They are NOT in `predictions`
(issue #196): they used to be appended there at ranks anchored on `MAX_MODEL_RANK + 1`, which
evaluators read as the model predicting them, and a rank computed live changed meaning as
memberships changed. Model ranks are untouched, and every rank in a response is now one of the
retrieval run's own.

A query joins a reviewer directory by **naming its CCL key** beside "None of the top N", not by
pressing a rank: `POST /api/feedback` takes an optional `ccl_key` with `outcome: none_of_top_k`
and `FeedbackDB.insert_none_of_top_k` writes the assessment and the directory action in one
`BEGIN IMMEDIATE` transaction. The four branches (`matched_labelled_dir`, `joined_reviewer_dir`,
`created_reviewer_dir`, `seed_taken`) are recorded on the feedback row in `ccl_key_action` /
`ccl_key_dir`; `correct_dir` keeps its old meaning and is never written by this path.
`services/ccl_keys.py` owns the normalisation (stored as typed, matched case-folded).

**`correct_dir` is always resolved server-side from `correct_rank`**, never read from the
request body, and a rank with no candidate behind it is a 422. That is both the anti-spoof
guard and the real rank validation — `MAX_CANDIDATE_RANK` is only an outer bound.

Confirming a reviewer directory as the correct answer appends that query to the directory's
members (idempotent, `INSERT OR IGNORE`, and the directory must exist). That membership row is
what flips the badge to `matched`, which is why it may only follow a server-resolved choice.

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
- Post-processing variant: fixed at `DEFAULT_VARIANT` (`sif_abtt`) in
  `AppContext.activeVariant`. The reviewer-facing picker was removed for every role in
  issue #94, so predictions, candidate texts, feedback drafts and token highlights all
  come from one pipeline. The backend still serves all four variants and feedback rows
  still carry a variant column. The attribution artifacts name the uncorrected variant
  `baseline` where the prediction CSVs name it `raw`; `toAttributionVariant` in
  `src/api/variants.ts` is the only place that bridges the two vocabularies.
- Confidence bands: `src/utils/confidenceBands.ts` owns the band function, the copy and the
  styling, but **not** the thresholds — those come from `GET /api/models`
  (`confidence_bands`) via `bandsFrom()`, because the backend applies the same numbers to
  reviewer directories. What they decide there is `has_potential_match` (an unconfirmed
  lead), **not** status: `status_of` returns `matched` only once a human has filed a second,
  distinct witness. The literals in that file are the pre-flight fallback only.
  `PredictionList` resolves them once and passes them to each `PredictionCard`.
  Below the no-match band the list renders `NoMatchCallout` (issue #94's red `role="alert"`
  framing). Since issue #196 it is a HINT and nothing else: the red
  "New directory / New file" button, its caption and the naming form are retired, at
  Prof. Firey's request, and no control in the prediction panel writes anything.
  `SavedDirectoryNotice` reports a directory the document already seeds, read-only.
- Directory guidance (issue #162, trimmed by #196): every reviewer-facing sentence about
  provisional directories and about the CCL key lives in
  `src/utils/reviewerDirectoryCopy.ts` (`DIRECTORY_CREATION_COPY`, `CCL_KEY_COPY`), so
  wording and behaviour change independently. Two facts must survive any rewrite: a
  grouping is a permanent write with no rename or removal, and a saved directory is offered
  only where the model can score it — never "every other document". The key field is
  optional and must never ask a general evaluator to search the CCL by hand.
- The assessment panel (issue #196) holds the ten large rank buttons (`MatchPills`) and the
  blue `NoneOfTopTenAction`, which carries one optional "CCL key of the source, if known"
  field and its OWN submit, posting through `src/api/cclKey.ts`. The panel's Save button is
  withheld while None is held, so one decision cannot be written twice.
- Candidate provenance (issue #162): `src/utils/documentProvenance.ts` maps a candidate to
  `labeled_reference` (a witness the labelled corpus already groups) or `reviewer_group` (an
  originally unlabeled witness in a provisional reviewer directory), and
  `DocumentPanel`/`DocumentHeader` caption the panel from it. `provenanceOf` prefers the
  prediction's `source`, falls back to the backend's `reviewer-dir-` prefix rule for an
  off-list candidate, and otherwise stays `unknown`: an unidentified candidate must never be
  captioned as a labelled reference. `CenterArea` derives the prop by the same override-wins
  rule as `candidateDir`, so a gallery example is captioned from its own directory and never
  from the rank selected underneath it, and a ranked candidate is captioned only from a
  prediction the shared state (issue #156) has already accepted as current.
- Default model: `DEFAULT_MODEL_SLUG` in `src/api/models.ts` (`google_mt5-base`, displayed
  as "mT5-base" by `services/data_store.py`), with a fallback to the first served model.
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
