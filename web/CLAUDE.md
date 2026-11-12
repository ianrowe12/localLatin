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
- Roles (issue #265): `POST /api/auth/accounts/{id}/role {role, note?}` (PI/admin only) calls
  `FeedbackDB.set_account_role` and returns the updated `AccountPublic`. Guards: your own role is
  400 (same rule as self-reset), a change that would leave no approved, active PI/admin is 409
  (`LastPiAdminError`; pending, rejected and deactivated admins do not count), unknown account
  404, an unchanged role a 200 that writes nothing. Each change writes one row to
  `account_role_changes` (`account_id, old_role, new_role, changed_by_account_id, note,
  changed_at`), created in `_migrate()` from `ACCOUNT_ROLE_CHANGES_DDL`. No session is revoked:
  `get_account_by_session` re-reads the account row on every request, so the new role applies to
  the account's next request. `scripts/webapp/set_account_role.py` (stdlib, deploy host) writes
  the same audit row with `changed_by_account_id` NULL and a `cli: ` note, and carries a copy of
  the DDL that `tests/test_set_account_role.py` keeps byte-identical. It selects by
  `--username` or `--account-id`; `.github/workflows/admin-set-account-role.yml` always runs it
  with `--redact` (masked identifiers only) and `--find ... --max-matches=5`, because the repo is
  public and its Actions logs are world-readable. `AccountApprovalPanel`'s Roles section is the
  UI: a confirm step, disabled on your own row.
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
`BEGIN IMMEDIATE` transaction. The six branches (`matched_labelled_dir`, `joined_reviewer_dir`,
`already_joined`, `created_reviewer_dir`, `seed_taken`) are recorded on the feedback row in
`ccl_key_action` / `ccl_key_dir` / `ccl_key_rank`; `correct_dir` keeps its old meaning and is never
written by this path. `services/ccl_keys.py` owns the normalisation (stored as typed, matched
case-folded), and a directory is reached by key first, then by label -- the label is how a
pre-#196 directory (empty `ccl_key`) stays joinable at all. An identical repeat of the caller's own
newest row returns it with a 200 and appends nothing; the log is still append-only.

**`correct_dir` is always resolved server-side from `correct_rank`**, never read from the
request body, and a rank with no candidate behind it is a 422. That is both the anti-spoof
guard and the real rank validation — `MAX_CANDIDATE_RANK` is only an outer bound.

Confirming a reviewer directory as the correct answer appends that query to the directory's
members (idempotent, `INSERT OR IGNORE`, and the directory must exist). That membership row is
what flips the badge to `matched`, which is why it may only follow a server-resolved choice.

## CCL key assignments export (issue #270)

`GET /api/feedback/export/ccl_keys` (PI/admin only, `require_pi_admin`) is the file CCL's import
is built on. It returns a CSV named `ccl_key_assignments_YYYY-MM-DD.csv` with one row per reviewed
unlabelled query, sorted by `xml_id`. `/api/feedback/export` is unchanged and stays the raw log.
The optional `status=agreed|conflict|no_key|needs_check` keeps one status; the default is every reviewed query.
The dashboard's "Download CCL key assignments" button (`ExportControls`) opens it without filters.
Like the raw export it is a `window.open`, so the mock fetch handler never sees it.

Resolution rule, implemented in `services/ccl_key_export.py` and read by
`FeedbackDB.ccl_key_assignments`:

1. Only `matched_rank` and `none_of_top_k` rows count, the same pair that makes a query
   `reviewed` on the dashboard, so the export covers exactly the reviewed queries. Skipped and
   legacy-unresolved rows are ignored, and a later skip does not retract an earlier decision.
2. A reviewer's current decision is their newest decision row on each (model, variant), newest by
   timestamp and then row id. That is the decision half of `GET /api/feedback/latest`. A save
   under a different model does not replace one under another model, because the app still shows
   the reviewer both.
3. Each current decision is an answer. A `matched_rank` row names the directory behind every
   selected rank: a labelled directory names itself and a reviewer directory names its `ccl_key`. A
   `none_of_top_k` row names its typed `ccl_key`; one without a key is the answer "none of these"
   and rejects every candidate it was made against. It is a dissent, not an abstention.
4. A pick is used only when it can be shown to be what the reviewer saw. A row saved since #273
   stores the directory behind every selected rank in `selected_dirs_json`, resolved server-side
   from the snapshot the reviewer saw, and every pick is trusted as stored; no ranking is read.
   The rest of this point applies only to older rows. A single pick is trusted
   as stored in `correct_dir`, which the server resolved at save time. `correct_dir` holds only the
   first selection, so the later selections of a multi-rank pick are read from the ranking the
   server serves now for that model and variant, and only after that ranking is shown to still hold
   the stored first pick at its rank. A pick is unresolved when the anchor moved, the rank or the
   query is no longer served, the row needs the ranking but has no variant, the row stores no
   directory at all, the rank is above 10 (a legacy reviewer-directory pick from before #196), or it
   names a reviewer directory that has no key. An unresolved pick never supplies a key.
5. Keys are compared with `ccl_keys.match_form`. The status is the first of these that applies:
   `needs_check` when any counted decision holds an unresolved pick (a human must look before
   importing); `conflict` when two or more different keys are named (by two reviewers, one reviewer
   under two models, or one multi-rank pick) or when a key is named and another counted decision is
   a no-key "none of these"; `agreed` when exactly one key is named and every counted decision
   names it; `no_key` when every counted decision is a no-key "none of these".
6. Each reviewer counts once, under the display name on their newest counted decision. Rows from
   before accounts existed are told apart by that name. `reviewers` lists everyone with a counted
   decision, whatever they answered, so in a `conflict` or `needs_check` row it includes people who
   rejected a key. Only an `agreed` row has a `ccl_key`, and there every listed reviewer named it, so
   nobody is credited with a key they rejected.

When reviewers typed one key in different capitalisations, the printed form is the labelled
directory's own name if the key names one; otherwise the earliest form that is key-shaped
(`ctou.567.16` then `CTOU.567.16` prints `CTOU.567.16`); otherwise the form on the earliest decision
that named it. Every candidate is a form somebody stored; nothing is re-cased. Rows are grouped
by the `xml_id` stored on them (#273); a row without one is named through the served corpus, and
one that neither names is left out (logged as a count), because it has no `xml_id`.

Columns (`EXPORT_SCHEMA_VERSION = 1`; bump it when a column is added, removed or changes meaning):

| column | value |
|---|---|
| `xml_id` | the query filename without `.txt`, which is the CCL `xml:id` (`scripts/data/tei_to_canon.py`) |
| `ccl_key` | the agreed key; empty for every other status |
| `status` | `agreed`, `conflict`, `no_key` or `needs_check` (a human must look before importing) |
| `conflicting_keys` | for `conflict` and `needs_check`, every distinct key, sorted, then `(none_of_top_k)` if anyone answered None without a key, then `(unresolved)` if any pick is unresolved, joined by ` \| ` |
| `key_source` | `model_suggestion`, `reviewer_typed` or `both`, over the decisions that named a key |
| `suggested_by` | `model:variant:rank` per picked ranked candidate, joined by ` \| `; in `conflict` and `needs_check` rows each entry is prefixed `KEY=`, and an unresolved pick is listed as `(unresolved)=model:variant:rank` |
| `key_shaped` | `true` or `false` by `ccl_keys.KEY_SHAPED`, a flag and never a filter; empty without an agreed key |
| `reviewer_count`, `reviewers` | people with a counted decision, and their display names joined by ` \| ` |
| `first_decided_at`, `last_decided_at` | earliest and latest counted decision, ISO 8601 UTC |
| `export_schema_version`, `exported_at` | `1`, and the export time in ISO 8601 UTC |
| `data_release` | the release tag `deploy/deploy.sh` recorded in `.deploy-cache/installed.state` under `data_root` (`paths.data_release_state`), read at startup; empty when there is none, as in local development |

Lists are joined by ` | ` because 18 labelled directory names contain a semicolon and 103 a comma,
and none contains `|`. Display names appear only in this download, never in a log line. Usernames
and emails never appear.

### Typed text that reaches the export (issue #274)

Keys, labels and display names are exported verbatim, because escaping would change the value
CCL's import reads. So they are refused at INPUT instead: the typed `ccl_key` on
`POST /api/feedback`, `label` on `POST /api/reviewer_dirs`, and `display_name` on
`POST /api/auth/register` and `POST /api/auth/accounts` (the app has no other display-name edit
path; `scripts/webapp/provision_smoke_account.py` carries a literal copy of the rule). A value is
refused when, after leading whitespace and control characters (tab, CR, LF, BOM), it starts with
`=`, `+`, `-` or `@`, or when it contains `|` anywhere. A bare `|` is refused too, not only ` | `:
a value ending in ` |` or starting with `| ` corrupts the ` | `-joined list just the same, and no
labelled directory name contains one (`tests/test_input_safety.py` checks every name under
`data/canon_labelled/`). The rule lives in `services/input_safety.py` (pydantic field validators
in `models.py`), and `app.py` turns its failure into a 422 `{error: {code: "UNSAFE_TEXT",
message}}` naming the field, never echoing the value; other validation errors keep FastAPI's
shape. The frontend mirror is `frontend/src/utils/inputSafety.ts`, shown inline under the field
with the submit disabled. Both sides test against `frontend/src/utils/inputSafetyCases.json`,
which also holds the exact messages. Stored values are never re-checked or rewritten.

## Witness ids (issue #273)

`query_id` is a position in the sorted unlabelled corpus, so a release that adds or removes one
query file would shift every later review onto another witness. Every table that references a
query therefore also stores the witness `xml_id` (filename stem): `feedback.xml_id`,
`reviewer_dirs.seed_xml_id`, `reviewer_dir_members.xml_id`. Feedback rows also store
`selected_dirs_json`, the directory behind each selected rank as the reviewer saw it.

- Migration (`_migrate`): nullable columns only. Before its first write to a populated pre-#273
  DB, `_backup_before_xml_id_migration` writes `<db>.pre-xml-id-<UTC stamp>` beside it (online
  backup API, 0600, integrity-checked), once per database. It holds password hashes.
- `FeedbackDB.adopt_corpus(store.file_id_to_filename, data_release=..., ...)` runs in the app
  lifespan after `connect()`. When the corpus is unchanged it BACKFILLS NULL witness ids from the
  served corpus, writing nothing else; with a `data_release` set the corpus must hash
  (`corpus_digest`) to `PRE_XML_ID_CORPUS_SHA256`, the listing every pre-#273 row was saved
  against, or startup refuses (`CorpusMismatchError`). One `xml_id_backfills` audit row records
  what was filled. Once every row has an xml_id this check never runs again, so it does not block
  a later corpus change. When the corpus changed, see "Corpus changes (issue #278)" below. Either
  way it then BINDS the corpus: later writes record the witness, and every read goes by `xml_id`.

## Corpus changes (issue #278)

A data release may add, remove or reorder unlabelled query files (new CCL material goes into the
webapp's query corpus, never into `data/canon_labelled/`, which is benchmark v1). Every stored
`query_id` is then a position in a listing that no longer exists. Nothing rewrites it: the
tables are append-only, and the stored `xml_id` says which witness a row is about.

**Detection.** `adopt_corpus` compares the served corpus digest with the one the DB was last
bound to (newest `corpus_changes` row, else newest `xml_id_backfills` row). A DB that never
recorded one saved every keyed row against the corpus it last served, so there a change means a
stored witness now sits at another id or is gone. The production DB has its #273 backfill row,
so an unchanged release is detected as unchanged and nothing is written or backed up.

**On a change, in order** (each refusal raises before anything is written and the app does
not start):

1. Any stored row with a NULL xml_id refuses (`UnkeyedRowsError`): its stored id is the only
   thing naming its witness, and the new corpus may give that id to another text. Recover by
   serving the release those rows were saved against once (the backfill fills them).
2. If more than `corpus_change.max_missing_reviewed_share` (default `0.05`) of the reviewed
   witnesses the previous corpus served are missing, it refuses (`MissingWitnessThresholdError`),
   because that is usually a broken release. "Reviewed" means a `matched_rank` or
   `none_of_top_k` row, a reviewer-directory seed or a membership. The error prints the new corpus
   digest; to accept an intended removal set `corpus_change.accept_missing_reviewed_sha256` to
   that digest (in `config.production.yaml`, or the env var
   `LOCALLATIN_ACCEPT_MISSING_REVIEWED_SHA256`). It applies to that exact corpus only, so leaving
   it set cannot wave a later broken release through.
3. Backup `<db>.pre-corpus-change-<UTC stamp>` (`_backup_database`, the #273 pattern: online
   backup API, 0600, O_EXCL, integrity-checked; a failed copy is deleted and startup stops). It
   holds every password hash and is gitignored.
4. One transaction: rebuild `reviewer_dir_members` without its positional
   `UNIQUE (dir_id, query_id)` (first change only; see below), one `corpus_changes` audit row
   (old and new digest and size, `data_release`, backup name, counts of unchanged, moved,
   missing, newly missing and restored witnesses, reviewed witnesses and how many of them went
   missing, whether the override was used, whether the members table was rebuilt), and one
   `corpus_change_witnesses` row per stored witness (`xml_id`, `old_query_id`,
   `new_query_id`, NULL when not served). That id map is how the next change knows each
   witness's previous position, because the rows keep the id they were saved at. Any failure
   rolls all of it back; the backup stays.

**Reads** translate a stored `xml_id` through the bound corpus (`_served_query_id`), never the
stored id: statuses, stats, the review queue, "already reviewed", `get_latest_feedback` and
`get_feedback_for_query` (whose returned `query_id` is the SERVED id; the DB row keeps its own),
the identical-repeat check, reviewer-directory seeds, members and q-q scoring, both exports.
`_match_query` falls back to the stored id only for a row with no xml_id, which can only have
been saved against the corpus served now (a changed corpus refuses while one exists).

**Membership** is unique by witness: `idx_reviewer_dir_members_witness` (a partial unique index
on `(dir_id, xml_id)`) plus a by-witness check before every insert. The old positional
`UNIQUE (dir_id, query_id)` is a table constraint SQLite cannot drop, and after a shift it would
refuse a witness that now sits at an old member's id and let a moved witness join twice. The
first corpus change therefore rebuilds the table after the backup, inside the change
transaction: same columns in the same order (any other shape, or a trigger on the table,
refuses), every row copied with its id and checked equal before the original is dropped, the
AUTOINCREMENT counter carried over, indexes recreated. Before any change the constraint and the
witness rule agree, so it is harmless there.

**Witnesses missing from the served corpus** keep every row. They are hidden from the review
queue, statuses, recent activity and needs-attention, and from `reviewed_count` /
`skipped_count`; `/api/stats` counts them in `unserved_witness_count` and
`unserved_reviewed_count`, and the dashboard shows a line when there are any. A reviewer
directory whose seed is gone reports `seed_query_id = -1` (`UNSERVED_QUERY_ID`, never a real id;
the DB record also carries `seed_served: false`, the API model only the -1), and its member list
holds only served members; a directory stays `matched` when a member witness leaves, because
`status_of` counts stored member witnesses (`member_witness_count`), served or not. They stay in
the CCL key export unchanged, since the key assignment for that `xml_id` is still valid for CCL;
no column marks them, so `EXPORT_SCHEMA_VERSION` stays 1 (an older row that needs the served
ranking for a later pick becomes `needs_check`, as for any unserved query). The raw export keeps
`query_id` as saved and appends `served_query_id` (empty when not served).

**Release consistency.** Prediction CSVs and q-q matrices address queries by position, so
`build_store` calls `verify_release_matches_corpus` before anything else is served and raises
`ReleaseCorpusMismatchError` (the app does not start) unless: the served map (default variant,
first model) has unique filenames that are exactly the `*.txt` names in `canon_unlabelled/`;
every model of every configured variant's CSV has the same `file_id -> filename` map (the
non-default variants read only those columns at startup, and a lazy load re-checks); and every
discovered `qq_sim_<slug>.npz` has those file ids and an n x n `sim` (only the header is read). A
CSV or matrix that cannot be read at all keeps the old rule: that variant or matrix is not
served. A CCL import therefore has to ship the new `canon_unlabelled/` listing and a release
rebuilt from it (`scripts/resubmit/index_unlabelled.py`, then predictions and
`build_qq_matrices.py`) together.

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
  THE RECEIPT OUTLIVES THE FORM: on success the panel closes the form and the component keeps
  the sentence naming the branch, and on a revisit it rebuilds that sentence from
  `/api/feedback/latest` (`recordedKeyAnswer`) with a "Change this answer" action. A saved
  `none_of_top_k` row therefore does NOT prefill the None selection -- `draftFromEntry` returns
  `correctRank: null` -- because re-opening a live Record button over an answer already in an
  append-only log is how a document gets two. `NoneOfTopTenPanel.test.tsx` drives the real panel
  wiring; testing the component with `open` hard-coded is what hid this the first time.
- The way OUT of a recorded None answer (issue #221): once this reviewer has one for the
  document on screen (just recorded, or prefilled by `recordedKeyAnswer`) and no pending
  selection, and the blue form is closed, the bottom button becomes a plain **"Next"** that
  runs `advanceToNextActionable` and **posts nothing**. Before it, the withheld button had no
  successor at all and the reviewer was stranded on an answered document. The rule is stated
  on the caller's own recorded answer because Next is not a write: the worst a wrong answer
  can do is offer a move nobody needed. Recording never auto-advances (the receipt is the
  only place the key's outcome is stated), a pending selection keeps the ordinary
  "Submit & Next", and with the form open the button stays disabled beside a one-line
  `assessment-primary-blocked` explanation. `AssessmentNextAfterNone.test.tsx` pins label,
  move and POST count together, since a "Next" that advanced by saving would pass the first two.
- `PredictionList`'s reviewer-directory block ("Directories created by reviewers") is behind
  `showReviewerDirectories`, **default false** since issue #221, at Prof. Firey's request:
  evaluators read it as something to act on and there is no act to perform in it. Rendering
  only. `reviewer_dir_candidates`, the shared state and every type are untouched, so the
  block comes back by flipping that default.
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
