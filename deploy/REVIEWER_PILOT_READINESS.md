# Reviewer Pilot Deployment Readiness

This deployment is for a protected reviewer pilot, not a public unauthenticated site.

## Access Posture

- Serve the app behind TLS at the configured origin, currently `https://ai.csr.uky.edu`.
- Keep `cors.allow_origins` narrow and aligned with the deployed origin.
- Require LocalLatin app accounts for reviewers and PI/admin users.
- Treat CSV exports and PDF packets as PI/admin-only data-release artifacts.
- If the host is reachable outside UKY, keep it behind an approved protective layer such as institutional SSO, VPN, reverse-proxy auth, or equivalent network controls.

## Production Settings

- `web/config.production.yaml` must use an explicit `paths.data_root`.
- `deploy/locallatin.service` sets `LOCALLATIN_CONFIG` and `LOCALLATIN_DATA_ROOT` explicitly.
- The service uses one Uvicorn worker because the app keeps an in-memory data store and writes SQLite feedback.
- The feedback DB is configured at `data/feedback.db`; the deploy script verifies that `data/` exists and is writable by the service user.
- Backend dependencies come from `web/requirements.txt`; frontend dependencies come from `web/frontend/package-lock.json`.

## Data Payload (gitignored)

`deploy/deploy.sh` only git-pulls, but the webapp reads files that are gitignored and therefore never arrive that way:

| Path | Source |
|---|---|
| `runs/active/resubmit/unlabelled/unlabelled_predictions_{raw,abtt,sif,sif_abtt}.csv` | data release |
| `runs/active/ig_examples/phase12f_examples.csv` | data release |
| `runs/active/ig_examples/artifacts/**/*.npz` | data release |
| `data/canon_unlabelled/`, `data/canon_labelled/` | host-resident, verified only |

The deploy host is reachable only through the self-hosted Actions runner, so the first three travel as a versioned **GitHub Release asset**:

```bash
# On the research machine (repo root), build and publish:
bash scripts/webapp/make_data_release.sh --tag data-YYYYMMDD
gh release create data-YYYYMMDD \
    dist/locallatin-data-YYYYMMDD.tar.gz \
    dist/locallatin-data-YYYYMMDD.tar.gz.sha256 \
    --title data-YYYYMMDD --notes "webapp data payload"

# Point the deploy at it:
gh variable set DATA_RELEASE_TAG --body data-YYYYMMDD
```

On the host, `deploy.sh` then downloads the asset and its `.sha256`, verifies the checksum, and installs it. The stage is opt-in and idempotent:

- No `DATA_RELEASE_TAG` — nothing is downloaded and nothing on disk changes. A host that already carries the payload, and local development, are unaffected.
- A release already installed byte-for-byte is a no-op. The installed state lives in a single file, `.deploy-cache/installed.state`, holding `<tag> <sha256>`. Keying it on the content hash rather than the tag alone is what lets you **roll back to an earlier tag** (the state file names a different release, so it reinstalls) and **re-publish the same tag with new bytes** (the hash differs, so it reinstalls). A per-tag marker would silently skip both.
- A cached tarball whose checksum still matches is not re-downloaded.
- Each file is installed with an atomic rename, so the still-running old service never reads a half-written CSV.

### Failure handling

`sync_data_release` is called from a `|| ...` context, which suppresses `set -e` for everything inside it. Nothing in the function relies on errexit: every command whose failure matters is checked explicitly and turned into a non-zero return. In particular

- `tar`'s exit status is checked, and the extracted file count is reconciled against the archive listing;
- each copy is checked for a short write, and the installed count is reconciled against the same listing;
- `installed.state` is written **only after** both reconciliations pass, so a part-way failure is retried on the next deploy rather than recorded as success;
- staging is pruned on the success and the failure path, including stages left by an earlier crashed run, so the roughly 350 MB extract never accumulates;
- other releases' cached tarballs are pruned after a successful install (they are re-downloadable);
- a `df` preflight refuses to start the extract without roughly 4x the tarball size free.

The failure this chain is built around is ENOSPC part-way through unpacking a 224 MB tarball into a ~350 MB tree.

### Feedback database safety

Every member of the tarball is under `runs/active/`, checked when the archive is built **and** again on the host before extraction. The reviewer feedback DB resolves to `<data_root>/data/feedback.db` = `/homes/ipro222/localLatin/data/feedback.db`, outside that prefix, so the payload structurally cannot reach it. Nothing in `deploy.sh` removes it either (only `web/static/` is deleted). The script fingerprints the DB before and after the sync and aborts if it changed.

The canon text corpora are deliberately *not* in the payload: keeping `data/` out of the archive is what makes the guarantee above structural rather than a convention.

### Fail-fast contract check

After the sync, `deploy.sh` runs `scripts/webapp/export_webapp_data.sh <root> --strict`, which exits non-zero if any required path is absent **or present but empty**: a predictions CSV must be non-zero bytes and carry at least one data row, and `artifacts/` must contain at least one non-empty `.npz`. Existence alone is not a useful check, because what a truncated sync leaves behind is a zero-byte file that `[ -e ]` accepts and the CSV parser then rejects at startup. Without this gate a missing or empty CSV surfaces as an opaque uvicorn traceback behind the health check. Set `DEPLOY_SKIP_DATA_CHECK=1` to bypass.

## Account Roles (issue #265)

An account is a `reviewer` or a `pi_admin`. PI/admins can approve accounts, reset passwords, see `/api/stats` and export all feedback, so a promotion is a data-release decision.

**From the site.** A signed-in PI/admin uses the Roles section of the account panel ("Make PI/admin" / "Make reviewer", then a confirm step). It calls `POST /api/auth/accounts/{id}/role`.

**Guards**, enforced by the server and by the CLI below:

- nobody changes their own role (400 in the app); a second PI/admin does it, as with password resets;
- a change that would leave no approved, active PI/admin is refused (409), because there is no email recovery;
- an unknown account is 404, and setting the role it already has writes nothing.

The new role applies on the account's next request. Sessions are not revoked, because the app reads the role from the database on every request.

**Audit.** Every change adds a row to `account_role_changes` in the feedback DB: `account_id`, `old_role`, `new_role`, `changed_by_account_id` (NULL when the CLI made it), `note` (prefixed `cli: ` from the CLI) and `changed_at`. The app creates the table on boot and the CLI creates it if it is missing; existing rows are never touched.

**From GitHub, on the production DB.** When no PI/admin can do it in the app, run the `Admin set account role` workflow (`.github/workflows/admin-set-account-role.yml`). It runs `scripts/webapp/set_account_role.py` on the self-hosted production runner against `$DEPLOY_PATH/data/feedback.db`, in the deploy concurrency group, only when `ENABLE_PRODUCTION_DEPLOY` is `true`. The script comes from a fresh checkout of `main` in the runner's workspace, so the deploy checkout is never modified. Without `apply` it only reads.

```bash
# 1. Find the account (prints id, username, display name, role, approval status, created_at)
gh workflow run admin-set-account-role.yml -f find=noah
# 2. Dry run: what would change
gh workflow run admin-set-account-role.yml -f username=<login> -f role=pi_admin \
    -f approve_if_pending=true -f note="issue #265"
# 3. Apply: backs the DB up first, then makes the change in one transaction
gh workflow run admin-set-account-role.yml -f username=<login> -f role=pi_admin \
    -f approve_if_pending=true -f note="issue #265" -f apply=true
```

With `apply`, the script first writes `data/feedback.db.pre-role-change-<UTC timestamp>` with SQLite's online backup API (mode 0600, integrity-checked) and prints its path and size. That file is a full copy of the reviewer DB, password hashes included: keep the most recent and remove older ones only once another good copy exists (`scripts/webapp/backup_feedback_db.py`). `approve_if_pending` also approves a pending account; without it a pending account keeps its pending status.

## Smoke Checks

After deployment, run:

```bash
LOCALLATIN_SMOKE_USERNAME=<pi-admin-user> \
LOCALLATIN_SMOKE_PASSWORD=<pi-admin-password> \
bash deploy/deploy.sh
```

The deploy script always fails if `/api/models` does not become healthy. With credentials, it also runs:

```bash
python3 scripts/webapp/smoke_reviewer_pilot.py \
  --base-url http://127.0.0.1:8080 \
  --username <pi-admin-user> \
  --password <pi-admin-password>
```

The smoke script checks static frontend load, SPA refresh, sign-in, `/api/stats`, `/api/models`, `/api/queries`, predictions, token-map artifacts, CSV export, and the PDF packet endpoint.

### Python version

The webapp requires **Python 3.11 or newer** and production runs 3.12
(`deploy/deploy.sh` builds the venv with `PYTHON_BIN`). `web/models.py` uses
`enum.StrEnum`, which arrived in 3.11, so on 3.10 the package cannot be imported
at all: `pytest web/tests` fails at collection, and the repo's HPC conda env
(`localLatin`, 3.10) cannot run the webapp suite. That is why CI's 3.10 matrix
leg runs `tests/` only and the 3.12 leg runs `tests web/tests` — the split is
deliberate, not an oversight, and anyone reproducing a webapp test locally needs
a 3.11+ interpreter.

Reviewer-directory checks (read-only, always run, issue #196):

- `/api/query/{id}/predictions` carries a `reviewer_dir_candidates` list. Its absence means the host is running a backend from before the split, which would still serve reviewer directories as ranked candidates 11-15.
- No entry in that list carries a `rank`, and every entry in `predictions` is a model candidate ranked 1-10. A reviewer directory numbered alongside the model's answers is exactly the confusion this release removed, so a deploy that reintroduces it fails the smoke run rather than reaching an evaluator.
- `seeded_dirs` is still present and well-typed (issue #95).

All three pass on a deployment with no reviewer directories at all, which is what a fresh host looks like.

Since issue #221 the reviewer-facing page does not DRAW the reviewer-directory
block at all (`PredictionList`'s `showReviewerDirectories`, default false), at
Prof. Firey's request. These checks are deliberately unchanged: they are about
what the API serves and how it is shaped, and the data is still served, still
carried by the frontend's shared state, and one prop away from being displayed
again. A backend that went back to numbering directories among the model's ten
would still be wrong, whether or not anything on the page shows it.

Per-variant checks (read-only, always run):

- `/api/models` advertises all four variants (`raw`, `abtt`, `sif`, `sif_abtt`) for **every** model, and its `default_variant` is among them. A short list here means a predictions CSV never reached the host.
- `/api/query/{id}/predictions?variant=` returns a non-empty ranked list for each variant and echoes the variant it served.
- The same request with no `?variant=` serves the deployment's configured default.
- `/api/token_map/{id}?method=ig&variant=` returns non-empty `query_tokens` and an `ig` matrix for each of the four attribution variants (`baseline`/`abtt`/`sif`/`sif_abtt` — the artifacts call the uncorrected variant `baseline` where the CSVs call it `raw`).
- `/api/token_map/{id}?method=ig&variant=` for the default variant reports `variant_served == variant_requested`, and its `query_words` contain at least one word built from more than one model piece. The check scans the gallery examples in order and passes on the first that shows both, so reordering the examples CSV cannot fail a deploy; it fails only when none of the first ten does. The first is the highlight following the reviewer's pipeline rather than a fixed preference order (issue #216); the second is the word-level grouping actually running on the deployed artifacts (issue #211). A failure here means the highlights are back to one outline per subword piece, or are describing another pipeline's evidence.

To verify DB write/read intentionally, add:

```bash
LOCALLATIN_SMOKE_WRITE=1
```

Accepted values are `1`/`true`/`yes`/`on` to enable and `0`/`false`/`no`/`off`/unset to disable; anything else is a hard error. (It is parsed rather than tested for non-emptiness, because `LOCALLATIN_SMOKE_WRITE=0` is the obvious way to say no and a `${VAR:+...}` test would have read it as yes.)

Enabled, the run writes **exactly one row** into the reviewer feedback DB, and that single row carries every write-path assertion: the DB accepts the write, `/api/feedback/export` reads it back, `GET /api/feedback/latest?variant=` prefills both the notes and the multi-select answer (the reviewer's reload path), and the note does not leak into a different variant's prefill.

In GitHub Actions the deploy job passes `LOCALLATIN_SMOKE_USERNAME` / `LOCALLATIN_SMOKE_PASSWORD` from repository secrets. When they are unset, `deploy.sh` skips the authenticated checks instead of failing.

The production config sets `auth.secure_cookies: true`, so the session cookie is marked `Secure`, while the checks above run against `http://127.0.0.1:8080` on purpose: they exercise the service itself, not nginx and DNS in front of it. `smoke_reviewer_pilot.py` therefore returns `Secure` cookies to loopback origins only (`LoopbackSecureCookiePolicy`); a stock cookie jar silently drops them and every request after sign-in goes out anonymous, which is how the first authenticated deploy run failed with `GET /api/stats returned 401` (issue #126). Pointing `--base-url` at the public site keeps the standard rule and still refuses to send a session cookie over http.
