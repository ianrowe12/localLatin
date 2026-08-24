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

Per-variant checks (read-only, always run):

- `/api/models` advertises all four variants (`raw`, `abtt`, `sif`, `sif_abtt`) for **every** model, and its `default_variant` is among them. A short list here means a predictions CSV never reached the host.
- `/api/query/{id}/predictions?variant=` returns a non-empty ranked list for each variant and echoes the variant it served.
- The same request with no `?variant=` serves the deployment's configured default.
- `/api/token_map/{id}?method=ig&variant=` returns non-empty `query_tokens` and an `ig` matrix for each of the four attribution variants (`baseline`/`abtt`/`sif`/`sif_abtt` — the artifacts call the uncorrected variant `baseline` where the CSVs call it `raw`).

To verify DB write/read intentionally, add:

```bash
LOCALLATIN_SMOKE_WRITE=1
```

Accepted values are `1`/`true`/`yes`/`on` to enable and `0`/`false`/`no`/`off`/unset to disable; anything else is a hard error. (It is parsed rather than tested for non-emptiness, because `LOCALLATIN_SMOKE_WRITE=0` is the obvious way to say no and a `${VAR:+...}` test would have read it as yes.)

Enabled, the run writes **exactly one row** into the reviewer feedback DB, and that single row carries every write-path assertion: the DB accepts the write, `/api/feedback/export` reads it back, `GET /api/feedback/latest?variant=` prefills both the notes and the multi-select answer (the reviewer's reload path), and the note does not leak into a different variant's prefill.

In GitHub Actions the deploy job passes `LOCALLATIN_SMOKE_USERNAME` / `LOCALLATIN_SMOKE_PASSWORD` from repository secrets. When they are unset, `deploy.sh` skips the authenticated checks instead of failing.
