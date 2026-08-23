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
- An already-installed tag is a no-op (marker at `.deploy-cache/installed-<tag>`).
- A cached tarball whose checksum still matches is not re-downloaded.
- Each file is installed with an atomic rename, so the still-running old service never reads a half-written CSV.

### Feedback database safety

Every member of the tarball is under `runs/active/`, checked when the archive is built **and** again on the host before extraction. The reviewer feedback DB resolves to `<data_root>/data/feedback.db` = `/homes/ipro222/localLatin/data/feedback.db`, outside that prefix, so the payload structurally cannot reach it. Nothing in `deploy.sh` removes it either (only `web/static/` is deleted). The script fingerprints the DB before and after the sync and aborts if it changed.

The canon text corpora are deliberately *not* in the payload: keeping `data/` out of the archive is what makes the guarantee above structural rather than a convention.

### Fail-fast contract check

After the sync, `deploy.sh` runs `scripts/webapp/export_webapp_data.sh <root> --strict`, which exits non-zero if any required path is absent. Without it a missing CSV surfaces as an opaque uvicorn `FileNotFoundError` behind the health check. Set `DEPLOY_SKIP_DATA_CHECK=1` to bypass.

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

That writes one `legacy_unresolved` smoke feedback row and verifies it through CSV export, then round-trips notes and a multi-select answer through `POST /api/feedback` + `GET /api/feedback/latest?variant=` for two variants — the reviewer's reload path — and checks that a note saved under one variant does not leak into another variant's prefill.

In GitHub Actions the deploy job passes `LOCALLATIN_SMOKE_USERNAME` / `LOCALLATIN_SMOKE_PASSWORD` from repository secrets. When they are unset, `deploy.sh` skips the authenticated checks instead of failing.
