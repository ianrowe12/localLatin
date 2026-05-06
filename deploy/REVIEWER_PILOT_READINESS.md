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
  --base-url http://127.0.0.1:8000 \
  --username <pi-admin-user> \
  --password <pi-admin-password>
```

The smoke script checks static frontend load, SPA refresh, sign-in, `/api/stats`, `/api/models`, `/api/queries`, predictions, token-map artifacts, CSV export, and the PDF packet endpoint.

To verify DB write/read intentionally, add:

```bash
LOCALLATIN_SMOKE_WRITE=1
```

That writes one `legacy_unresolved` smoke feedback row and verifies it through CSV export.
