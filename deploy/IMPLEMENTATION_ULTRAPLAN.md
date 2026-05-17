# LocalLatin Deployment Ultraplan

This plan covers the implementation path for production hosting, automated
deployment after pushes, and the test-driven workflow for future changes to the
LocalLatin reviewer webapp.

## Success Criteria

- Backend is hosted as a FastAPI service on the target VM, bound to
  `127.0.0.1:8080`, supervised by user `systemd`, and configured with explicit
  production paths.
- Frontend is built by Vite and served as static files from `web/static` behind
  nginx at `https://ai.csr.uky.edu`.
- GitHub Actions runs backend tests and frontend build checks on every push to
  `main`.
- GitHub Actions deploys after the checks pass on `main` only when the
  `ENABLE_PRODUCTION_DEPLOY` repository variable is set to `true`.
- Deployment is repeatable with `bash deploy/deploy.sh` and fails if the API
  health check does not pass.
- Every behavior change starts with a failing or updated test, then code, then
  local and CI verification.

## Hosting Architecture

### Backend Hosting

- Runtime: Python 3.10+ with `web/requirements.txt`.
- Process manager: user-level `systemd` unit at `deploy/locallatin.service`.
- Entry point:
  `uvicorn web.app:create_app --factory --host 127.0.0.1 --port 8080 --workers 1`.
- Config:
  `LOCALLATIN_CONFIG=/homes/ipro222/localLatin/web/config.production.yaml`.
- Data root:
  `LOCALLATIN_DATA_ROOT=/homes/ipro222/localLatin`.
- Feedback persistence:
  `data/feedback.db`, writable by the service user.
- Health gate:
  `deploy/deploy.sh` polls `http://127.0.0.1:8080/api/models`.

### Frontend Hosting

- Runtime: static Vite build, no Node process in production.
- Build command: `cd web/frontend && npm ci && npm run build`.
- Artifact path: `web/frontend/dist`.
- Deployed path: `web/static`.
- Web server: nginx using the active `ai.csr.uky.edu` virtual host.
- Routing:
  `/` proxies to FastAPI, which serves `/api/*` and the built SPA.
- Cache policy:
  Vite-fingerprinted assets under `/assets/` use long immutable caching.

## GitHub Actions Deployment

The workflow in `.github/workflows/deploy.yml` is the production gate.

Required repository setting:

- Repository variable `ENABLE_PRODUCTION_DEPLOY`: must be `true` before the
  self-hosted production deploy job can run.

Optional repository settings:

- Repository variable `DEPLOY_PATH`: local repo path on the self-hosted runner.
  Defaults to `/homes/ipro222/localLatin`.
- Repository variable `PUBLIC_BASE_PATH`: public route prefix. Defaults to `/`.

Push-to-production flow:

1. A push lands on `main`.
2. The `test` job installs Python dependencies and runs `python -m pytest web/tests`.
3. The `test` job installs frontend dependencies and runs `npm run build`.
4. If `ENABLE_PRODUCTION_DEPLOY=true`, the self-hosted runner on the VM starts
   the `deploy` job.
5. The VM fetches `origin/main`, fast-forwards the checked out repo, and runs
   `PUBLIC_BASE_PATH=/ bash deploy/deploy.sh`.
6. The deploy script installs dependencies, builds the frontend, refreshes
   `web/static`, restarts `locallatin.service`, and verifies `/api/models`.

Deployment will not run if tests or the frontend build fail, and it remains
disabled while `ENABLE_PRODUCTION_DEPLOY` is unset or not `true`.

## Test Driven Development Plan

Use this loop for every implementation slice:

1. Define the user-visible behavior or deployment invariant.
2. Add or update the smallest test that proves the invariant.
3. Run the targeted test and confirm it fails for the expected reason.
4. Implement the code or config change.
5. Run the targeted test until it passes.
6. Run the broader gate:
   `python -m pytest web/tests` and `cd web/frontend && npm run build`.
7. For deployment changes, run `python -m pytest web/tests/test_deploy_readiness.py`.
8. After merge to `main`, let GitHub Actions run the same gates before deployment.

Current deployment invariants covered by tests:

- Production config uses explicit paths and secure cookies.
- The systemd unit uses a single worker and explicit environment variables.
- The deploy script installs dependencies and fails unhealthy API deployments.
- The smoke script remains syntactically valid Python.
- This ultraplan and the GitHub Actions deployment workflow are present and
  contain the required gates.

## Implementation Phases

### Phase 1: Lock The Deployment Contract

- Keep production paths in `web/config.production.yaml`.
- Keep service behavior in `deploy/locallatin.service`.
- Keep reverse proxy behavior in `deploy/nginx.conf`.
- Keep repeatable server deployment in `deploy/deploy.sh`.
- Add tests for any new deployment invariant before changing the files.

### Phase 2: CI Gate

- Run backend tests from a clean checkout.
- Run frontend `npm ci` from `web/frontend/package-lock.json`.
- Run the TypeScript/Vite production build.
- Fail the workflow on any dependency, typing, test, or build error.

### Phase 3: Automated Production Deploy

- Allow deployment only after the CI gate passes.
- Use the self-hosted runner to trigger deployment on the VM.
- Use `git pull --ff-only` to avoid deploying unresolved merge states.
- Restart through `deploy/deploy.sh`, not ad hoc remote commands.
- Require `/api/models` to respond before the deployment is considered healthy.

### Phase 4: Authenticated Smoke Checks

- Store smoke reviewer credentials on the server as environment variables or
  inject them only for manual deployment runs.
- Run `scripts/webapp/smoke_reviewer_pilot.py` from `deploy/deploy.sh` when
  `LOCALLATIN_SMOKE_USERNAME` and `LOCALLATIN_SMOKE_PASSWORD` are set.
- Use `LOCALLATIN_SMOKE_WRITE=1` only when an intentional feedback DB write is
  acceptable.

### Phase 5: Rollback And Recovery

- If Actions fails before deployment, fix the failing test or build locally.
- If deployment fails during remote execution, inspect:
  `journalctl --user -u locallatin.service -n 100`.
- Roll back by SSHing to the VM, checking out the last known good commit, and
  running `bash deploy/deploy.sh`.
- If nginx config changes, validate with `sudo nginx -t` before reloading.

## Local Verification Commands

Run these before pushing deployment changes:

```bash
python -m pytest web/tests
cd web/frontend && npm ci && npm run build
```

For a production host smoke run:

```bash
LOCALLATIN_SMOKE_USERNAME=<pi-admin-user> \
LOCALLATIN_SMOKE_PASSWORD=<pi-admin-password> \
bash deploy/deploy.sh
```
