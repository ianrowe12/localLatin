#!/usr/bin/env bash
#
# deploy.sh — Build frontend and (re)start the LocalLatin service.
# Safe to re-run. Run from anywhere on the target VM.
#
set -euo pipefail

REPO_DIR="${DEPLOY_PATH:-/homes/ipro222/localLatin}"
FRONTEND_DIR="${REPO_DIR}/web/frontend"
STATIC_DIR="${REPO_DIR}/web/static"
DATA_DIR="${REPO_DIR}/data"
SERVICE_NAME="locallatin.service"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${REPO_DIR}/.venv"
LOCAL_BASE_URL="${LOCAL_BASE_URL:-http://127.0.0.1:8080}"
PUBLIC_BASE_PATH="${PUBLIC_BASE_PATH:-/}"

# Gitignored data payload (prediction CSVs + IG attribution artifacts), shipped
# as a GitHub Release asset because this host is only reachable through the
# Actions runner and the step above only git-pulls. Unset => no data sync,
# which keeps a checkout with the data already in place working unchanged.
DATA_RELEASE_TAG="${DATA_RELEASE_TAG:-}"
DATA_RELEASE_REPO="${DATA_RELEASE_REPO:-ianrowe12/localLatin}"
DATA_CACHE_DIR="${DATA_CACHE_DIR:-${REPO_DIR}/.deploy-cache}"

info()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; }

if [[ ! -d "${REPO_DIR}" ]]; then
    error "Repo not found at ${REPO_DIR}"
    exit 1
fi

# --- Data release sync -------------------------------------------------------
# Idempotent: an already-installed tag is a no-op, and a cached tarball whose
# checksum still matches is not re-downloaded.
sync_data_release() {
    if [[ -z "${DATA_RELEASE_TAG}" ]]; then
        info "DATA_RELEASE_TAG not set — skipping data sync (using data already on disk)."
        return 0
    fi

    local asset="locallatin-${DATA_RELEASE_TAG}.tar.gz"
    # Overridable so the sync can be exercised against a local directory
    # (file://...) without publishing a release.
    local base_url="${DATA_RELEASE_BASE_URL:-https://github.com/${DATA_RELEASE_REPO}/releases/download/${DATA_RELEASE_TAG}}"
    local tarball="${DATA_CACHE_DIR}/${asset}"
    local checksum="${tarball}.sha256"
    local marker="${DATA_CACHE_DIR}/installed-${DATA_RELEASE_TAG}"

    if [[ -f "${marker}" ]]; then
        info "Data release ${DATA_RELEASE_TAG} already installed — skipping."
        return 0
    fi

    mkdir -p "${DATA_CACHE_DIR}"

    info "Fetching data release ${DATA_RELEASE_TAG} from ${DATA_RELEASE_REPO}..."
    if ! curl -sSfL --retry 3 --retry-delay 5 -o "${checksum}.part" "${base_url}/${asset}.sha256"; then
        error "Could not download ${base_url}/${asset}.sha256"
        error "Check that release ${DATA_RELEASE_TAG} exists and carries both assets."
        return 1
    fi
    mv -f "${checksum}.part" "${checksum}"

    local have_valid_tarball=0
    if [[ -f "${tarball}" ]] && ( cd "${DATA_CACHE_DIR}" && sha256sum -c --status "$(basename "${checksum}")" ); then
        info "Cached ${asset} already matches the published checksum."
        have_valid_tarball=1
    fi

    if [[ "${have_valid_tarball}" -eq 0 ]]; then
        if ! curl -sSfL --retry 3 --retry-delay 5 -o "${tarball}.part" "${base_url}/${asset}"; then
            error "Could not download ${base_url}/${asset}"
            return 1
        fi
        mv -f "${tarball}.part" "${tarball}"
        if ! ( cd "${DATA_CACHE_DIR}" && sha256sum -c "$(basename "${checksum}")" ); then
            error "sha256 verification failed for ${asset} — refusing to unpack."
            rm -f "${tarball}"
            return 1
        fi
        info "Checksum verified."
    fi

    # Every member must live under runs/active/. This is what keeps the payload
    # from ever reaching the reviewer feedback database, which the production
    # config puts at data/feedback.db — outside this prefix and never touched.
    info "Validating archive member paths..."
    local entry
    while IFS= read -r entry; do
        case "${entry}" in
            runs/active/*) ;;
            *) error "Refusing archive: member outside runs/active/: ${entry}"; return 1 ;;
        esac
        case "${entry}" in
            *..*) error "Refusing archive: path traversal member: ${entry}"; return 1 ;;
        esac
    done < <(tar -tzf "${tarball}")

    local stage="${DATA_CACHE_DIR}/stage-${DATA_RELEASE_TAG}"
    rm -rf "${stage}"
    mkdir -p "${stage}"
    info "Unpacking to staging directory..."
    tar -xzf "${tarball}" -C "${stage}"

    # Install file by file with an atomic rename, so a reviewer hitting the
    # still-running old service never reads a half-written CSV.
    info "Installing data payload into ${REPO_DIR}..."
    local installed=0 src rel dest
    while IFS= read -r -d '' src; do
        rel="${src#"${stage}/"}"
        dest="${REPO_DIR}/${rel}"
        mkdir -p "$(dirname "${dest}")"
        cp -f "${src}" "${dest}.deploy-tmp"
        mv -f "${dest}.deploy-tmp" "${dest}"
        installed=$((installed + 1))
    done < <(find "${stage}" -type f -print0)
    rm -rf "${stage}"

    if [[ "${installed}" -eq 0 ]]; then
        error "Data release ${DATA_RELEASE_TAG} contained no files."
        return 1
    fi

    printf '%s\n' "${DATA_RELEASE_TAG}" > "${marker}"
    info "Installed ${installed} data files from ${DATA_RELEASE_TAG}."
}

# Ensure node is available
export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
if [[ -s "${NVM_DIR}/nvm.sh" ]]; then
    source "${NVM_DIR}/nvm.sh"
fi
if ! command -v node &>/dev/null; then
    error "node not found. Install via nvm: nvm install --lts"
    exit 1
fi

info "Using node $(node --version), npm $(npm --version)"
if ! command -v "${PYTHON_BIN}" &>/dev/null; then
    error "Python not found: ${PYTHON_BIN}"
    exit 1
fi
info "Using python $("${PYTHON_BIN}" --version)"

# Pull latest
cd "${REPO_DIR}"
if git rev-parse --is-inside-work-tree &>/dev/null; then
    info "Pulling latest changes..."
    git pull --ff-only || { error "git pull failed — resolve manually"; exit 1; }
fi

# Reviewer feedback must survive every deploy. The production config resolves
# paths.feedback_db ("data/feedback.db") against data_root, i.e.
# ${REPO_DIR}/data/feedback.db. Nothing below deletes it: the data payload is
# confined to runs/active/, and only web/static/ is ever removed. Fingerprint
# it anyway so a regression shows up here rather than as lost reviewer work.
FEEDBACK_DB="${FEEDBACK_DB:-${DATA_DIR}/feedback.db}"
feedback_fingerprint() {
    if [[ -f "${FEEDBACK_DB}" ]]; then
        stat -c '%i:%s' "${FEEDBACK_DB}"
    else
        printf 'absent\n'
    fi
}
FEEDBACK_BEFORE="$(feedback_fingerprint)"
info "Feedback DB before deploy: ${FEEDBACK_DB} (${FEEDBACK_BEFORE})"

# Gitignored data (prediction CSVs, IG artifacts) does not arrive via git.
sync_data_release || { error "Data release sync failed."; exit 1; }

FEEDBACK_AFTER="$(feedback_fingerprint)"
if [[ "${FEEDBACK_BEFORE}" != "${FEEDBACK_AFTER}" ]]; then
    error "Feedback DB changed during the data sync: ${FEEDBACK_BEFORE} -> ${FEEDBACK_AFTER}"
    error "The data payload must never touch ${FEEDBACK_DB}."
    exit 1
fi

if [[ "${DEPLOY_SKIP_DATA_CHECK:-0}" == "1" ]]; then
    info "DEPLOY_SKIP_DATA_CHECK=1 — not verifying the webapp data contract."
else
    info "Verifying webapp data contract..."
    bash "${REPO_DIR}/scripts/webapp/export_webapp_data.sh" "${REPO_DIR}" --strict || {
        error "Required webapp data is missing — the service would fail to start."
        exit 1
    }
fi

info "Installing backend dependencies..."
"${PYTHON_BIN}" -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/python" -m pip install -r "${REPO_DIR}/web/requirements.txt"

# Build frontend
info "Installing frontend dependencies..."
cd "${FRONTEND_DIR}"
npm ci --prefer-offline --include=dev

info "Building frontend..."
VITE_BASE_PATH="${PUBLIC_BASE_PATH}" npm run build

# Deploy to web/static/
info "Deploying frontend to ${STATIC_DIR}..."
rm -rf "${STATIC_DIR}"
cp -r "${FRONTEND_DIR}/dist" "${STATIC_DIR}"

# Ensure data directory
mkdir -p "${DATA_DIR}"
if [[ ! -w "${DATA_DIR}" ]]; then
    error "Feedback/data directory is not writable by $(whoami): ${DATA_DIR}"
    exit 1
fi

# Install systemd user unit
info "Installing systemd user unit..."
mkdir -p "${SYSTEMD_USER_DIR}"
cp "${REPO_DIR}/deploy/locallatin.service" "${SYSTEMD_USER_DIR}/${SERVICE_NAME}"
systemctl --user daemon-reload

# Enable linger
if ! loginctl show-user "$(whoami)" -p Linger 2>/dev/null | grep -q "Linger=yes"; then
    info "Enabling linger for $(whoami)..."
    loginctl enable-linger "$(whoami)" 2>/dev/null || \
        error "Could not enable linger — ask sysadmin: loginctl enable-linger $(whoami)"
fi

# (Re)start service
info "Restarting ${SERVICE_NAME}..."
systemctl --user enable "${SERVICE_NAME}"
systemctl --user restart "${SERVICE_NAME}"

sleep 2
if systemctl --user is-active --quiet "${SERVICE_NAME}"; then
    info "Service is running."
    systemctl --user status "${SERVICE_NAME}" --no-pager
else
    error "Service failed to start. Check: journalctl --user -u ${SERVICE_NAME} -n 50"
    exit 1
fi

# Health check. /api/models is intentionally unauthenticated and proves the
# backend loaded the model/data store without requiring a reviewer session.
info "Checking API health..."
healthy=0
for attempt in {1..30}; do
    if curl -sf "${LOCAL_BASE_URL}/api/models" >/dev/null 2>&1; then
        healthy=1
        break
    fi
    sleep 2
done
if [[ "${healthy}" -eq 1 ]]; then
    info "API responding at ${LOCAL_BASE_URL}."
else
    error "API did not become healthy at ${LOCAL_BASE_URL}/api/models"
    error "Check: journalctl --user -u ${SERVICE_NAME} -n 100"
    exit 1
fi

# Exercise a real auth database read without needing or modifying a production
# account. An unknown user must be rejected normally, never crash the route.
info "Checking authentication database path..."
auth_status="$(curl -sS -o /dev/null -w '%{http_code}' \
    -X POST "${LOCAL_BASE_URL}/api/auth/signin" \
    -H "Content-Type: application/json" \
    --data '{"username":"__deployment_healthcheck__","password":"not-a-real-password"}')"
if [[ "${auth_status}" == "401" ]]; then
    info "Authentication database path is healthy."
else
    error "Authentication health check returned HTTP ${auth_status}; expected 401"
    journalctl --user -u "${SERVICE_NAME}" -n 50 --no-pager >&2 || true
    exit 1
fi

if [[ -n "${LOCALLATIN_SMOKE_USERNAME:-}" && -n "${LOCALLATIN_SMOKE_PASSWORD:-}" ]]; then
    info "Running authenticated reviewer-pilot smoke checks..."
    "${PYTHON_BIN}" "${REPO_DIR}/scripts/webapp/smoke_reviewer_pilot.py" \
        --base-url "${LOCAL_BASE_URL}" \
        --username "${LOCALLATIN_SMOKE_USERNAME}" \
        --password "${LOCALLATIN_SMOKE_PASSWORD}" \
        ${LOCALLATIN_SMOKE_WRITE:+--write-check}
else
    info "Skipping authenticated smoke checks. Set LOCALLATIN_SMOKE_USERNAME and LOCALLATIN_SMOKE_PASSWORD to enable them."
fi

info ""
info "Deployment complete."
info "  Frontend: ${STATIC_DIR}"
info "  Config:   ${REPO_DIR}/web/config.production.yaml"
info "  Local URL: ${LOCAL_BASE_URL}"
info "  Public path: ${PUBLIC_BASE_PATH}"
info "  Logs:     journalctl --user -u ${SERVICE_NAME} -f"
info ""
info "If nginx not configured yet:"
info "  ensure the active ai.csr.uky.edu server block proxies / to http://127.0.0.1:8080"
info "  sudo nginx -t && sudo systemctl reload nginx"
