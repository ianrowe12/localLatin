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

info()  { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
error() { printf '\033[1;31m[deploy]\033[0m %s\n' "$*" >&2; }

if [[ ! -d "${REPO_DIR}" ]]; then
    error "Repo not found at ${REPO_DIR}"
    exit 1
fi

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
