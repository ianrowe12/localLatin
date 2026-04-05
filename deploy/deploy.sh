#!/usr/bin/env bash
#
# deploy.sh — Build frontend and (re)start the LocalLatin service.
# Safe to re-run. Run from anywhere on the target VM.
#
set -euo pipefail

REPO_DIR="/u/irowerojas/localLatin"
FRONTEND_DIR="${REPO_DIR}/web/frontend"
STATIC_DIR="${REPO_DIR}/web/static"
DATA_DIR="${REPO_DIR}/data"
SERVICE_NAME="locallatin.service"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

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

# Pull latest
cd "${REPO_DIR}"
if git rev-parse --is-inside-work-tree &>/dev/null; then
    info "Pulling latest changes..."
    git pull --ff-only || { error "git pull failed — resolve manually"; exit 1; }
fi

# Build frontend
info "Installing frontend dependencies..."
cd "${FRONTEND_DIR}"
npm ci --prefer-offline

info "Building frontend..."
npm run build

# Deploy to web/static/
info "Deploying frontend to ${STATIC_DIR}..."
rm -rf "${STATIC_DIR}"
cp -r "${FRONTEND_DIR}/dist" "${STATIC_DIR}"

# Ensure data directory
mkdir -p "${DATA_DIR}"

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

# Health check
info "Checking API health..."
if curl -sf http://127.0.0.1:8000/api/stats >/dev/null 2>&1; then
    info "API responding on port 8000."
else
    info "API not yet responding — may still be loading. Check: journalctl --user -u ${SERVICE_NAME} -f"
fi

info ""
info "Deployment complete."
info "  Frontend: ${STATIC_DIR}"
info "  Config:   ${REPO_DIR}/web/config.production.yaml"
info "  Logs:     journalctl --user -u ${SERVICE_NAME} -f"
info ""
info "If nginx not configured yet:"
info "  sudo cp ${REPO_DIR}/deploy/nginx.conf /etc/nginx/conf.d/locallatin.conf"
info "  sudo nginx -t && sudo systemctl reload nginx"
