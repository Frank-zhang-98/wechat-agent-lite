#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   sudo bash deploy/bootstrap_ubuntu.sh /path/to/wechat-agent-lite

APP_SRC="${1:-}"
if [[ -z "${APP_SRC}" ]]; then
  echo "Usage: sudo bash deploy/bootstrap_ubuntu.sh /path/to/wechat-agent-lite"
  exit 1
fi

APP_USER="ShadowKun"
APP_HOME="/opt/wechat-agent-lite"
SERVICE_FILE="/etc/systemd/system/wechat-agent-lite.service"

echo "[1/8] create user: ${APP_USER}"
if ! id "${APP_USER}" &>/dev/null; then
  useradd -m -s /bin/bash "${APP_USER}"
fi

echo "[2/8] install system packages"
apt-get update
apt-get install -y python3 python3-venv python3-pip rsync

echo "[3/8] prepare dirs"
mkdir -p "${APP_HOME}" /var/log/wechat-agent-lite
chown -R "${APP_USER}:${APP_USER}" /var/log/wechat-agent-lite

echo "[4/8] sync project"
rsync -av --delete \
  --exclude ".git" \
  --exclude ".env" \
  --exclude ".venv" \
  --exclude "data/" \
  --exclude "output/" \
  --exclude "tmp/" \
  --exclude "__pycache__/" \
  "${APP_SRC}/" "${APP_HOME}/"
chown -R "${APP_USER}:${APP_USER}" "${APP_HOME}"

echo "[5/8] setup venv and deps"
sudo -u "${APP_USER}" bash -lc "
cd ${APP_HOME}
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
"

echo "[6/8] create .env if missing"
if [[ ! -f "${APP_HOME}/.env" ]]; then
  cp "${APP_HOME}/.env.example" "${APP_HOME}/.env"
  chown "${APP_USER}:${APP_USER}" "${APP_HOME}/.env"
fi

echo "[7/8] install systemd service"
cp "${APP_HOME}/deploy/systemd/wechat-agent-lite.service" "${SERVICE_FILE}"
systemctl daemon-reload
systemctl enable wechat-agent-lite.service

echo "[8/8] start service"
systemctl restart wechat-agent-lite.service
systemctl status --no-pager wechat-agent-lite.service || true

echo "Done. Access console through SSH tunnel:"
echo "ssh -L 18080:127.0.0.1:8080 ${APP_USER}@<server-ip>"
echo "Then open: http://127.0.0.1:18080"
