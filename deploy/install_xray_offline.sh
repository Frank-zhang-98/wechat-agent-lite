#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Offline install Xray from local zip package.

Usage:
  sudo bash deploy/install_xray_offline.sh /path/to/Xray-linux-64.zip

Defaults:
  ZIP_PATH=/tmp/Xray-linux-64.zip
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "[ERROR] Please run as root."
  exit 1
fi

ZIP_PATH="${1:-/tmp/Xray-linux-64.zip}"
if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "[ERROR] Zip not found: ${ZIP_PATH}"
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "[1/6] Installing unzip..."
  apt-get update
  apt-get install -y unzip
fi

echo "[2/6] Unpacking ${ZIP_PATH} ..."
WORKDIR="$(mktemp -d /tmp/xray-offline.XXXXXX)"
cleanup() { rm -rf "${WORKDIR}" >/dev/null 2>&1 || true; }
trap cleanup EXIT
unzip -o "${ZIP_PATH}" -d "${WORKDIR}" >/dev/null

if [[ ! -f "${WORKDIR}/xray" ]]; then
  echo "[ERROR] Invalid package: missing xray binary"
  exit 1
fi

echo "[3/6] Installing binary and data files..."
install -m 755 "${WORKDIR}/xray" /usr/local/bin/xray
mkdir -p /usr/local/share/xray /usr/local/etc/xray
if [[ -f "${WORKDIR}/geoip.dat" ]]; then
  install -m 644 "${WORKDIR}/geoip.dat" /usr/local/share/xray/geoip.dat
fi
if [[ -f "${WORKDIR}/geosite.dat" ]]; then
  install -m 644 "${WORKDIR}/geosite.dat" /usr/local/share/xray/geosite.dat
fi

echo "[4/6] Ensuring systemd service..."
if [[ ! -f /etc/systemd/system/xray.service ]]; then
  cat >/etc/systemd/system/xray.service <<'EOF'
[Unit]
Description=Xray Service
After=network.target nss-lookup.target

[Service]
Type=simple
ExecStart=/usr/local/bin/xray run -config /usr/local/etc/xray/config.json
Restart=on-failure
RestartSec=5s
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF
fi

if [[ ! -f /usr/local/etc/xray/config.json ]]; then
  cat >/usr/local/etc/xray/config.json <<'EOF'
{
  "log": { "loglevel": "warning" },
  "inbounds": [
    {
      "listen": "127.0.0.1",
      "port": 10808,
      "protocol": "socks",
      "settings": { "auth": "noauth", "udp": true }
    }
  ],
  "outbounds": [
    { "protocol": "freedom" }
  ]
}
EOF
fi

echo "[5/6] Reloading and enabling service..."
systemctl daemon-reload
systemctl enable xray >/dev/null 2>&1 || true

echo "[6/6] Done."
xray version | head -n 1 || true
echo "You can now run:"
echo "  bash /tmp/setup_xray_placeholder.sh 'vless://...'"
