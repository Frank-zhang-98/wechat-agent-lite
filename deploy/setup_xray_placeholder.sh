#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-click Xray client setup from VLESS share link.

Usage:
  sudo bash deploy/setup_xray_placeholder.sh "<vless://...>"

Optional env vars:
  SOCKS_LISTEN=127.0.0.1
  SOCKS_PORT=10808
  CONFIG_FILE=/usr/local/etc/xray/config.json
  INSTALL_XRAY=true

Notes:
  - Supports vless:// links (including reality + xhttp).
  - Will install xray if missing (unless INSTALL_XRAY=false).
  - Will backup existing config file before overwrite.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${EUID}" -ne 0 ]]; then
  echo "[ERROR] Please run as root: sudo bash $0 \"<vless://...>\""
  exit 1
fi

SHARE_LINK="${1:-${SHARE_LINK:-}}"
if [[ -z "${SHARE_LINK}" ]]; then
  read -r -p "Paste VLESS share link: " SHARE_LINK
fi
if [[ -z "${SHARE_LINK}" ]]; then
  echo "[ERROR] Empty share link"
  exit 1
fi

SOCKS_LISTEN="${SOCKS_LISTEN:-127.0.0.1}"
SOCKS_PORT="${SOCKS_PORT:-10808}"
CONFIG_FILE="${CONFIG_FILE:-/usr/local/etc/xray/config.json}"
INSTALL_XRAY="${INSTALL_XRAY:-true}"
XRAY_INSTALL_SCRIPT_URL="${XRAY_INSTALL_SCRIPT_URL:-https://github.com/XTLS/Xray-install/raw/main/install-release.sh}"
XRAY_INSTALL_SCRIPT_URL_FALLBACK="${XRAY_INSTALL_SCRIPT_URL_FALLBACK:-https://raw.githubusercontent.com/XTLS/Xray-install/main/install-release.sh}"
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH}"

for cmd in python3 curl systemctl; do
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "[ERROR] Missing command: ${cmd}"
    exit 1
  fi
done

if ! command -v xray >/dev/null 2>&1; then
  if [[ "${INSTALL_XRAY}" != "true" ]]; then
    echo "[ERROR] xray not found and INSTALL_XRAY=false"
    exit 1
  fi
  echo "[1/6] Installing xray..."
  TMP_INSTALL_SCRIPT="$(mktemp -t xray-install.XXXXXX.sh)"
  cleanup_install_script() { rm -f "${TMP_INSTALL_SCRIPT}" >/dev/null 2>&1 || true; }
  trap cleanup_install_script EXIT

  if ! curl -fsSL --retry 3 --retry-delay 2 "${XRAY_INSTALL_SCRIPT_URL}" -o "${TMP_INSTALL_SCRIPT}"; then
    echo "[WARN] Primary installer URL failed. Trying fallback URL..."
    curl -fsSL --retry 3 --retry-delay 2 "${XRAY_INSTALL_SCRIPT_URL_FALLBACK}" -o "${TMP_INSTALL_SCRIPT}"
  fi
  chmod +x "${TMP_INSTALL_SCRIPT}"
  bash "${TMP_INSTALL_SCRIPT}"

  if ! command -v xray >/dev/null 2>&1; then
    echo "[ERROR] xray install did not produce executable in PATH."
    echo "        Try offline install, then rerun with INSTALL_XRAY=false."
    exit 1
  fi
else
  echo "[1/6] xray already installed."
fi

echo "[2/6] Parsing share link..."
PARSED_JSON="$(python3 - "${SHARE_LINK}" <<'PY'
import json
import sys
from urllib.parse import parse_qs, urlparse, unquote

link = (sys.argv[1] if len(sys.argv) > 1 else "").strip()
if not link.startswith("vless://"):
    raise SystemExit("Only vless:// is supported by this script")

u = urlparse(link)
host = (u.hostname or "").strip()
port = int(u.port or 0)
uuid = (u.username or "").strip()
if not host or not port or not uuid:
    raise SystemExit("Invalid vless link: host/port/uuid is required")

q = parse_qs(u.query)
def qv(key: str, default: str = "") -> str:
    val = q.get(key, [default])[0]
    return (val or default).strip()

obj = {
    "scheme": "vless",
    "host": host,
    "port": port,
    "uuid": uuid,
    "encryption": qv("encryption", "none"),
    "security": qv("security", "none"),
    "type": qv("type", "tcp"),
    "sni": qv("sni", ""),
    "fp": qv("fp", ""),
    "pbk": qv("pbk", ""),
    "sid": qv("sid", ""),
    "spx": unquote(qv("spx", "")),
    "path": unquote(qv("path", "")),
    "flow": qv("flow", ""),
    "tag": unquote((u.fragment or "").strip()),
}
print(json.dumps(obj, ensure_ascii=False))
PY
)"

echo "[3/6] Writing xray config to ${CONFIG_FILE} ..."
export PARSED_JSON SOCKS_LISTEN SOCKS_PORT CONFIG_FILE
python3 <<'PY'
import json
import os
from pathlib import Path

info = json.loads(os.environ["PARSED_JSON"])
socks_listen = os.environ["SOCKS_LISTEN"]
socks_port = int(os.environ["SOCKS_PORT"])
config_path = Path(os.environ["CONFIG_FILE"])
config_path.parent.mkdir(parents=True, exist_ok=True)

network = (info.get("type") or "tcp").strip() or "tcp"
security = (info.get("security") or "none").strip() or "none"

user = {
    "id": info["uuid"],
    "encryption": info.get("encryption", "none") or "none",
}
if info.get("flow"):
    user["flow"] = info["flow"]

stream = {
    "network": network,
    "security": security,
}

if security == "reality":
    reality = {}
    if info.get("sni"):
        reality["serverName"] = info["sni"]
    if info.get("fp"):
        reality["fingerprint"] = info["fp"]
    if info.get("pbk"):
        reality["publicKey"] = info["pbk"]
    if info.get("sid"):
        reality["shortId"] = info["sid"]
    if info.get("spx"):
        reality["spiderX"] = info["spx"]
    if reality:
        stream["realitySettings"] = reality
elif security == "tls":
    tls = {}
    if info.get("sni"):
        tls["serverName"] = info["sni"]
    if tls:
        stream["tlsSettings"] = tls

if network == "xhttp":
    xhttp = {}
    if info.get("path"):
        xhttp["path"] = info["path"]
    if xhttp:
        stream["xhttpSettings"] = xhttp
elif network == "ws":
    ws = {}
    if info.get("path"):
        ws["path"] = info["path"]
    if ws:
        stream["wsSettings"] = ws
elif network == "grpc":
    grpc = {}
    if info.get("path"):
        grpc["serviceName"] = info["path"].lstrip("/")
    if grpc:
        stream["grpcSettings"] = grpc

config = {
    "log": {"loglevel": "warning"},
    "inbounds": [
        {
            "tag": "socks-in",
            "listen": socks_listen,
            "port": socks_port,
            "protocol": "socks",
            "settings": {"auth": "noauth", "udp": True},
            "sniffing": {
                "enabled": True,
                "destOverride": ["http", "tls", "quic"],
            },
        }
    ],
    "outbounds": [
        {
            "tag": "proxy",
            "protocol": "vless",
            "settings": {
                "vnext": [
                    {
                        "address": info["host"],
                        "port": info["port"],
                        "users": [user],
                    }
                ]
            },
            "streamSettings": stream,
        },
        {"tag": "direct", "protocol": "freedom"},
        {"tag": "block", "protocol": "blackhole"},
    ],
}

if config_path.exists():
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    config_path.replace(backup)

with config_path.open("w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

echo "[4/6] Validating xray config..."
xray run -test -config "${CONFIG_FILE}"

echo "[5/6] Restarting xray service..."
systemctl daemon-reload
systemctl enable xray >/dev/null 2>&1 || true
systemctl restart xray
sleep 1
systemctl --no-pager --full status xray | sed -n '1,20p'

echo "[6/6] Testing outbound through local proxy..."
set +e
EGRESS_IP="$(curl -sS --max-time 15 --socks5-hostname "${SOCKS_LISTEN}:${SOCKS_PORT}" https://api.ipify.org)"
CURL_CODE=$?
set -e
if [[ ${CURL_CODE} -ne 0 || -z "${EGRESS_IP}" ]]; then
  echo "[WARN] Proxy test failed. Check logs:"
  echo "  journalctl -u xray -n 200 --no-pager"
  exit 2
fi

echo
echo "Done. Xray outbound is working."
echo "Detected egress IP: ${EGRESS_IP}"
echo
echo "Use these values in wechat-agent-lite:"
echo "  proxy.enabled = true"
echo "  proxy.all_proxy = socks5h://${SOCKS_LISTEN}:${SOCKS_PORT}"
echo "  proxy.share_link = <your vless link>"
