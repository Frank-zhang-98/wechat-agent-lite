#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   sudo bash deploy/deploy_uploaded_zip.sh /path/to/wechat-agent-lite-YYYYMMDD-HHMMSS.zip

ZIP_PATH="${1:-}"
if [[ -z "${ZIP_PATH}" ]]; then
  echo "Usage: sudo bash deploy/deploy_uploaded_zip.sh /path/to/wechat-agent-lite-YYYYMMDD-HHMMSS.zip"
  exit 1
fi

if [[ ! -f "${ZIP_PATH}" ]]; then
  echo "Zip not found: ${ZIP_PATH}"
  exit 1
fi

WORK_DIR="/tmp/wechat-agent-lite-release-$$"
mkdir -p "${WORK_DIR}"
trap 'rm -rf "${WORK_DIR}"' EXIT

echo "[1/3] unpack release zip"
unzip -oq "${ZIP_PATH}" -d "${WORK_DIR}"

echo "[2/3] run bootstrap deploy"
bash "${WORK_DIR}/deploy/bootstrap_ubuntu.sh" "${WORK_DIR}"

echo "[3/3] deployment complete"
