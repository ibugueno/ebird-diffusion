#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ebird-event-reconstruction-baselines:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker build -f "${SCRIPT_DIR}/Dockerfile" -t "${IMAGE_NAME}" "${SCRIPT_DIR}"
