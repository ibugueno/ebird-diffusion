#!/usr/bin/env bash
set -euo pipefail

LOG_ROOT="${OUTPUT_ROOT:-/app/outputs}/logs"
mkdir -p "${LOG_ROOT}"

echo "Starting E2VID"
bash "$(dirname "$0")/run_e2vid.sh" 2>&1 | tee "${LOG_ROOT}/e2vid.log"

echo "Starting ET-Net"
bash "$(dirname "$0")/run_etnet.sh" 2>&1 | tee "${LOG_ROOT}/etnet.log"

echo "Both baseline runs completed"
