#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-ebird-event-reconstruction-baselines:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-ebird-event-reconstruction-baselines}"
GPU_ID="${GPU_ID:-0}"
DATA_DIR="${RGBE_VALIDATION_DIR:-/home/ignacio/Projects/UOH/Event-Cameras/dataset/event-based_datasets/reconstruction/rgbe-gaze-validation_exp5}"
WEIGHTS_DIR="${BASELINE_WEIGHTS_DIR:-$(pwd)/weights}"
OUTPUT_DIR="${BASELINE_OUTPUT_DIR:-$(pwd)/outputs}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${WEIGHTS_DIR}" "${OUTPUT_DIR}"

if [[ ! -d "${DATA_DIR}" ]]; then
    echo "Data directory does not exist: ${DATA_DIR}" >&2
    exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker run --gpus "device=${GPU_ID}" -it \
    --name "${CONTAINER_NAME}" \
    --shm-size=16g \
    -v "${SCRIPT_DIR}:/app/event_reconstruction_baselines:ro" \
    -v "${DATA_DIR}:/app/data:ro" \
    -v "${WEIGHTS_DIR}:/app/weights:ro" \
    -v "${OUTPUT_DIR}:/app/outputs" \
    "${IMAGE_NAME}" bash
