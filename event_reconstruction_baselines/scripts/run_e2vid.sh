#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${E2VID_WEIGHTS:-/app/weights/E2VID_lightweight.pth.tar}"
DATA_ROOT="${DATA_ROOT:-/app/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/app/outputs/e2vid}"
GPU_ID="${GPU_ID:-0}"
WINDOW_MS="${WINDOW_MS:-33.33}"

if [[ ! -f "${WEIGHTS}" ]]; then
    echo "Missing E2VID checkpoint: ${WEIGHTS}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

mapfile -t EVENT_FILES < <(find "${DATA_ROOT}/e2vid_firenet_512" -type f -name events.txt | sort)
if [[ "${#EVENT_FILES[@]}" -eq 0 ]]; then
    echo "No E2VID event streams found below ${DATA_ROOT}/e2vid_firenet_512" >&2
    exit 1
fi

for event_file in "${EVENT_FILES[@]}"; do
    relative="${event_file#"${DATA_ROOT}/e2vid_firenet_512/}"
    sequence="${relative%/events.txt}"
    output_dir="${OUTPUT_ROOT}/${sequence}"
    mkdir -p "${output_dir}"
    echo "[E2VID] ${sequence}"
    (
        cd /app/event_reconstruction_baselines/third_party/e2vid
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python run_reconstruction.py \
            --path_to_model "${WEIGHTS}" \
            --input_file "${event_file}" \
            --fixed_duration \
            --window_duration "${WINDOW_MS}" \
            --output_folder "${output_dir}"
    ) 2>&1 | tee "${output_dir}/inference.log"
done
