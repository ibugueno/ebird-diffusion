#!/usr/bin/env bash
set -euo pipefail

WEIGHTS="${ETNET_WEIGHTS:-/app/weights/etnet_checkpoint.pth}"
DATA_ROOT="${DATA_ROOT:-/app/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/app/outputs/etnet}"
GPU_ID="${GPU_ID:-0}"

if [[ ! -f "${WEIGHTS}" ]]; then
    echo "Missing ET-Net checkpoint: ${WEIGHTS}" >&2
    exit 1
fi
mkdir -p "${OUTPUT_ROOT}"

mapfile -t EVENT_FILES < <(find "${DATA_ROOT}/etnet_256" -type f -name events.h5 | sort)
if [[ "${#EVENT_FILES[@]}" -eq 0 ]]; then
    echo "No ET-Net HDF5 sequences found below ${DATA_ROOT}/etnet_256" >&2
    exit 1
fi

for event_file in "${EVENT_FILES[@]}"; do
    relative="${event_file#"${DATA_ROOT}/etnet_256/}"
    sequence="${relative%/events.h5}"
    output_dir="${OUTPUT_ROOT}/${sequence}"
    mkdir -p "${output_dir}"
    echo "[ET-Net] ${sequence}"
    (
        cd /app/event_reconstruction_baselines
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python scripts/run_etnet_inference.py \
            --checkpoint_path "${WEIGHTS}" \
            --events_file_path "${event_file}" \
            --output_folder "${output_dir}" \
            --device 0 \
            --loader_type H5 \
            --voxel_method between_frames
    ) 2>&1 | tee "${output_dir}/inference.log"
done
