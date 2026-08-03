FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
        libgl1 \
        libglib2.0-0 \
        tmux \
        vim \
        wget \
        zip \
    && rm -rf /var/lib/apt/lists/*

# The repositories use deprecated NumPy/skimage APIs (for example np.int and
# skimage.measure.compare_ssim), so these versions are intentional.
RUN pip install --no-cache-dir \
        "numpy==1.23.5" \
        "pandas==1.5.3" \
        "scipy==1.10.1" \
        "scikit-image==0.19.3" \
        "opencv-python-headless==4.8.1.78" \
        "h5py==3.8.0" \
        "PyYAML==6.0.1" \
        "tqdm==4.66.1"

WORKDIR /app/event_reconstruction_baselines
COPY third_party ./third_party
COPY scripts ./scripts
COPY THIRD_PARTY_VERSIONS.txt README.md CODEX_CONTEXT.md ./

RUN mkdir -p /app/data /app/weights /app/outputs /app/logs

ENV E2VID_ROOT=/app/event_reconstruction_baselines/third_party/e2vid \
    ETNET_ROOT=/app/event_reconstruction_baselines/third_party/et-net \
    DATA_ROOT=/app/data \
    WEIGHTS_ROOT=/app/weights \
    OUTPUT_ROOT=/app/outputs \
    PYTHONPATH=/app/event_reconstruction_baselines/third_party/e2vid:/app/event_reconstruction_baselines/third_party/et-net

CMD ["bash"]
