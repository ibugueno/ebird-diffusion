FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Base utilities and libraries required by OpenCV.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
        git \
        htop \
        libgl1 \
        libglib2.0-0 \
        tmux \
        vim \
        wget \
        zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Miniforge. It uses conda-forge and does not require accepting the
# Anaconda default-channel Terms of Service during a non-interactive build.
RUN wget -q \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
        -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

# environment.yml installs PyTorch 2.1 and its CUDA 12.1 wheels.
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Run subsequent build commands inside the EVDiff environment.
SHELL ["conda", "run", "--no-capture-output", "-n", "EVDiff", "/bin/bash", "-c"]

RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda)"

WORKDIR /app
COPY . /app

RUN mkdir -p /app/Rislab_Event_influence_volume

ENV PATH=/opt/conda/envs/EVDiff/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=EVDiff \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Activate EVDiff automatically in interactive shells.
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate EVDiff" >> /root/.bashrc

CMD ["bash"]
