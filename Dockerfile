FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# Dependencias básicas y librerías requeridas por OpenCV.
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

# Instalar Miniconda.
RUN wget -q \
        https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:$PATH

# environment.yml instala PyTorch 2.1 y sus wheels para CUDA 12.1.
COPY environment.yml /tmp/environment.yml

RUN conda env create -f /tmp/environment.yml && \
    conda clean -afy

# Ejecutar los siguientes RUN dentro del entorno EVDiff.
SHELL ["conda", "run", "--no-capture-output", "-n", "EVDiff", "/bin/bash", "-c"]

RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda)"

WORKDIR /app
COPY . /app

RUN mkdir -p /app/Rislab_Event_influence_volume

ENV PATH=/opt/conda/envs/EVDiff/bin:/opt/conda/bin:$PATH \
    CONDA_DEFAULT_ENV=EVDiff \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Activar EVDiff automáticamente al abrir una shell interactiva.
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate EVDiff" >> /root/.bashrc

CMD ["bash"]
