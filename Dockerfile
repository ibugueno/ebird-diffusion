FROM continuumio/miniconda3

WORKDIR /app

# Dependencias del sistema requeridas por OpenCV.
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY environment.yml /app/environment.yml

RUN conda env create -f /app/environment.yml && \
    conda clean -afy

COPY . /app

RUN mkdir -p /app/Rislab_Event_influence_volume

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Ejecuta las tres etapas definidas en src/run.py dentro del entorno Conda.
CMD ["conda", "run", "--no-capture-output", "-n", "EVDiff", "python", "/app/src/run.py"]
