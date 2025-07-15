# Usa una imagen base con Python y Conda
FROM continuumio/miniconda3

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia los archivos del proyecto al contenedor
COPY . /app

# Instala las dependencias desde environment.yml
RUN conda env create -f environment.yml

# Activa el entorno y configura SHELL para que los siguientes comandos lo usen
SHELL ["conda", "run", "-n", "EVDiff", "/bin/bash", "-c"]

RUN ["mkdir", "-p", "/app/Rislab_Event_influence_volume"]

CMD ["python", "run.py"]  # Ejecutar el script principal
