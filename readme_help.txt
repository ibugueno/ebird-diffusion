REQUISITOS DEL SERVIDOR

- Docker Engine
- GPU NVIDIA y driver compatible con CUDA 12.1
- NVIDIA Container Toolkit
- Dataset organizado bajo Rislab_Event_influence_volume/dataset


1. CONSTRUIR LA IMAGEN

docker build -t ebird-mnist:cu121 .


2. EJECUTAR TODO EL FLUJO

El directorio indicado en /ruta/datos-y-resultados debe contener el dataset y
también recibirá checkpoints, logs y muestras generadas.

docker run --rm \
  --gpus "device=0" \
  --ipc=host \
  --name ebird-training \
  -v /ruta/datos-y-resultados:/app/Rislab_Event_influence_volume \
  ebird-mnist:cu121


3. ABRIR UNA TERMINAL DE DEPURACIÓN

docker run --rm -it \
  --gpus "device=0" \
  --ipc=host \
  -v /ruta/datos-y-resultados:/app/Rislab_Event_influence_volume \
  ebird-mnist:cu121 bash

Dentro del contenedor:

conda run --no-capture-output -n EVDiff python src/Train_Image_Branch.py --config src/default.yaml
conda run --no-capture-output -n EVDiff python src/Train_Conditional_Partition.py --config src/default.yaml
conda run --no-capture-output -n EVDiff python src/DualSample_ajustable_evalgen_boost.py --config src/default.yaml


4. VERIFICAR LA GPU DENTRO DEL CONTENEDOR

conda run -n EVDiff python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
