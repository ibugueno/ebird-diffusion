# Flujo de prueba RGBE-Gaze en el servidor

Esta guía ejecuta el pipeline a 256×256 usando inicialmente sólo `user_1` y la
GPU 1. Los datos originales permanecen a 512×512 y se redimensionan durante la
carga, sin modificar los PNG.

## 1. Actualizar y construir

En el host, desde la raíz del repositorio:

```bash
git pull origin develop
docker build -t ignacio_event_ebird .
./run_docker.sh
```

El contenedor expone todas las GPU. La selección se hace en cada comando con
`--device` o, para DDP, mediante `CUDA_VISIBLE_DEVICES`.

## 2. Comprobar rutas y GPU

Dentro del contenedor:

```bash
git status
python -c "import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
ls /app/Rislab_Event_influence_volume/dataset/rgbe-gaze
```

## 3. Generar manifests sólo para user_1

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1 \
  --users user_1
```

El generador utiliza únicamente pares con la misma ruta relativa bajo
`gray_frames` y `event_accumulate_frames`. Los archivos sin pareja se omiten y
se reportan como `skipped_without_event` o `skipped_without_target`. Para exigir
un dataset perfectamente pareado puede añadirse `--strict-pairs`.

Verifica los archivos generados:

```bash
ls -lh /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1
wc -l /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1/train.csv
```

## 4. Validar imágenes indexadas

```bash
python scripts/validate_rgbe_dataset.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --manifest /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1/train.csv
```

## 5. Smoke de la rama de imágenes

Ejecuta dos batches a 256×256 en la GPU 1:

```bash
python scripts/train_rgbe.py \
  --stage image \
  --device 1 \
  --config configs/rgbe_gaze/256_smoke.yaml
```

## 6. Smoke de la rama condicional

Esta etapa carga automáticamente el mejor checkpoint de la rama anterior:

```bash
python scripts/train_rgbe.py \
  --stage conditional \
  --device 1 \
  --config configs/rgbe_gaze/256_smoke.yaml
```

## 7. Generar una muestra

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/256_smoke.yaml \
  --limit 1
```

## 8. Entrenamiento completo con GPU 1, 2 y 4

Cuando existan suficientes usuarios para train, validación y test, genera los
manifests sin `--users` y usa la configuración completa:

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests
```

Rama base:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/256.yaml
```

Rama condicional:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/256.yaml
```

No uses `--device` junto con `torchrun`: cada proceso recibe su GPU mediante el
rango local de DDP.
