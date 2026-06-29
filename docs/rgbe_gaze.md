# Entrenamiento RGBE-Gaze a 256×256

La implementación RGBE-Gaze es independiente de los scripts MNIST ubicados en
`src/`. Conserva el mismo flujo conceptual de `readme_help.txt`:

1. entrenar la rama DDPM de imágenes;
2. congelar esa rama y entrenar el condicionamiento por eventos;
3. generar rostros desde representaciones de eventos acumulados.

## 1. Construir los manifests

El volumen de entrada debe exponer:

```text
/app/Rislab_Event_influence_volume/dataset/rgbe-gaze/
├── gray_frames/user_N/expM/*.png
└── event_accumulate_frames/user_N/expM/*.png
```

Cada par debe compartir la misma ruta relativa y nombre. Para construir splits
sin mezclar un usuario entre train, validación y test:

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests \
  --val-ratio 0.1 \
  --test-ratio 0.1
```

Para una prueba preliminar usando exclusivamente `user_1`:

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1 \
  --users user_1
```

En ese caso todas las muestras quedan en `train.csv`; no se crean muestras de
validación o test porque la separación se realiza por identidad.

Validación opcional de todos los PNG:

```bash
python scripts/validate_rgbe_dataset.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --manifest /app/Rislab_Event_influence_volume/rgbe-gaze/manifests/all.csv
```

## 2. Entrenar la rama de imágenes

Prueba primero dos batches reales en una sola GPU:

```bash
python scripts/train_rgbe.py \
  --stage image \
  --device 1 \
  --config configs/rgbe_gaze/256_smoke.yaml
```

Repite el smoke test con `--stage conditional`. Cuando ambas etapas terminen,
utiliza `256.yaml` para el entrenamiento completo.

Cuando el primer epoch corto funcione, usa las tres GPU con DDP:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/256.yaml
```

## 3. Entrenar la rama condicional

Esta etapa exige el checkpoint `best.pt` de la rama anterior:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/256.yaml
```

## 4. Generar reconstrucciones

```bash
python scripts/sample_rgbe.py \
  --config configs/rgbe_gaze/256.yaml \
  --device 1 \
  --limit 16
```

## Configuración de memoria

La configuración inicial usa por GPU:

- resolución 256×256;
- batch 1;
- AMP;
- acumulación de 8 pasos;
- gradient checkpointing;
- atención únicamente a 32×32 y 16×16.

Con tres procesos DDP, el batch efectivo es 24. DDP replica el modelo en cada
GPU; no suma sus memorias para una sola muestra.
