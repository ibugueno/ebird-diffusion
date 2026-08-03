
#baseline architecture

CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1.yaml

CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1.yaml

python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --limit 100 \
  --seed 44

#v2 architecture

## Sí. Quedó garantizado mediante directorios independientes:

Baseline:       runs/512-user1
V2 40 épocas:   runs/512-user1-v2-40
V2 80 épocas:   runs/512-user1-v2
V2 smoke:       runs/512-user1-v2-smoke

CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1_v2_smoke.yaml \
  --no-resume

CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1_v2_40.yaml


CUDA_VISIBLE_DEVICES=1,3,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1_v2_40.yaml

python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1_v2_40.yaml \
  --limit 100 \
  --seed 44

python scripts/evaluate_rgbe_metrics.py \
  --samples-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/512-user1-v2-40

# --------------------------------------------------------------------------------------------

# Correr end to end en el servidor
# para exp1, exp5, exp6, con stride de 5

python scripts/run_generalization_v2.py \
  --protocol reduced \
  --gpus 1,2,4 \
  --force-prepare \
  --execute

# Evaluar generico

python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute

# Evaluar especifico

python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-evaluate \
  --sampling-device 1 \
  --samples-per-user 7 \
  --validation-limit 100 \
  --checkpoint-epochs 0 5 10 15 20 25 30 35 40 \
  --execute


# --------------------------------------------------------------------------------------------

# Correr end to end en el servidor
# para exp1-exp4, exp5, exp6, con stride de 5


python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --gpus 1,5,7 \
  --execute

# Evaluar generico

python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute

# Evaluar especifico

python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps specific-evaluate \
  --sampling-device 1 \
  --samples-per-user 7 \
  --validation-limit 100 \
  --checkpoint-epochs 40 \
  --execute