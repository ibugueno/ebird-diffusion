# RGBE-Gaze server workflow

This guide documents two separate 512x512 experiments:

1. **Baseline (recommended):** the five-level architecture that already
   produced good event-conditioned face reconstructions.
2. **V2 (experimental):** the proposed six-level architecture with a 16x16
   bottleneck. Its commands are intentionally kept at the end of this guide.

Never reuse checkpoints across these architectures because their parameter
shapes are incompatible.

## 1. Update and start the container

Run on the host from the repository root:

```bash
git pull origin develop
docker build -t ignacio_event_ebird .
./run_docker.sh
```

The container exposes every GPU. Use `--device` for single-GPU sampling and
`CUDA_VISIBLE_DEVICES` with `torchrun` for DDP training.

## 2. Check paths and GPUs

Run inside the container:

```bash
git status
python -c "import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
ls /app/Rislab_Event_influence_volume/dataset/rgbe-gaze
```

## 3. Build the `user_1` manifests

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split \
  --users user_1 \
  --val-experiments exp5 \
  --test-experiments exp6
```

This assigns `exp1` through `exp4` to training, `exp5` to validation, and
`exp6` to testing. Only matching relative paths under `gray_frames` and
`event_accumulate_frames` are indexed. Unpaired files are skipped and reported.

Check and validate the resulting manifests:

```bash
wc -l /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split/{train,val,test}.csv

python scripts/validate_rgbe_dataset.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --manifest /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split/all.csv
```

# Baseline architecture (recommended)

The successful baseline is defined by `configs/rgbe_gaze/512_user1.yaml`:

```text
Resolution:                  512x512
U-Net levels:                [32, 64, 128, 256, 512]
Deepest resolution:          32x32
Learning rate:               0.0001
Epochs per stage:            40
Batch per GPU:               1
Gradient accumulation:       27
Effective batch on 3 GPUs:   1 x 3 x 27 = 81
Output directory:            runs/512-user1
```

The current trainer evaluates every epoch with fixed validation noise and
timesteps, making future `best.pt` selections reproducible. This does not alter
or invalidate the baseline checkpoints that already exist.

## 4. Train the baseline image branch

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1.yaml
```

Training resumes from `runs/512-user1/image/checkpoints/last.pt` when it exists.
Do not add `--no-resume` unless a fresh run in an empty output directory is
intended.

## 5. Inspect baseline image-branch samples

```bash
python scripts/sample_image_branch.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --num-samples 4 \
  --batch-size 1 \
  --seed 44
```

The output contains individual unconditional samples, `grid.png`,
`reference_grid.png`, and `metadata.json`. These samples are not paired
reconstructions; event conditioning is learned in the next stage.

To inspect a specific image checkpoint, add:

```bash
--checkpoint /path/to/image/checkpoint.pt \
--output-dir /path/to/a/separate/sample-directory
```

## 6. Train the baseline conditional branch

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1.yaml
```

This stage freezes the trained image branch and learns the ControlNet-style
event encoder. It resumes from
`runs/512-user1/conditional/checkpoints/last.pt` when available.

## 7. Reconstruct baseline test samples

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --limit 100 \
  --seed 44
```

Use `--limit -1` to reconstruct every entry in `test.csv`. The generated,
target, and event images are written under `samples/512-user1`.

To compare `best.pt`, `last.pt`, or another checkpoint without editing YAML:

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --base-checkpoint /path/to/image/checkpoint.pt \
  --conditional-checkpoint /path/to/conditional/checkpoint.pt \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/baseline-comparison \
  --limit 8 \
  --seed 44
```

## 8. Evaluate baseline reconstructions

```bash
python scripts/evaluate_rgbe_metrics.py \
  --samples-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/512-user1
```

The evaluator writes `metrics/per_image_metrics.csv` and
`metrics/summary.json`. Report the test sample count and mean plus standard
deviation for MSE, SSIM, and PSNR.

## 9. Multi-user baseline after dataset completion

Once enough users are available, rebuild identity-disjoint manifests and use a
dedicated configuration/output directory. Do not mix a multi-user run with the
current `user_1` checkpoints.

# V2 architecture (experimental)

V2 is retained for a future controlled comparison. It does not replace the
successful baseline.

```text
Resolution:                  512x512
U-Net levels:                [32, 64, 128, 256, 512, 512]
Deepest resolution:          16x16
Image-branch parameters:     approximately 52.5 million
Learning rate:               0.0001
Epochs per stage:            80
Batch per GPU:               3
Gradient accumulation:       9
Effective batch on 3 GPUs:   3 x 3 x 9 = 81
Output directory:            runs/512-user1-v2
```

## 10. Run the V2 memory smoke test

Run the image stage first:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1_v2_smoke.yaml \
  --no-resume
```

Then test the more memory-intensive conditional stage:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1_v2_smoke.yaml \
  --no-resume
```

Monitor both commands with `nvidia-smi` before starting a full V2 run.

## 11. Train V2

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1_v2.yaml
```

After the V2 image stage finishes:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1_v2.yaml
```

V2 uses deterministic validation and stores model-only snapshots at epochs 20,
40, 60, and 80. Its checkpoints and samples remain completely separate from
the baseline.
