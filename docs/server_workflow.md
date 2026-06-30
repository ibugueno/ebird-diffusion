# RGBE-Gaze server workflow

This guide trains the 512x512 pipeline using only `user_1` on GPUs 1, 2, and 4.
The loader uses the source resolution directly and never changes the PNG files.

## 1. Update and start the container

Run these commands on the host from the repository root:

```bash
git pull origin develop
docker build -t ignacio_event_ebird .
./run_docker.sh
```

The container exposes every GPU. Select one with `--device`, or select several
with `CUDA_VISIBLE_DEVICES` when using DDP.

## 2. Check paths and GPUs

Run these commands inside the container:

```bash
git status
python -c "import torch; print(torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
ls /app/Rislab_Event_influence_volume/dataset/rgbe-gaze
```

## 3. Build `user_1` manifests

Use complete experiments for each split. This prevents adjacent frames from the
same recording from appearing in both training and evaluation:

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
`event_accumulate_frames` are indexed. Unpaired files are skipped and reported;
add `--strict-pairs` only when incomplete pairs should be treated as an error.

Check the resulting split sizes:

```bash
wc -l /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split/{train,val,test}.csv
```

## 4. Validate the indexed images

```bash
python scripts/validate_rgbe_dataset.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --manifest /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split/all.csv
```

## 5. Train at 512x512 on GPUs 1, 2, and 4

The paper used learning rate `0.0001`, batch size `80`, and `40` epochs. With
three DDP processes, this configuration uses one image per GPU and accumulates
27 micro-batches, producing an effective global batch of 81:

```text
1 image/GPU x 3 GPUs x 27 accumulation steps = 81 images/update
```

Train the image branch first:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/512_user1.yaml
```

Then train the event-conditioned branch. It loads the best image checkpoint
automatically:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config configs/rgbe_gaze/512_user1.yaml
```

Both commands resume from `last.pt` by default. Add `--no-resume` only when a
fresh run is intended. Do not pass `--device` to `torchrun`; physical GPUs
1, 2, and 4 become local CUDA devices 0, 1, and 2 inside the DDP processes.

## 6. Visualize the image branch

After the image stage finishes, generate four unconditional face examples from
its best checkpoint:

```bash
python scripts/sample_image_branch.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --num-samples 4 \
  --batch-size 1 \
  --seed 44
```

The command writes individual PNG files, `grid.png`, `reference_grid.png`, and
`metadata.json` under `runs/512-user1/image/samples/seed-44/`. The reference
grid contains real training images for qualitative context, but its entries are
not paired with the generated images. These outputs represent the learned face
distribution, not event-conditioned reconstructions. Sampling uses one GPU and
does not require DDP.

## 7. Reconstruct the test split

The fixed seed makes this preliminary evaluation reproducible:

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --limit 100 \
  --seed 44
```

Use `--limit -1` to reconstruct every sample in `test.csv`.

## 8. Compute MSE, SSIM, and PSNR

```bash
python scripts/evaluate_rgbe_metrics.py \
  --samples-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/512-user1
```

The command writes per-image values to `metrics/per_image_metrics.csv` and
aggregate statistics to `metrics/summary.json`. MSE is better when lower; SSIM
and PSNR are better when higher. For a paper, report the test sample count and
mean plus standard deviation for every metric.

## 9. Preliminary 256x256 memory test

The existing `256_user1.yaml` configuration remains available for comparison.
Its checkpoints and samples use separate output directories, so they cannot
overwrite the 512x512 experiment.

## 10. Multi-user training after the dataset is complete

Once enough users are available, prefer identity-disjoint train, validation,
and test splits. The general configuration can run on GPUs 1, 2, and 4:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/256.yaml
```

Repeat with `--stage conditional`. Revisit the global batch configuration when
changing the number of GPUs or the per-GPU batch size.
