# RGBE-Gaze server workflow

This guide trains the 256x256 pipeline using only `user_1` on GPU 1. Source
images remain at 512x512; the data loader resizes them in memory and never
changes the PNG files.

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

## 5. Train at 256x256 on GPU 1

Train the image branch first:

```bash
python scripts/train_rgbe.py \
  --stage image \
  --device 1 \
  --config configs/rgbe_gaze/256_user1.yaml
```

Then train the event-conditioned branch. It loads the best image checkpoint
automatically:

```bash
python scripts/train_rgbe.py \
  --stage conditional \
  --device 1 \
  --config configs/rgbe_gaze/256_user1.yaml
```

Both commands resume from `last.pt` by default. Add `--no-resume` only when a
fresh run is intended.

## 6. Reconstruct the test split

The fixed seed makes this preliminary evaluation reproducible:

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/256_user1.yaml \
  --limit 100 \
  --seed 44
```

Use `--limit -1` to reconstruct every sample in `test.csv`.

## 7. Compute MSE, SSIM, and PSNR

```bash
python scripts/evaluate_rgbe_metrics.py \
  --samples-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/256-user1
```

The command writes per-image values to `metrics/per_image_metrics.csv` and
aggregate statistics to `metrics/summary.json`. MSE is better when lower; SSIM
and PSNR are better when higher. For a paper, report the test sample count and
mean plus standard deviation for every metric.

## 8. Multi-GPU training after the dataset is complete

Once enough users are available, prefer identity-disjoint train, validation,
and test splits. The general configuration can run on GPUs 1, 2, and 4:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage image \
  --config configs/rgbe_gaze/256.yaml
```

Repeat with `--stage conditional`. Do not combine `--device` with `torchrun`;
DDP assigns one visible GPU to each process.
