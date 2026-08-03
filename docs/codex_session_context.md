# Codex session context: eBIRD / RGBE-Gaze reconstruction

This document summarizes the working context for a new Codex session.

## Main repository

Local path:

```bash
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/ebird-diffusion
```

Remote repository:

```bash
https://github.com/ibugueno/ebird-diffusion.git
```

Main working branch:

```bash
develop
```

Original upstream/reference repository:

```bash
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/fv_event-based_diffusion_modeling_image_reconstruction
```

Reference MNIST/eBIRD repository:

```bash
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/Ebird_MNIST
```

Related Pix2Pix repository:

```bash
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/event-based_reconstruction_pix2pix
```

## Goal

Train and evaluate event-based image reconstruction models for RGBE-Gaze faces.

The input is an accumulated event image and the target is the synchronized grayscale RGB frame crop. The main reconstruction model is a DDPM/eBIRD-style pipeline with:

- image branch;
- conditional branch / ControlNet-style branch;
- RGBE-Gaze-specific dataset loaders, manifests, sampling and metrics;
- support for generic models and user-specific fine-tuning.

The current scientific objective is to evaluate whether V2 can learn a generic model over many users and then adapt to unseen/new users with only ControlNet/conditional fine-tuning, without retraining the full image branch.

## Docker / server mount assumptions

Inside the DDPM/eBIRD container:

```bash
/app
/app/Rislab_Event_influence_volume
/app/Rislab_Event_influence_volume/dataset/rgbe-gaze
```

Host paths on the DGX/server:

```bash
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird
/home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction
```

Typical Docker volume mapping:

```bash
-v /home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird:/app/Rislab_Event_influence_volume
-v /home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction:/app/Rislab_Event_influence_volume/dataset:ro
```

Dataset path inside container:

```bash
/app/Rislab_Event_influence_volume/dataset/rgbe-gaze
```

Output path inside container:

```bash
/app/Rislab_Event_influence_volume/rgbe-gaze
```

Equivalent host output root:

```bash
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird/rgbe-gaze
```

## RGBE-Gaze processed dataset structure

Expected structure:

```text
processed_data/
└── reconstruction/
    └── rgbe-gaze/
        ├── gray_frames/
        │   ├── user_1/
        │   │   ├── exp1/
        │   │   ├── exp2/
        │   │   └── ...
        │   └── ...
        ├── event_accumulate_frames/
        │   ├── user_1/
        │   │   ├── exp1/
        │   │   ├── exp2/
        │   │   └── ...
        │   └── ...
        ├── metadata_user_1_exp1.csv
        ├── metadata_user_1_exp2.csv
        └── ...
```

Pairing convention:

- `event_accumulate_frames/.../<filename>.png` is the input.
- `gray_frames/.../<same filename>.png` is the target.
- If a pair is incomplete, the manifest builder should ignore unpaired samples rather than fail.

File naming format:

```text
user_<id>_exp<id>_frame_<frame_idx>_rgbts_<cpu_timestamp>_evts_<event_start>-<event_end>_bbox_<x1>-<y1>-<x2>-<y2>_size_512_winms_33_cropkb_<size>.png
```

Dataset summary:

- RGB target is grayscale.
- Event input is accumulated event representation.
- Current final size is 512x512.
- Event window is 33 ms.
- Face crop is square and resized to 512x512.
- Samples are filtered during dataset generation by cropped event PNG size, usually `>= 17 kB`.

## Important implemented scripts

Main training:

```bash
scripts/train_rgbe.py
```

Manifest generation:

```bash
scripts/build_rgbe_manifest.py
```

Conditional sampling:

```bash
scripts/sample_rgbe.py
```

Image branch sampling:

```bash
scripts/sample_image_branch.py
```

Metrics:

```bash
scripts/evaluate_rgbe_metrics.py
```

Triplet composition:

```bash
scripts/compose_rgbe_triplets.py
```

Generalization pipeline:

```bash
scripts/run_generalization_v2.py
```

## Metrics

The reconstruction metrics used are:

- MSE
- PSNR
- SSIM

Metrics are generated from `generated_*.png` against `target_*.png`.

Typical output files:

```text
metrics/summary.json
metrics/per_image_metrics.csv
```

The triplet visualization order should be:

```text
input event image | generated reconstruction | target grayscale frame
```

## Baseline single-user configs

Important configs:

```bash
configs/rgbe_gaze/256_smoke.yaml
configs/rgbe_gaze/256_user1.yaml
configs/rgbe_gaze/512_user1.yaml
configs/rgbe_gaze/512_user1_v2_40.yaml
```

The baseline architecture produced visually good results for 512x512 faces after 40 epochs, especially with the conditional branch. Avoid accidentally overwriting baseline outputs.

Known good sample command:

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --limit 8 \
  --seed 44
```

Known image branch sample command:

```bash
python scripts/sample_image_branch.py \
  --device 1 \
  --config configs/rgbe_gaze/512_user1.yaml \
  --num-samples 4 \
  --batch-size 1 \
  --seed 44
```

## GPU usage

Preferred server GPUs for DDPM/eBIRD experiments:

```bash
1,2,4
```

Distributed training pattern:

```bash
CUDA_VISIBLE_DEVICES=1,2,4 \
torchrun --standalone --nproc_per_node=3 scripts/train_rgbe.py \
  --stage conditional \
  --config <config.yaml>
```

Inside the code, `--device 1` means selecting a visible CUDA device for non-distributed scripts. For distributed training, use `CUDA_VISIBLE_DEVICES`.

## Generalization V2 experiment plan

Primary experiment design:

1. Train a generic model with users 1-50.
2. Train image branch for 40 epochs.
3. Train conditional branch for 40 epochs.
4. Validate with exp5.
5. Test with exp6 only after training.
6. Fine-tune a specific ControlNet/conditional model for users 51-66.
7. Use generic conditional weights as initialization.
8. Fine-tune only the conditional branch for 40 epochs.
9. Save checkpoints every 5 epochs.
10. Evaluate specific checkpoints every 5 epochs to estimate how many epochs are needed for acceptable reconstruction.

Important rule:

```text
Test exp6 is not used during training, best checkpoint selection, or hyperparameter tuning.
```

## Generalization protocols

The pipeline has these protocols:

### reduced

- Generic users: 1-50.
- Generic train: exp1, every 5 samples.
- Specific users: 51-66.
- Specific train: exp2, every 5 samples.
- Validation: exp5.
- Test: exp6.

### stride5_all

New planned experiment:

- Generic users: 1-50.
- Specific users: 51-66.
- Train: exp1, exp2, exp3, exp4.
- Train sampling: every 5 samples.
- Validation: exp5, every 5 samples.
- Test: exp6.

This protocol should store results in a separate folder, so it does not overwrite `reduced`.

### full

Future option:

- Train: exp1, exp2, exp3, exp4.
- Use all samples.
- Validation: exp5.
- Test: exp6.

## Generalization V2 output locations

Inside container:

```bash
/app/Rislab_Event_influence_volume/rgbe-gaze/manifests/generalization-v2/<protocol>
/app/Rislab_Event_influence_volume/rgbe-gaze/runs/generalization-v2/<protocol>
/app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/<protocol>
```

On host:

```bash
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird/rgbe-gaze/manifests/generalization-v2/<protocol>
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird/rgbe-gaze/runs/generalization-v2/<protocol>
/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird/rgbe-gaze/samples/generalization-v2/<protocol>
```

For `reduced`, important trained checkpoint folders already seen on the server:

```bash
/app/Rislab_Event_influence_volume/rgbe-gaze/runs/generalization-v2/reduced/generic-1-50/conditional/checkpoints
/app/Rislab_Event_influence_volume/rgbe-gaze/runs/generalization-v2/reduced/specific-51-66/conditional/checkpoints
```

Both had:

```text
best.pt
last.pt
epoch_0005.pt
epoch_0010.pt
...
epoch_0040.pt
```

## Commands: train reduced protocol

Full reduced pipeline:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --gpus 1,2,4 \
  --force-prepare \
  --execute
```

If manifests already exist and are compatible, do not use `--force-prepare`.

## Commands: evaluate reduced protocol

Generic final model on its own users 1-50, test exp6:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Specific model over users 51-66, validation exp5, every 5 epochs:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-evaluate \
  --sampling-device 1 \
  --samples-per-user 7 \
  --validation-limit 100 \
  --checkpoint-epochs 0 5 10 15 20 25 30 35 40 \
  --execute
```

Specific final model over users 51-66, test exp6:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Expected reduced sample outputs:

```bash
/app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/generic-1-50/test-best/limit-100-per-user-7
/app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/validation-checkpoints/limit-100-per-user-7
/app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/test-best/limit-100-per-user-7
```

Each sample folder should contain:

```text
metrics/summary.json
metrics/per_image_metrics.csv
comparisons/user_*/exp*/comparison_*.png
user_*/exp*/event_*.png
user_*/exp*/generated_*.png
user_*/exp*/target_*.png
```

For checkpoint evaluation, there should also be:

```text
checkpoint_metrics.csv
epoch_0000_generic/
epoch_0005/
epoch_0010/
...
epoch_0040/
```

## Commands: train stride5_all protocol

This is the new experiment with train exp1-exp4, every 5 samples:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --gpus 1,2,4 \
  --execute
```

If incompatible or stale manifests already exist:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --gpus 1,2,4 \
  --force-prepare \
  --execute
```

Expected output isolation:

```bash
/app/Rislab_Event_influence_volume/rgbe-gaze/runs/generalization-v2/stride5_all
/app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/stride5_all
```

## Commands: evaluate stride5_all protocol

Generic final model, users 1-50, test exp6:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Specific model, users 51-66, validation exp5, every 5 epochs:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps specific-evaluate \
  --sampling-device 1 \
  --samples-per-user 7 \
  --validation-limit 100 \
  --checkpoint-epochs 0 5 10 15 20 25 30 35 40 \
  --execute
```

Specific final model, users 51-66, test exp6:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps specific-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

## Local comparison folders

Some generated outputs were copied locally under:

```bash
/home/ignacio/tmp4
```

Known folders:

```text
ddpm-512-user1
ddpm-512-user1-v2-40
pix2pix-512-user1
```

A helper script was requested/generated to stack comparisons vertically in this order:

```text
ddpm-512-user1
ddpm-512-user1-v2-40
pix2pix-512-user1
```

## Pix2Pix related context

Related repo:

```bash
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/event-based_reconstruction_pix2pix
```

Original module should not be modified:

```bash
pix2pix_unet_faces/
```

New module for RGBE-Gaze:

```bash
pix2pix_rgbe_gaze/
```

Run inside container:

```bash
cd /app/pix2pix_rgbe_gaze
```

Use `--rep`, not `--x1`, for representation selection.

Example:

```bash
python train.py \
  --device dgx-1 \
  --rep acc_events \
  --gpu 7 \
  --users user_1
```

Pix2Pix output naming was designed to include timestamp, dataset/model, representation and users, so outputs should not overwrite previous runs.

Pix2Pix sample visualization should also use:

```text
input | generated | target
```

## Git notes

Current desired behavior:

- Keep user edits in `docs/notes.md` untouched unless explicitly requested.
- Do not commit unrelated/untracked notes automatically.
- Prefer committing only relevant pipeline/code/doc files.

Useful status command:

```bash
git -C /home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/ebird-diffusion status --short --branch
```

## Style and language decisions

Code logs, print messages, comments and `.md` instructions should be in English.

Conversation with Ignacio can be in Spanish.

## Key reminders for a new Codex session

1. Do not overwrite existing baseline outputs unless explicitly requested.
2. Keep `reduced`, `stride5_all`, and `full` outputs separated by protocol folder.
3. For final scientific evaluation, use exp6 only after training.
4. Use exp5 for validation during training and checkpoint comparison.
5. For user-specific adaptation, compare specific checkpoints every 5 epochs.
6. Always generate both metrics and visual triplets.
7. Triplets must be left-to-right: input event, generated, target.
8. If manifests already exist and match the current protocol, reuse them.
9. If manifests are stale or incompatible, use `--force-prepare`.
10. Preferred DDPM GPUs on the server are `1,2,4`.
