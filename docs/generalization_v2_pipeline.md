# V2 cross-user generalization pipeline

This protocol measures whether V2 learns a reusable face prior and how quickly
its conditional branch adapts to unseen identities. It is isolated from all
existing `user_1`, baseline, and V2 runs.

## Experimental design

The reduced protocol is the initial experiment:

| Phase | Users | Train | Validation | Sampling |
|---|---:|---|---|---:|
| Generic image branch | 1-50 | `exp1` | `exp5` | every fifth training pair |
| Generic conditional branch | 1-50 | `exp1` | `exp5` | every fifth training pair |
| Specific conditional branch | 51-66 | `exp2` | `exp5` | every fifth training pair |

Validation always uses every available paired sample from `exp5`. A requested
user is excluded from all splits when the required training experiment has no
valid pair. Missing users and excluded users are recorded in `summary.json`.

The full protocol is already supported. It changes training to `exp1` through
`exp4` and uses every pair (`stride=1`), while retaining `exp5` for validation.

## Transfer behavior

The specific run loads:

- the frozen generic image branch from `generic-1-50/image/checkpoints/best.pt`;
- the generic ControlNet weights from
  `generic-1-50/conditional/checkpoints/best.pt`;
- a new optimizer and a new output directory for users 51-66.

Only the conditional branch is fine-tuned. Resume checkpoints from the specific
run take precedence after its first epoch.

## Output isolation

Reduced outputs:

```text
runs/generalization-v2/reduced/generic-1-50/
runs/generalization-v2/reduced/specific-51-66/
samples/generalization-v2/reduced/specific-51-66/
manifests/generalization-v2/reduced/
```

The full protocol uses the same layout under `generalization-v2/full/`. No path
overlaps with the existing `runs/512-user1*` experiments.

## Preview the pipeline

Inside the container, run a dry run first:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --gpus 1,2,4
```

This prints the three DDP commands without creating manifests or starting
training.

## Prepare manifests only

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps prepare \
  --execute
```

Inspect the resulting user selection before training:

```bash
cat /app/Rislab_Event_influence_volume/rgbe-gaze/manifests/generalization-v2/reduced/summary.json
```

The summary lists included users and `excluded_requested_users`. The latter
contains users that are absent or have no valid pair in the required training
experiment.

## Run the reduced training pipeline

Run all training phases sequentially on GPUs 1, 2, and 4:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --gpus 1,2,4 \
  --execute
```

The sequence is:

1. prepare generic and specific manifests;
2. train the generic image branch for 40 epochs;
3. train the generic conditional branch for 40 epochs;
4. initialize and fine-tune the specific conditional branch for 40 epochs.

Each stage resumes from its own `last.pt` when rerun. Individual phases can
also be resumed explicitly:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-conditional \
  --gpus 1,2,4 \
  --execute
```

Existing manifests are never replaced implicitly. A second full-pipeline call
stops at preparation; resume the desired training phase as shown above. Use
`--force-prepare` only when intentionally defining a new dataset snapshot.

## Checkpoints and training metrics

Both generic and specific configurations save:

- `last.pt` every epoch;
- `best.pt` when deterministic validation loss improves;
- model-only snapshots at epochs 5, 10, 15, ..., 40;
- `epoch_metrics.csv` with train loss, validation loss, best-model status, and
  snapshot path for every epoch.

Saving every five epochs is preferable to saving every five samples: the latter
would create thousands of large checkpoints and substantially slow training.

## Measure adaptation speed

After specific fine-tuning finishes, evaluate the generic zero-shot model
(`epoch 0`) and every five-epoch specific snapshot on the same balanced subset
of `exp5`:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-evaluate \
  --sampling-device 1 \
  --samples-per-user 1 \
  --execute
```

This reconstructs one validation pair per included user with a fixed diffusion
seed, then computes MSE, SSIM, and PSNR. Increase `--samples-per-user` for the
final analysis. Sampling uses 1,000 reverse-diffusion steps per image, so this
evaluation is intentionally separate from training. To inspect only selected
checkpoints, add for example `--checkpoint-epochs 0 5 10 20 40`. The aggregate
curve is written to:

```text
samples/generalization-v2/reduced/specific-51-66/
└── validation-checkpoints/
    ├── epoch_0000_generic/
    ├── epoch_0005/
    ├── ...
    ├── epoch_0040/
    └── checkpoint_metrics.csv
```

Use epoch 0 as the zero-shot reference. The minimum acceptable adaptation epoch
should be selected using a predeclared SSIM/PSNR criterion plus visual review,
not visual inspection alone.

## Run the future full-data protocol

The command is identical except for `--protocol full`:

```bash
python scripts/run_generalization_v2.py \
  --protocol full \
  --gpus 1,2,4 \
  --execute
```

Its manifests, runs, checkpoints, and samples remain separate from the reduced
protocol.
