# V2 cross-user generalization pipeline

This protocol measures whether V2 learns a reusable face prior and how quickly
its conditional branch adapts to unseen identities. It is isolated from all
existing `user_1`, baseline, and V2 runs.

## Experimental design

The reduced protocol is the initial experiment:

| Phase | Users | Train | Validation | Test | Sampling |
|---|---:|---|---|---|---:|
| Generic image branch | 1-50 | `exp1` | `exp5` | `exp6` | every fifth train/validation pair |
| Generic conditional branch | 1-50 | `exp1` | `exp5` | `exp6` | every fifth train/validation pair |
| Specific conditional branch | 51-66 | `exp2` | `exp5` | `exp6` | every fifth train/validation pair |

The reduced protocol uses every fifth paired sample from `exp5` during
validation to reduce epoch time. `exp6` remains complete and is reserved for
final held-out reconstruction metrics. A requested user is excluded from all
splits when the required training experiment has no valid pair. Missing users
and excluded users are recorded in `summary.json`.

The full protocol is already supported. It changes training to `exp1` through
`exp4` and uses every pair (`stride=1`) for training and validation, while
retaining `exp6` for complete testing.

Three isolated protocols are available:

| Protocol | Train experiments | Train stride | Validation | Test |
|---|---|---:|---|---|
| `reduced` | Generic: `exp1`; specific: `exp2` | 5 | `exp5`, stride 5 | complete `exp6` |
| `stride5_all` | `exp1` through `exp4` | 5 | `exp5`, stride 5 | complete `exp6` |
| `full` | `exp1` through `exp4` | 1 | `exp5`, stride 1 | complete `exp6` |

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

If manifests were created with an earlier version of this pipeline, rebuild
them once so validation uses stride 5 and `test.csv` contains complete `exp6`:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps prepare \
  --force-prepare \
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
  --force-prepare \
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

Existing manifests are never replaced implicitly. A full-pipeline call reuses
them when their train, validation, and test experiments match the selected
protocol. Incompatible or older manifests require `--force-prepare`; use that
flag only when intentionally rebuilding the dataset snapshot.

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

After specific fine-tuning, evaluate the generic zero-shot model (`epoch 0`)
and every five-epoch specific snapshot on the same balanced subset of `exp5`:

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

Review the specific adaptation curve with:

```bash
cat /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/validation-checkpoints/limit-100-per-user-7/checkpoint_metrics.csv
```

The command saves the generated, target, and event images for every evaluated
checkpoint, then computes global MSE, SSIM, and PSNR summaries over the sampled
set. It also creates horizontal `Input (event) | Generated | Target`
comparisons. Sampling uses 1,000
reverse-diffusion steps per image, so this evaluation is intentionally separate
from training. The specific output is organized as:

```text
samples/generalization-v2/reduced/specific-51-66/
└── validation-checkpoints/
    └── limit-100-per-user-7/
        ├── epoch_0000_generic/
        │   ├── comparisons/
        │   └── metrics/summary.json
        ├── epoch_0005/
        ├── ...
        ├── epoch_0040/
        └── checkpoint_metrics.csv
```

Use epoch 0 as the zero-shot reference. The minimum acceptable adaptation epoch
should be selected using a predeclared SSIM/PSNR criterion plus visual review,
not visual inspection alone.

## Generate final test reconstructions and metrics

After selecting each `best.pt` using `exp5`, evaluate the models once on
held-out `exp6`. The following commands generate up to 100 samples distributed
round-robin across the included users and compute MSE, SSIM, and PSNR.

First, the generic model can be evaluated on users 1-50:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Review its test metrics and generated images:

```bash
cat /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/generic-1-50/test-best/limit-100-per-user-7/metrics/summary.json
ls /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/generic-1-50/test-best/limit-100-per-user-7
```

The `comparisons/` subtree contains the horizontal `Input (event) | Generated |
Target` images. `metrics/summary.json` contains the global metrics over the 100
generated test samples, while `metrics/per_image_metrics.csv` retains the
individual values.

Then evaluate the adapted specific model on users 51-66:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Review its test metrics and generated images:

```bash
cat /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/test-best/limit-100-per-user-7/metrics/summary.json
ls /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/test-best/limit-100-per-user-7
```

This output follows the same `comparisons/` and `metrics/` layout as the
generic model.

For the final paper evaluation, process every available `exp6` pair:

```bash
python scripts/run_generalization_v2.py \
  --protocol reduced \
  --steps specific-test \
  --sampling-device 1 \
  --test-limit -1 \
  --test-samples-per-user 0 \
  --execute
```

Use the same command with `--steps generic-test` to evaluate every generic-user
test pair.

Equivalent manual commands for the balanced 100-sample evaluation are:

```bash
python scripts/sample_rgbe.py \
  --device 1 \
  --config configs/rgbe_gaze/generalization_v2/specific_51_66_reduced.yaml \
  --split test \
  --limit 100 \
  --samples-per-user 7 \
  --seed 44 \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/test-best/limit-100-per-user-7

python scripts/evaluate_rgbe_metrics.py \
  --samples-dir /app/Rislab_Event_influence_volume/rgbe-gaze/samples/generalization-v2/reduced/specific-51-66/test-best/limit-100-per-user-7
```

The generated images, per-image CSV, and metric summary remain separate from
the validation-checkpoint outputs.

## Run the `exp1`-through-`exp4` stride-5 protocol

This experiment preserves the reduced sampling rate while using all four
training experiments. Its manifests, checkpoints, and samples are isolated
under `generalization-v2/stride5_all/`.

Preview its commands:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --gpus 1,2,4
```

Run the complete generic-plus-specific training pipeline:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --gpus 1,2,4 \
  --execute
```

Evaluate the final generic model on up to 100 balanced `exp6` samples:

```bash
python scripts/run_generalization_v2.py \
  --protocol stride5_all \
  --steps generic-test \
  --sampling-device 1 \
  --test-limit 100 \
  --test-samples-per-user 7 \
  --execute
```

Evaluate specific adaptation every five epochs:

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

## Run the future full-data protocol

The command is identical except for `--protocol full`:

```bash
python scripts/run_generalization_v2.py \
  --protocol full \
  --gpus 1,2,4 \
  --execute
```

Its manifests, runs, checkpoints, and samples remain separate from both
`reduced` and `stride5_all`.
