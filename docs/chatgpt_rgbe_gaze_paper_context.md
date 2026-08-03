# RGBE-Gaze Reconstruction Context for ChatGPT Paper Drafting

This document is intended as a factual context file for ChatGPT when drafting a
workshop paper submission for ECCV. It summarizes the two reconstruction
pipelines currently used on RGBE-Gaze:

- DDPM / eBIRD-style diffusion reconstruction
- Pix2Pix grayscale image translation

Use this file as a source of implementation and protocol details. If a result,
number, or claim is not explicitly stated here, ChatGPT should treat it as
unknown and ask for confirmation rather than inventing it.

## Scope

These experiments focus on face reconstruction from event-camera data in the
RGBE-Gaze dataset.

The task is:

- input: accumulated event image
- target: synchronized grayscale face image

The broader goal is to study how well event-based inputs can reconstruct facial
appearance, geometry, and pose, and whether a diffusion model can first learn a
generic representation over many users and then adapt to unseen users with
limited fine-tuning.

## Repository Locations

Main DDPM repository:

```text
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/ebird-diffusion
```

Related Pix2Pix repository:

```text
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/event-based_reconstruction_pix2pix
```

Pix2Pix RGBE-Gaze submodule:

```text
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/event-based_reconstruction_pix2pix/pix2pix_rgbe_gaze
```

## Dataset and Preprocessing

The processed RGBE-Gaze dataset is organized as paired image folders. The two
modalities are:

- `event_accumulate_frames`: accumulated event representations
- `gray_frames`: grayscale face targets

Expected structure:

```text
rgbe-gaze/
├── event_accumulate_frames/
│   └── user_<id>/exp<id>/*.png
├── gray_frames/
│   └── user_<id>/exp<id>/*.png
└── metadata_user_<id>_exp<id>.csv
```

Pairing rule:

- an event image and a grayscale target are considered a pair when they share
  the same relative path and filename under their respective roots

Filename pattern:

```text
user_<id>_exp<id>_frame_<frame_idx>_rgbts_<cpu_timestamp>_evts_<event_start>-<event_end>_bbox_<x1>-<y1>-<x2>-<y2>_size_512_winms_33_cropkb_<size>.png
```

Confirmed preprocessing properties:

- the RGB modality is converted to grayscale
- the event modality is represented as an accumulated event image
- event and frame synchronization uses a 33 ms temporal window
- face crops are square and resized to `512 x 512`
- event crops with very small content were filtered during dataset generation
- unpaired files are skipped by default during dataset discovery and manifest
  building

## Common Experimental Split

The main split convention used for RGBE-Gaze is:

- `exp1` to `exp4`: training
- `exp5`: validation
- `exp6`: test

Important methodological rule:

- `exp6` is reserved for held-out testing and is not used for training,
  checkpoint selection, or hyperparameter tuning

## Evaluation Metrics

Both DDPM and Pix2Pix use the same reconstruction metrics:

- MSE
- SSIM
- PSNR

Metrics are computed by comparing the generated reconstruction against the
ground-truth grayscale target.

For visual inspection, the standard comparison layout is:

```text
event input | generated reconstruction | target grayscale image
```

## DDPM / eBIRD Pipeline

### General Description

The DDPM implementation is an eBIRD-style two-stage pipeline:

1. an image branch is trained first
2. a conditional branch is trained second using event inputs

The conditional branch plays the role of a ControlNet-style adaptation module.
In the generalization experiments, the image branch can be kept fixed while
only the conditional branch is fine-tuned for new users.

### Relevant DDPM Scripts

Main scripts in the DDPM repository:

- `scripts/build_rgbe_manifest.py`
- `scripts/train_rgbe.py`
- `scripts/sample_rgbe.py`
- `scripts/sample_image_branch.py`
- `scripts/evaluate_rgbe_metrics.py`
- `scripts/compose_rgbe_triplets.py`
- `scripts/run_generalization_v2.py`

### Baseline Single-User DDPM Configuration

Single-user baseline configs include:

- `configs/rgbe_gaze/512_user1.yaml`
- `configs/rgbe_gaze/512_user1_v2_40.yaml`

Confirmed baseline training settings for `512_user1.yaml`:

- image size: `512`
- input channels: `1`
- UNet channels: `[32, 64, 128, 256, 512]`
- attention resolutions: `[32, 16]`
- time embedding dimension: `256`
- number of attention heads: `4`
- diffusion timesteps: `1000`
- beta schedule: `0.0001` to `0.02`
- image branch epochs: `40`
- conditional branch epochs: `40`
- learning rate: `1e-4`
- mixed precision: enabled
- batch size per GPU: `1`
- gradient accumulation steps: `27`
- effective batch size: `27` per optimization step if using one GPU, or scaled
  further in multi-GPU distributed training
- seed: `44`

### DDPM V2 Configuration

The V2-style configuration used for higher-capacity RGBE-Gaze experiments is
represented by `configs/rgbe_gaze/512_user1_v2_40.yaml`.

Confirmed V2 changes relative to the simpler baseline:

- UNet channels become `[32, 64, 128, 256, 512, 512]`
- image branch epochs remain `40`
- conditional branch epochs remain `40`
- learning rate remains `1e-4`
- mixed precision remains enabled
- batch size per GPU becomes `3`
- gradient accumulation steps become `9`
- checkpointing can be enabled at intermediate epochs

Conceptually, V2 increases model depth and preserves a smaller latent spatial
representation, making it more suitable for `512 x 512` facial reconstruction.

### DDPM Training Workflow

The standard workflow is:

1. create manifests
2. train image branch
3. optionally inspect unconditional image-branch samples
4. train conditional branch
5. generate conditional reconstructions on validation or test data
6. compute MSE, SSIM, and PSNR

Image-branch outputs are not paired reconstructions. They are unconditional
samples used mainly to inspect whether the image prior has learned a plausible
facial distribution.

Conditional-branch outputs are the actual event-to-image reconstructions used
for reporting and comparison.

### DDPM Generalization Study

The main DDPM generalization study is implemented in:

```text
docs/generalization_v2_pipeline.md
scripts/run_generalization_v2.py
```

Its purpose is to test whether a generic model trained on a group of users can
be adapted to new users by fine-tuning only the conditional branch.

#### Reduced Protocol

The `reduced` protocol is defined as:

- generic users: `1-50`
- specific users: `51-66`
- generic training: `exp1`, every 5th sample
- specific fine-tuning: `exp2`, every 5th sample
- validation: `exp5`
- test: `exp6`

In this protocol:

- the generic model trains an image branch and a conditional branch
- the specific model reuses the generic image branch
- the specific model initializes its conditional branch from the generic
  conditional checkpoint
- only the conditional branch is fine-tuned for the new users

#### Stride5-All Protocol

The `stride5_all` protocol extends training data while keeping the reduced
sampling rate:

- training experiments: `exp1`, `exp2`, `exp3`, `exp4`
- training stride: every 5th sample
- validation: `exp5`, every 5th sample
- test: `exp6`

This protocol exists to test whether using all four training experiments while
still subsampling improves the generic and specific diffusion models.

#### Full Protocol

The `full` protocol is the future full-data setting:

- training experiments: `exp1` to `exp4`
- all available samples are used
- validation: `exp5`
- test: `exp6`

### DDPM Checkpointing and Evaluation Strategy

For the generalization experiments:

- generic model checkpoints are stored separately from specific ones
- specific conditional checkpoints are saved every 5 epochs
- validation is performed on `exp5`
- final held-out evaluation is performed on `exp6`

The specific-user branch is intended to be inspected both quantitatively and
visually across intermediate epochs, in order to estimate the minimum number of
fine-tuning epochs needed for acceptable reconstruction quality.

## Pix2Pix Pipeline

### General Description

The Pix2Pix pipeline is implemented as a separate RGBE-Gaze module and is
intended as a direct supervised image-translation baseline from accumulated
event images to grayscale targets.

It is isolated from the older `pix2pix_unet_faces` code and uses the same
processed RGBE-Gaze dataset structure described above.

### Pix2Pix Architecture

Confirmed generator:

- grayscale-to-grayscale U-Net
- eight encoder levels for `512 x 512` inputs
- seven explicit encoder blocks plus a bottleneck
- transposed-convolution decoder with skip connections
- output activation: `tanh`

Confirmed discriminator:

- `70 x 70`-style PatchGAN discriminator
- no final sigmoid in the discriminator definition

This makes Pix2Pix the direct paired-translation baseline, whereas the DDPM
pipeline separates image prior learning from conditional event guidance.

### Pix2Pix Training Configuration

From `pix2pix_rgbe_gaze/yaml/config_dgx-1.yaml`:

- dataset root inside container: `/app/input/rgbe-gaze`
- output root inside container: `/app/output/pix2pix_rgbe_gaze`
- image size: `512`
- batch size: `16`
- epochs: `100`
- checkpoint saving frequency: every `10` epochs
- number of workers: `4`
- maximum validation batches: `100`
- learning rate: `0.0002`
- Adam `beta1`: `0.5`
- L1 loss weight: `100.0`
- mixed precision: enabled
- seed: `44`
- strict pair checking: disabled by default
- default user list: `user_1`
- train split: `exp1` to `exp4`
- validation split: `exp5`
- test split: `exp6`

### Pix2Pix Workflow

Main scripts:

- `check_dataset.py`
- `train.py`
- `sample.py`
- `evaluate.py`

||
1. validate dataset structure and pair discovery
2. train Pix2Pix on training experiments
3. select the best checkpoint using validation
4. evaluate on `exp6`
5. export triplet visualizations and global metrics

Pix2Pix evaluation writes:

- per-image metrics in CSV format
- a summary JSON file
- event, generated, and target images
- horizontal comparison images

The evaluation code is intended to report global MSE, SSIM, and PSNR over the
selected split, and its summary format has been aligned with the DDPM style so
that mean, standard deviation, median, minimum, and maximum can be reported for
each metric when available.

## Relationship Between the Two Pipelines

The DDPM and Pix2Pix models are complementary baselines rather than redundant
implementations.

DDPM / eBIRD emphasizes:

- a generative image prior
- a separate conditional event-guidance stage
- user generalization and conditional fine-tuning experiments
- analysis of how much adaptation is needed for new identities

Pix2Pix emphasizes:

- direct paired translation
- simpler supervised training
- a conventional encoder-decoder GAN baseline
- a stronger point of comparison for deterministic reconstruction quality

For the paper, the main comparative framing can therefore be:

- diffusion-based event-to-face reconstruction
- direct adversarial paired translation
- quantitative comparison with MSE, SSIM, and PSNR
- qualitative comparison through standardized triplet visualizations
- optional analysis of generic-to-specific adaptation in DDPM

## Important Methodological Details for the Paper

The following points are safe to state, because they reflect the current
implementation and workflow:

- both methods use paired accumulated-event and grayscale face images
- both methods operate on `512 x 512` crops
- both use `exp5` for validation and `exp6` for held-out testing
- both use MSE, SSIM, and PSNR for evaluation
- visual comparisons are saved as `event | generated | target`
- DDPM supports generic-to-specific adaptation by reusing the generic image
  branch and fine-tuning only the conditional branch for new users
- Pix2Pix serves as a direct paired image-translation baseline

The following points should only be stated if accompanied by actual measured
results from logs or exported summaries:

- one method outperforms the other
- V2 is better than the baseline
- the generic DDPM model generalizes successfully to unseen users
- a specific number of fine-tuning epochs is sufficient
- any absolute performance claim on the full test set

## Suggested Paper Framing

A reasonable workshop framing is:

- reconstruct grayscale face images from accumulated event inputs on RGBE-Gaze
- compare a diffusion-based method against a Pix2Pix baseline
- report both quantitative metrics and qualitative triplets
- study whether the diffusion model can be trained generically and then adapted
  to new users by fine-tuning only the conditional branch

Possible contribution wording, subject to final results:

- a practical RGBE-Gaze reconstruction benchmark setup for event-to-face
  synthesis
- an implementation of a two-stage diffusion pipeline for event-guided face
  reconstruction
- a controlled comparison against a Pix2Pix baseline
- an analysis of generic pretraining plus user-specific adaptation

## What ChatGPT Should Not Assume

ChatGPT should not assume:

- the exact number of final training samples unless provided from manifest
  counts
- final metric values unless copied from actual summary files or logs
- that the same protocol was used in every run
- that the baseline and V2 always use the same model depth
- that Pix2Pix and DDPM were trained on the exact same subset unless the run
  configuration explicitly confirms it

## Files to Inspect for Exact Numbers

If exact wording, run details, or sample counts are needed, inspect these files
before drafting the final paper text:

- `ebird-diffusion/docs/generalization_v2_pipeline.md`
- `ebird-diffusion/docs/server_workflow.md`
- `ebird-diffusion/configs/rgbe_gaze/*.yaml`
- `ebird-diffusion/configs/rgbe_gaze/generalization_v2/*.yaml`
- `event-based_reconstruction_pix2pix/pix2pix_rgbe_gaze/README.md`
- `event-based_reconstruction_pix2pix/pix2pix_rgbe_gaze/yaml/config_dgx-1.yaml`
- exported `summary.json` metric files from DDPM and Pix2Pix evaluations
- exported `per_image_metrics.csv` files
- DDPM training logs and Pix2Pix `metrics.csv`
