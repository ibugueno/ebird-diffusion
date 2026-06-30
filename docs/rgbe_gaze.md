# RGBE-Gaze training at 512x512

The RGBE-Gaze implementation is independent from the archived MNIST scripts in
`src/`. Its three stages are:

1. train the image DDPM branch;
2. freeze that branch and train event conditioning;
3. reconstruct faces from accumulated-event representations.

## Expected data layout

```text
/app/Rislab_Event_influence_volume/dataset/rgbe-gaze/
├── gray_frames/user_N/expM/*.png
└── event_accumulate_frames/user_N/expM/*.png
```

Each target/event pair must have the same relative path and filename. The
loader converts both images to grayscale, resizes them to the configured
resolution, and scales pixels to `[-1, 1]`.

## Split strategies

For the current `user_1` experiment, reserve complete recordings:

```bash
python scripts/build_rgbe_manifest.py \
  --dataset-root /app/Rislab_Event_influence_volume/dataset/rgbe-gaze \
  --output-dir /app/Rislab_Event_influence_volume/rgbe-gaze/manifests-user-1-split \
  --users user_1 \
  --val-experiments exp5 \
  --test-experiments exp6
```

This is suitable for checking whether the model learns one identity. It is not
evidence of generalization to new people. Once all users are available, omit
`--users` and the experiment flags to create identity-disjoint splits using
`--val-ratio` and `--test-ratio`.

## Training and reconstruction

Use `configs/rgbe_gaze/512_user1.yaml` for the recommended single-user baseline
and follow [server_workflow.md](server_workflow.md) for exact commands. The
image branch must finish before the conditional branch.

Use `scripts/sample_image_branch.py` after the image stage to inspect
unconditional samples from its best checkpoint. The script saves individual
images, a generated grid, and an unpaired real-image reference grid. It does
not use event inputs because conditioning is learned in the second stage.

The successful baseline uses five U-Net levels, 40 epochs per stage, batch size
1 per GPU, and 27 gradient-accumulation steps. The optional V2 configuration is
documented separately at the end of the server workflow. Its isolated
`512_user1_v2_40.yaml` experiment also runs for 40 epochs, allowing a controlled
architecture comparison before considering the independent 80-epoch V2 run.

## Reconstruction metrics

`scripts/evaluate_rgbe_metrics.py` compares each `generated_*.png` with its
corresponding `target_*.png` and reports:

- MSE: pixel error; lower is better.
- SSIM: structural similarity; higher is better.
- PSNR in dB: signal-to-error ratio; higher is better.

All metrics operate on grayscale images scaled to `[0, 1]`. Evaluate only the
held-out test split. Use a fixed diffusion seed for reproducibility and report
the number of samples, mean, and standard deviation. A later paper experiment
should additionally use unseen identities and, ideally, multiple sampling seeds.
