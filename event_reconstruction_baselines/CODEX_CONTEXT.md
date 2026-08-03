# Codex context: E2VID and ET-Net on RGBE-Gaze

## Repository

Working directory:

```text
/home/ignacio/Projects/UOH/Event-Cameras/methods/reconstruction/ebird-diffusion/event_reconstruction_baselines
```

This is a vendored comparison package. The parent eBIRD implementation must
not be changed when running these baselines.

## Goal

Run pretrained E2VID and ET-Net on the same RGBE-Gaze face-centered validation
data used to evaluate eBIRD (Pix2Pix, DDPM generic and DDPM specific). This
package performs inference only. It does not train either baseline.

## Source revisions

- E2VID: `uzh-rpg/rpg_e2vid`, commit
  `d0a7c005f460f2422f2a4bf605f70820ea7a1e5f`.
- ET-Net: `WarranWeng/ET-Net`, commit
  `3806acdf27d3534498f9e49c38a93b1de12d9b93`.

## Data contract

The dataset-preparation repository exports `exp5` validation data with no
stride. For the specific-user protocol, the selected users are 51--66. The
export uses the existing RGBE-Gaze bounding boxes as an oracle and remaps raw
Prophesee events into a fixed square sequence ROI. It preserves one continuous
stream per user/experiment.

Required paths inside the container:

```text
/app/data/manifest.csv
/app/data/e2vid_firenet_512/user_*/exp5/events.txt
/app/data/etnet_256/user_*/exp5/events.h5
/app/data/targets_512/user_*/exp5/*.png
```

E2VID and FireNet-style text streams use a `width height` header followed by
`timestamp_seconds x y polarity`, with signed polarity. ET-Net uses HDF5
datasets `events/xs`, `events/ys`, `events/ts`, `events/ps` and synchronized
`images/imageXXXXXXXXX` datasets. The ET-Net export is 256x256; E2VID is
512x512.

## Docker lifecycle

1. Build from this directory with `./build_docker.sh`.
2. Put checkpoints in a host `weights/` directory.
3. Start an interactive one-GPU container with `GPU_ID=0 ./run_docker.sh`.
4. Inside Bash run `bash scripts/run_e2vid.sh` and/or
   `bash scripts/run_etnet.sh`.
5. Inspect `/app/outputs/**/inference.log` and copy outputs from the mounted
   host output directory.

The wrappers use `CUDA_VISIBLE_DEVICES` and pass the container-visible device
`0` to ET-Net. Therefore `GPU_ID=4` means host GPU 4 is exposed as CUDA device
0 inside the container.

## What a future Codex agent should not assume

- Building the image does not download checkpoints.
- Starting the container does not run inference automatically; it opens Bash.
- Native output filenames are not guaranteed to equal RGBE-Gaze filenames.
- Metrics require timestamp/frame alignment using `manifest.csv`; do not pair
  files by arbitrary directory order.
- ET-Net outputs must be resized to 512x512 before comparison with eBIRD.
