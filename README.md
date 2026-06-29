# Ebird Diffusion

Conditional DDPM reconstruction of grayscale face images from accumulated event
representations. The current implementation supports RGBE-Gaze at 256x256,
single-GPU device selection, AMP, gradient checkpointing, and multi-GPU DDP.

This repository derives from Fabian Valderrama's `Ebird_MNIST` work. The
`fv-mnist` branch preserves that version, while `develop` contains the Docker
environment and the modular RGBE-Gaze pipeline. The archived MNIST code remains
under `src/`.

## Pipeline

1. Build paired target/event manifests.
2. Train the unconditional image DDPM.
3. Freeze the image branch and train event conditioning.
4. Reconstruct held-out targets and evaluate MSE, SSIM, and PSNR.

## Server requirements

- Docker Engine.
- NVIDIA GPU and driver compatible with CUDA 12.1.
- NVIDIA Container Toolkit.
- Storage for datasets, checkpoints, logs, and generated samples.

The image uses Ubuntu 20.04, Python 3.10, Miniforge, PyTorch 2.1, and CUDA 12.1.
It starts Bash by default and does not launch training automatically.

## Build and run

```bash
docker build -t ignacio_event_ebird .
./run_docker.sh
```

The script mounts:

| Host | Container | Purpose |
|---|---|---|
| `/home/ignacio.bugueno/cachefs/datasets/processed_data/reconstruction` | `/app/Rislab_Event_influence_volume/dataset` | Read-only input data |
| `/home/ignacio.bugueno/cachefs/event_reconstruction/output/ebird` | `/app/Rislab_Event_influence_volume` | Checkpoints, logs, samples, and reports |

Edit only the host-side paths in `run_docker.sh` when moving to another server.
The container exposes all GPUs; each command chooses its GPU with `--device`.

## RGBE-Gaze data

```text
/app/Rislab_Event_influence_volume/dataset/rgbe-gaze/
├── gray_frames/
│   └── user_N/expM/*.png
└── event_accumulate_frames/
    └── user_N/expM/*.png
```

A target and its event representation must share the same relative path and
filename. Unpaired files are ignored by default and reported by the manifest
builder. Input PNG files may remain at 512x512; the loader resizes them in
memory to the configured model resolution.

For the exact `user_1` split, 256x256 training, sampling, and evaluation
commands, follow [docs/server_workflow.md](docs/server_workflow.md). Design and
metric details are documented in [docs/rgbe_gaze.md](docs/rgbe_gaze.md).

## Smoke test

Test the environment and both model branches without a real dataset:

```bash
python tests/smoke_test.py --device cuda
```

The JSON report is saved under
`/app/Rislab_Event_influence_volume/smoke_test/`, which persists through the
host output mount.

## Legacy MNIST workflow

The original MNIST/N-MNIST implementation is preserved under `src/` and has not
been translated or refactored. Its stage commands are listed in
`readme_help.txt`.

## License

This repository retains the GNU General Public License v3 from `Ebird_MNIST`.
See `LICENSE` for its terms.
