# Dataset-free smoke test

This test checks Python, CUDA, the data loaders, diffusion scheduler, and both
model branches using temporary synthetic images. It does not require a dataset.

Inside the container:

```bash
python tests/smoke_test.py
```

Require CUDA and fail when it is unavailable:

```bash
python tests/smoke_test.py --device cuda
```

The report is also written to:

```text
/app/Rislab_Event_influence_volume/smoke_test/smoke_test_report.json
```

With `run_docker.sh`, the report persists under `smoke_test/` in the host output
directory.
