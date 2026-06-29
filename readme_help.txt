SERVER REQUIREMENTS

- Docker Engine
- NVIDIA GPU and a driver compatible with CUDA 12.1
- NVIDIA Container Toolkit
- Dataset under Rislab_Event_influence_volume/dataset


1. BUILD THE IMAGE

docker build -t ignacio_event_ebird .


2. START AN INTERACTIVE CONTAINER

The host output directory receives checkpoints, logs, generated samples, and
reports. The host dataset directory is mounted read-only.

./run_docker.sh


3. RUN THE RGBE-GAZE WORKFLOW

Follow docs/server_workflow.md for manifest generation, 256x256 training,
sampling, and MSE/SSIM/PSNR evaluation.


4. RUN THE ARCHIVED MNIST WORKFLOW

The original implementation remains under src/:

python src/Train_Image_Branch.py --config src/default.yaml
python src/Train_Conditional_Partition.py --config src/default.yaml
python src/DualSample_ajustable_evalgen_boost.py --config src/default.yaml


5. CHECK THE GPU INSIDE THE CONTAINER

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
