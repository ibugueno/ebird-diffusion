<h1 align="center">
  <br>
  Event-based Diffusion Modeling for Image Reconstruction.
  <br>
</h1>

<h4 align="center">Memoria para optar al título de Ingeniero Civil Industrial</a>.</h4>

## Getting Started

These instructions will give you a copy of the project up and running on
your local machine for development and testing purposes. 

## Prerequisites

Before you can run this project, ensure that you have the following installed:

- **Anaconda or Miniconda**:
Download and install either [Anaconda](https://www.anaconda.com/products/individual) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html), which provides the `conda` package and environment management system.

### Setting Up the Environment

Once Conda is installed, you can set up the environment for this project by following these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/uoh-rislab/fv_event-based_diffusion_modeling_image_reconstruction.git

2. **Create the Conda environment from the environment.yml file**:
   ```bash
   conda env create -f environment.yml

3. **Activate the newly created environment**:
   ```bash
   conda activate EVDiff

4. **Verify that the environment is activated and the dependencies are installed**:
   ```bash
   conda list
   
## Training Base U-Net 

The following code provides the training for the base Unet Model needed to get the weights given unconditional image inputs.

1. **Go to MNIST_Model folder**:
   ```bash
   cd your_route/MNIST_Model

2. **Run Train file**:
   ```bash
   python3 Train.py --config 'your_route/default.yaml'

You must remember to configure the yaml file.

## Training Conditional U-Net 

The following code provides the training for the Conditional Unet Model needed to get the weights given unconditional image inputs.

1. **Go to MNIST_Model folder**:
   ```bash
   cd your_route/MNIST_Model
   
2. **Run Conditional Train file**:
   ```bash
   python3 Conditional_Train.py --config 'your_route/default.yaml'

You must remember to configure the yaml file.

## Running the Sampling Base Unet

The following code provides the use for the sampling process.

1. **Go to MNIST_Model folder**:
   ```bash
   cd your_route/MNIST_Model
   
2. **Run sample ddpm file**:
   ```bash
   python3 sample_ddpm.py --config 'your_route/default.yaml'
   
You must remember to configure the yaml file.

## Running the Conditional Unet Sampling

The following code provides the use for the sampling process with an event condition input.

1. **Go to MNIST_Model folder**:
   ```bash
   cd your_route/MNIST_Model
   
2. **Run DualSample file**:
   ```bash
   python3 DualSample.py --config 'your_route/default.yaml'
   
You must remember to configure the yaml file and save the .pth files in default folder.

### Sample Tests



## Authors

  - **Fabian I. Valderrama**
    [Linkedin](https://www.linkedin.com/in/fabian-ignacio-valderrama-peñaloza-62650b213)



## License

This project is licensed under the [MIT License](LICENSE)

## Acknowledgments


