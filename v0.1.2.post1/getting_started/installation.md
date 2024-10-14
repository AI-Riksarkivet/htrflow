# Installation

<!-- termynal -->
```
> pip install htrflow
---> 100%
Installed
```

## With pip
Install HTRflow with [pip](https://pypi.org/project/htrflow):
```bash
pip install htrflow
```

Requirements:

- Python >=3.10 and <3.13 (Python 3.10 is required for OpenMMLab)
- With GPU: CUDA >=11.8 (required due to PyTorch 2.0, can still run on CPU)

Verify the installation of HTRflow with `htrflow --help`. If the installation was successful, the following message is shown:

<!-- termynal -->
```
> htrflow --help

 Usage: htrflow [OPTIONS] COMMAND [ARGS]...

 CLI inferface for htrflow

╭- Options ----------------------------------------------------╮
│ --help          Show this message and exit.                  │
╰--------------------------------------------------------------╯
╭- Commands ---------------------------------------------------╮
│ evaluate   Evaluate HTR transcriptions against ground truth  │
│ pipeline   Run a HTRflow pipeline                            │
╰--------------------------------------------------------------╯

```


Great! Read [Quickstart](quick_start.md) to learn to use the `htrflow pipeline` command.


!!! tip

    To speed up the installation of HTRflow use `uv`:

    ```sh
    pip install uv
    uv pip install htrflow
    ```


## From source

Requirements:

- [uv](https://docs.astral.sh/uv/) or pip
- Python 3.10
- With GPU: CUDA >=11.8 (required due to PyTorch 2.0, can still run on CPU)

Clone this repository and run:
```sh
uv pip install -e .  # or you can run: uv sync
```
This will install the HTRflow package in a virtual environment.

```sh
source .venv/bin/activate # activate virtual environment

```

## OpenMMLab

To use OpenMMLab models u need to seperatly install it, see  [Models/ OpenMMLab reference](../getting_started/models.md#openmmlab-models).


## Docker 

This guide explains how to run HTRflow using Docker Compose, ensuring a consistent environment and simplifying dependency management. Follow the instructions below to set up and run the application using Docker.

!!! info "HTRflow on Docker hub:"

    [Docker hub](https://hub.docker.com/r/airiksarkivet/htrflow)


### Prerequisites

- **Docker**: Install Docker from the [official website](https://www.docker.com/get-started).
- **Docker Compose**: Usually included with Docker installations. Verify by running `docker-compose --version`.
- **NVIDIA GPU** (Optional): If you plan to use GPU acceleration, ensure you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

### Docker compose configuration

The `docker-compose.yml` file defines the services, configurations, and volume mappings needed to run HTRflow.

```yaml title="docker-compose.yml"
version: "3.8"

services:
  htrflow:
    image: docker/htrflow.dockerfile
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

    command:
      [
        "/bin/sh",
        "-c",
        "htrflow pipeline pipeline/demo.yaml input --logfile logs/htrflow/htrflow.log",
      ]

    volumes:
      - ./examples/images/pages:/app/input      
      - ./output-volume:/app/outputs            
      - ./logs-volume:/app/logs                  
      - ./examples/pipelines:/app/pipeline       
      - ./.cache:/app/models                   
```

### Volume mappings

```yaml
volumes:
  - ./examples/images/pages:/app/input       # Input folder
  - ./output-volume:/app/outputs             # Output folder
  - ./logs-volume:/app/logs                  # Logs folder
  - ./examples/pipelines:/app/pipeline       # Pipeline configuration files
  - ./.cache:/app/models                     # Models cache
```

- **`./examples/images/pages:/app/input`**: Maps your local `examples/images/pages` directory to `/app/input` inside the container. This is where HTRflow reads input images.
- **`./output-volume:/app/outputs`**: Maps to `/app/outputs` inside the container for output files.
- **`./logs-volume:/app/logs`**: Maps to `/app/logs` inside the container for application logs.
- **`./examples/pipelines:/app/pipeline`**: Provides pipeline configuration files to the container.
- **`./.cache:/app/models`**: Shares the models cache to avoid re-downloading.

### Setup instructions

#### 1. Create necessary directories

Before running the Docker container, create the directories that will be used as volumes:

```sh
mkdir -p output-volume logs-volume .cache
```

This command creates:

- **`output-volume`**: Stores output files.
- **`logs-volume`**: Stores log files.
- **`.cache`**: Caches models and data.

#### 2. Build and run the Docker container

Use Docker Compose to build the image and start the container:

```sh
docker-compose up --build
```

- **`--build`**: Forces a rebuild of the Docker image.
- Docker Compose uses `docker-compose.yml` to set up the service and volumes.

#### 3. Stop the container

To stop the Docker container and remove resources:

```sh
docker-compose down --rmi all
```

- **`--rmi all`**: Removes all images used by services.