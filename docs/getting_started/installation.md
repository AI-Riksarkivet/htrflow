# Installation

<!-- termynal -->
```
> pip install htrflow
---> 100%
Installed
```

## With pip
Install HTRFlow with [pip](https://pypi.org/project/htrflow):
```bash
pip install htrflow
```

Verify the installation of HTRFlow with `htrflow --help`. If the installation was successful, the following message is shown:

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
│ pipeline   Run a HTRFlow pipeline                            │
╰--------------------------------------------------------------╯

```


Great! Read [Quickstart](quick_start.md) to learn to use the `htrflow pipeline` command.


!!! info "Extras"

    We natively support Huggingface and Ultraytics model, but if you want to also openmmlab models:

    ```sh
    pip install htrflow[all]" 
    ```

    > Note that this forces torch to 2.0.0, since openmmlabs depends on it for now..

!!! tip "Tip!"

    If you have not installed pytorch before it will probably take a while to install with pip. 
    To speed up the installation of HTRflow use uv!

    ```sh
    #pip install uv

    uv pip install htrflow[all]" 
    ```


## Install from source or for development

Requirements:

- [uv](https://docs.astral.sh/uv/) or pip
- Python 3.10
- With GPU: CUDA >=11.8 (can still run on CPU)

Clone this repository and run:
```sh
uv pip install -e .[all]

```
This will install the HTRflow package in a virtual environment.

```sh
source .venv/bin/activate # activate virtual environment

```


## Docker 

This guide explains how to run **HTRFlow** using Docker Compose, ensuring a consistent environment and simplifying dependency management. Follow the instructions below to set up and run the application using Docker.

!!! info "Our docker repo at docker hub:"

    [Docker Hub](https://hub.docker.com/r/airiksarkivet/htrflow)


## Prerequisites

- **Docker**: Install Docker from the [official website](https://www.docker.com/get-started).
- **Docker Compose**: Usually included with Docker installations. Verify by running `docker-compose --version`.
- **NVIDIA GPU** (Optional): If you plan to use GPU acceleration, ensure you have an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

## Docker Compose Configuration

The `docker-compose.yml` file defines the services, configurations, and volume mappings needed to run HTRFlow.

**docker-compose.yml**:

```yaml
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

### **Volume Mappings**

```yaml
volumes:
  - ./examples/images/pages:/app/input       # Input folder
  - ./output-volume:/app/outputs             # Output folder
  - ./logs-volume:/app/logs                  # Logs folder
  - ./examples/pipelines:/app/pipeline       # Pipeline configuration files
  - ./.cache:/app/models                     # Models cache
```

- **`./examples/images/pages:/app/input`**: Maps your local `examples/images/pages` directory to `/app/input` inside the container. This is where HTRFlow reads input images.
- **`./output-volume:/app/outputs`**: Maps to `/app/outputs` inside the container for output files.
- **`./logs-volume:/app/logs`**: Maps to `/app/logs` inside the container for application logs.
- **`./examples/pipelines:/app/pipeline`**: Provides pipeline configuration files to the container.
- **`./.cache:/app/models`**: Shares the models cache to avoid re-downloading.

## Setup Instructions

### 1. Create Necessary Directories

Before running the Docker container, create the directories that will be used as volumes:

```sh
mkdir -p output-volume logs-volume .cache
```

This command creates:

- **`output-volume`**: Stores output files.
- **`logs-volume`**: Stores log files.
- **`.cache`**: Caches models and data.

### 2. Build and Run the Docker Container

Use Docker Compose to build the image and start the container:

```sh
docker-compose up --build
```

- **`--build`**: Forces a rebuild of the Docker image.
- Docker Compose uses `docker-compose.yml` to set up the service and volumes.

### 4. Stop the Container

To stop the Docker container and remove resources:

```sh
docker-compose down --rmi all
```

- **`--rmi all`**: Removes all images used by services.