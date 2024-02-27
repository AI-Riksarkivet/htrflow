FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
#nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    python3.10 \
    python3-pip \
    git \
    ffmpeg \
    && ln -s /usr/bin/python3.10 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY . .

RUN pip install --upgrade pip && pip install poetry

RUN poetry install --extras "openmmlab"

# CMD ["poetry", "run", "python", "./src/htrflow_core/models/openmmlab/openmmlab_loader.py"]


## Usage
# docker build -t htrflow_core:latest .
# docker run --gpus all -it --rm --name htrflow htrflow_core:latestk

