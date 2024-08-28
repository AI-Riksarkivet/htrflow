FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

LABEL maintainer="Swedish-National-Archives-AI-lab"

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10.4

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    wget \
    && wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && ln -sf /usr/local/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/local/bin/python3.10 /usr/local/bin/python3 \
    && python3.10 -m ensurepip \
    && python3.10 -m pip install --upgrade pip \
    && cd .. && rm -rf Python-${PYTHON_VERSION}.tgz Python-${PYTHON_VERSION} \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*