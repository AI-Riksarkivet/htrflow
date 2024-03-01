# Builder stage for compiling Python
FROM ubuntu:20.04 as builder
ARG PYTHON_VERSION=3.10.4
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    zlib1g-dev \
    liblzma-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    wget

RUN wget https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz \
    && tar -xzf Python-${PYTHON_VERSION}.tgz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && cd .. && rm -rf Python-${PYTHON_VERSION}.tgz Python-${PYTHON_VERSION}

# Runtime stage with NVIDIA CUDA base
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04 as runtime

ARG PYTHON_VERSION_=3.10
ENV DEBIAN_FRONTEND=noninteractive

COPY --from=builder /usr/local /usr/local

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/local/bin/python${PYTHON_VERSION_} /usr/local/bin/python \
    && ln -sf /usr/local/bin/python${PYTHON_VERSION_} /usr/local/bin/python3 \
    && python${PYTHON_VERSION_} -m ensurepip \
    && python${PYTHON_VERSION_} -m pip install --upgrade pip

WORKDIR /code
COPY . .

RUN python -m pip install poetry \
    && poetry config virtualenvs.create true \
    && poetry install --all-extras

# CMD ["poetry", "run", "your-application-start-command"]
