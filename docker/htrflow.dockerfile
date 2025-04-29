ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04
ARG PYTHON_VERSION=3.10
ARG DEBIAN_FRONTEND=noninteractive

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS builder

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    python3-pip \
    python3-dev \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1
ENV UV_NO_CACHE=1

RUN uv venv --python ${PYTHON_VERSION}

# Install dependencies first (for better layer caching)
COPY uv.lock pyproject.toml /app/
RUN uv sync --frozen --no-install-project

COPY src/ /app/src/
COPY LICENSE README.md /app/

# Install project
RUN uv sync --frozen

FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu${UBUNTU_VERSION} AS runtime

ARG PYTHON_VERSION
ARG DEBIAN_FRONTEND

RUN apt-get update && apt-get install -y --no-install-recommends \
    python${PYTHON_VERSION} \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/src /app/src

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"