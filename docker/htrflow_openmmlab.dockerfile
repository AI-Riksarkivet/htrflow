FROM huggingface/transformers-pytorch-gpu:4.41.2
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/


WORKDIR /app

RUN uv venv --python 3.10.14


ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src LICENSE README.md examples /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen 

RUN uv pip install -U https://github.com/Swedish-National-Archives-AI-lab/openmim_install/raw/main/mmcv-2.0.0-cp310-cp310-manylinux1_x86_64.whl && \
    uv pip install -U mmdet==3.1.0 mmengine==0.7.2 mmocr==1.0.1 yapf==0.40.1

ENV PATH="/app/.venv/bin:$PATH"