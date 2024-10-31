FROM huggingface/transformers-pytorch-gpu:4.41.2
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/

WORKDIR /app

RUN uv venv --python 3.10.14

ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project

COPY src LICENSE README.md examples /app/
COPY Riksarkivet__trocr-base-handwritten-hist-swe-2-onnx /app/model
COPY test_ort_pipeline.yaml /app/
COPY trocr_example.png /app/


RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN uv pip install -U optimum[onnxruntime-gpu]

ENV PATH="/app/.venv/bin:$PATH"
