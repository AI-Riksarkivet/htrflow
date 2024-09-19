FROM huggingface/transformers-pytorch-gpu:4.41.2
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/


WORKDIR /app

RUN uv venv --python 3.10.14


ADD uv.lock /app/uv.lock
ADD pyproject.toml /app/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra openmmlab

COPY src LICENSE README.md examples /app/

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --extra openmmlab

ENV PATH="/app/.venv/bin:$PATH"