FROM airiksarkivet/cuda-12-py310:0.0.1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_SYSTEM_PYTHON=1

WORKDIR /app

RUN uv pip install htrflow"[all,openmmlab]"

# ADD uv.lock /app/uv.lock
# ADD pyproject.toml /app/pyproject.toml

# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --all-extras --frozen --no-install-project

# ADD . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --frozen --all-extras

# ENV PATH="/app/.venv/bin:$PATH"

CMD ["uv", "run", "htrflow", "--help"]
