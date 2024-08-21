FROM airiksarkivet/cuda-12-py310:0.0.1
WORKDIR /app
COPY dist/htrflow_core-0.0.1-py3-none-any.whl /tmp/
RUN python3 -m pip install /tmp/htrflow_core-0.0.1-py3-none-any.whl[local_models,cli] \
    && rm /tmp/htrflow_core-0.0.1-py3-none-any.whl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cd /app \
    && mkdir outputs input pipeline logs models

# CMD ["htrflow", "--help"]
# docker build -t airiksarkivet/htrflow:v0.1.0 -f docker/Dockerfile .
# docker push airiksarkivet/htrflow:v0.1.0
# docker run --gpus all -it --rm htrflow_im:latest /bin/bash
# docker exec -it <c_id> /bin/bash