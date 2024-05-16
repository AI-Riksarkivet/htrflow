FROM borg93/cuda-12-py310:0.0.1
WORKDIR /app
COPY ./dist/htrflow_core-0.0.1-py3-none-any.whl /tmp/
RUN python3 -m pip install /tmp/htrflow_core-0.0.1-py3-none-any.whl[local_models,cli] \
    && rm /tmp/htrflow_core-0.0.1-py3-none-any.whl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && cd /app \
    && mkdir outputs input pipeline logs models
# CMD ["htrflow", "--help"]
# docker build -t borg93/htrflow_im:0.0.1 .
# docker push borg93/htrflow_im:0.0.1
# docker run --gpus all -it --rm htrflow_im:latest /bin/bash
# docker exec -it <c_id> /bin/bash