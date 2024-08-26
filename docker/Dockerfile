# Use the base image
FROM airiksarkivet/cuda-12-py310:0.0.1

# Set the working directory
WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy the Poetry project files
COPY . .

# Install dependencies using Poetry
RUN poetry install --no-interaction --no-root --extras "huggingface ultralytics cli"

# Run the openmmlab_install script using Poetry
RUN poetry run scripts/openmmlab_install


# Create necessary directories
RUN mkdir -p /app/output /app/input /app/logs /app/pipeline /app/models

# Set the entry point (uncomment and modify if needed)
CMD ["poetry", "run", "htrflow", "--help"]


# CMD ["htrflow", "--help"]
# docker build -t airiksarkivet/htrflow:v0.1.1 -f docker/Dockerfile .
# docker push airiksarkivet/htrflow:v0.1.1
# docker run --gpus all -it --rm htrflow_im:latest /bin/bash
# docker exec -it <c_id> /bin/bash


