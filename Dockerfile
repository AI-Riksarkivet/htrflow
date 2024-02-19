# Use the official Python image with the version as specified by the cookiecutter template
FROM python:3.10

# Set the working directory in the container
WORKDIR /usr/src/app

# Install Poetry
RUN pip install poetry

# Copy the Python project files to the container
COPY . .

# Disable virtualenv creation by Poetry and install dependencies
# This ensures that the packages are installed globally in the container
RUN poetry config virtualenvs.create false && poetry install

# Set the command to run the application using Poetry
CMD ["poetry", "run", "python", "./src/htrflow_core/main.py"]
