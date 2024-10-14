# Dockerfile
FROM python:3.11

# Install Hera and Pydantic
RUN pip install hera-workflows pydantic

# Set the working directory
WORKDIR /app

# Copy only the test_hera.py file into the /app directory inside the container
COPY test_hera.py /app/test_hera.py

# Make sure /app is on the Python path
ENV PYTHONPATH=/app
