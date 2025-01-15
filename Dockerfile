# Use NVIDIA CUDA base image if GPU is available, otherwise use standard python
ARG USE_GPU=0
FROM nvidia/cuda:12.3.1-runtime-ubuntu22.04 AS gpu
FROM python:3.12-slim AS cpu

# Select final image based on USE_GPU arg
FROM ${USE_GPU:+gpu}${USE_GPU:-cpu}

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STANZA_RESOURCES_DIR=/app/stanza_resources

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir torch

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5004

# Run the application with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "5004", "--loop", "auto"]