# Use NVIDIA CUDA base image with Ubuntu 23.10
FROM nvidia/cuda:12.3.1-runtime-ubuntu23.10

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STANZA_RESOURCES_DIR=/app/stanza_resources

# Install Python and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

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