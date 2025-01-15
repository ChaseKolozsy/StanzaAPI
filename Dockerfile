# Modify the base image selection
ARG USE_GPU=1
FROM python:3.12-slim

# Install CUDA dependencies only if USE_GPU=1
RUN if [ "$USE_GPU" = "1" ] ; then \
    apt-get update && apt-get install -y --no-install-recommends \
    cuda-runtime-12-3 \
    && rm -rf /var/lib/apt/lists/* ; \
    fi

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    STANZA_RESOURCES_DIR=/app/stanza_resources

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