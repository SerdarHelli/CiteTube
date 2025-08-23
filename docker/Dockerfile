# Multi-stage Dockerfile for CiteTube
# Stage 1: Base image with system dependencies
FROM nvidia/cuda:12.1-devel-ubuntu22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    postgresql-client \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:$PATH"

# Create app directory
WORKDIR /app

# Create virtual environment with uv
RUN uv venv /opt/venv --python=3.12 --seed
ENV PATH="/opt/venv/bin:$PATH"

# Stage 2: Install Python dependencies
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies with uv (faster)
RUN uv pip install -r requirements.txt

# Install vLLM with CUDA support
RUN uv pip install -U vllm --torch-backend=cu128

# Stage 3: Application
FROM dependencies as app

# Copy application code
COPY . .

# Install the application in development mode
RUN uv pip install -e .

# Create data directories
RUN mkdir -p data/logs

# Expose ports
EXPOSE 7860 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Default command (can be overridden)
CMD ["python", "-m", "src.citetube.ui.app"]