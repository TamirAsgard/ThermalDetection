# Multi-stage Dockerfile for Thermal Detection System
# Supports both x86_64 and ARM64 (Raspberry Pi) architectures

ARG PYTHON_VERSION=3.11
ARG DEBIAN_VERSION=bookworm

# =============================================================================
# Base stage - Common dependencies
# =============================================================================
FROM python:${PYTHON_VERSION}-slim-${DEBIAN_VERSION} as base

LABEL maintainer="Thermal Detection Team <info@thermal-detection.com>"
LABEL description="Thermal Forehead Detection System"
LABEL version="1.0.0"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Essential build tools
    build-essential \
    cmake \
    pkg-config \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    # Image and video processing
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    # GUI and display (for debugging)
    libgtk-3-dev \
    libcanberra-gtk-module \
    libcanberra-gtk3-module \
    # Image libraries
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    # Math and scientific computing
    libatlas-base-dev \
    gfortran \
    libhdf5-dev \
    # USB and hardware access
    libusb-1.0-0-dev \
    udev \
    # Network and security
    curl \
    wget \
    ca-certificates \
    # Utilities
    git \
    unzip \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -r thermal && useradd -r -g thermal -d /app -s /bin/bash thermal

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt ./
COPY tests/test_requirements.txt ./tests/

# =============================================================================
# Dependencies stage - Install Python packages
# =============================================================================
FROM base as dependencies

# Upgrade pip
RUN pip install --upgrade pip setuptools wheel

# Install Python dependencies
RUN pip install -r requirements.txt

# Install test dependencies for development builds
ARG BUILD_TYPE=production
RUN if [ "$BUILD_TYPE" = "development" ]; then \
        pip install -r tests/test_requirements.txt; \
    fi

# =============================================================================
# Development stage - For development and testing
# =============================================================================
FROM dependencies as development

# Install additional development tools
RUN pip install \
    jupyter \
    ipython \
    black \
    flake8 \
    mypy \
    pre-commit

# Copy all source code
COPY . .

# Install the package in development mode
RUN pip install -e .

# Set up development environment
RUN chown -R thermal:thermal /app
USER thermal

# Expose ports
EXPOSE 8000 8888

# Development command
CMD ["python", "-m", "src.api.app"]

# =============================================================================
# Production stage - Optimized for deployment
# =============================================================================
FROM dependencies as production

# Copy only necessary files
COPY src/ ./src/
COPY config/ ./config/
COPY model/ ./model/
COPY sounds/ ./sounds/
COPY setup.py ./
COPY README.md ./
COPY LICENSE ./

# Install the package
RUN pip install .

# Create necessary directories
RUN mkdir -p logs data backups test_data/images && \
    chown -R thermal:thermal /app

# Switch to non-root user
USER thermal

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Expose port
EXPOSE 8000

# Production command
CMD ["thermal-api"]

# =============================================================================
# Raspberry Pi specific stage
# =============================================================================
FROM production as raspberry-pi

# Switch back to root for hardware setup
USER root

# Install Raspberry Pi specific packages
RUN apt-get update && apt-get install -y \
    # GPIO access
    python3-rpi.gpio \
    # Camera module
    python3-picamera \
    # I2C and SPI
    python3-smbus \
    i2c-tools \
    # Additional hardware support
    python3-gpiozero \
    && rm -rf /var/lib/apt/lists/*

# Install Pi-specific Python packages
RUN pip install \
    RPi.GPIO>=0.7.1 \
    picamera>=1.13 \
    adafruit-circuitpython-dht>=3.7.0 \
    gpiozero>=1.6.2

# Add user to GPIO and video groups
RUN usermod -a -G gpio,video,i2c,spi thermal

# Copy Pi-specific configuration
COPY config/raspberry_pi_config.yaml ./config/

# Switch back to thermal user
USER thermal

# Pi-specific command
CMD ["thermal-detection", "--config", "config/raspberry_pi_config.yaml"]

# =============================================================================
# GPU-enabled stage (for NVIDIA GPUs)
# =============================================================================
FROM nvidia/cuda:11.8-runtime-ubuntu22.04 as gpu

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy from production stage
COPY --from=production /app /app
COPY --from=production /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

WORKDIR /app

# Install GPU-specific packages
RUN pip install \
    onnxruntime-gpu>=1.16.0 \
    cupy-cuda11x

# Create user and set permissions
RUN groupadd -r thermal && useradd -r -g thermal thermal && \
    chown -R thermal:thermal /app

USER thermal

EXPOSE 8000

CMD ["thermal-api"]

# =============================================================================
# Build arguments and final stage selection
# =============================================================================
FROM ${TARGET_STAGE:-production} as final

# Add labels with build information
ARG BUILD_DATE
ARG VCS_REF
ARG VERSION

LABEL org.opencontainers.image.created=${BUILD_DATE} \
      org.opencontainers.image.url="https://github.com/yourusername/thermal-detection-system" \
      org.opencontainers.image.source="https://github.com/yourusername/thermal-detection-system" \
      org.opencontainers.image.version=${VERSION} \
      org.opencontainers.image.revision=${VCS_REF} \
      org.opencontainers.image.vendor="Thermal Detection Team" \
      org.opencontainers.image.title="Thermal Detection System" \
      org.opencontainers.image.description="Real-time thermal forehead temperature detection system" \
      org.opencontainers.image.documentation="https://thermal-detection-system.readthedocs.io/" \
      org.opencontainers.image.licenses="MIT"

# Copy startup script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]