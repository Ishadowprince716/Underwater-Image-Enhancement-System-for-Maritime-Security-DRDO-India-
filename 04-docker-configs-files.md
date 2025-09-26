### `src/sdk/python/client.py`
```python
#!/usr/bin/env python3
"""
Python SDK client for Underwater Image Enhancement API.
Provides easy access to enhancement services via REST and gRPC.
"""

import requests
import base64
import io
import numpy as np
import cv2
from typing import Optional, Union, Dict, Any, List, Tuple
from pathlib import Path
import logging
import grpc
import json

logger = logging.getLogger(__name__)

class UnderwaterEnhancementClient:
    """
    Client for underwater image enhancement API.
    
    Supports both REST and gRPC interfaces with automatic failover.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", 
                 grpc_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        """
        Initialize enhancement client.
        
        Args:
            base_url: Base URL for REST API
            grpc_url: gRPC service URL (optional)
            api_key: API key for authentication (optional)
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.grpc_url = grpc_url
        self.api_key = api_key
        self.timeout = timeout
        
        # Setup session
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
        
        # gRPC stub
        self._grpc_stub = None
        self._grpc_channel = None
        
        # Validate connection
        self._validate_connection()
    
    def enhance_image(self, image: Union[str, np.ndarray, bytes],
                     mode: str = "lightweight",
                     compute_metrics: bool = False,
                     output_format: str = "png") -> Dict[str, Any]:
        """
        Enhance a single image.
        
        Args:
            image: Input image (file path, numpy array, or bytes)
            mode: Enhancement mode ('lightweight' or 'hifi')
            compute_metrics: Whether to compute quality metrics
            output_format: Output format ('png' or 'jpeg')
            
        Returns:
            Enhancement result dictionary
        """
        try:
            # Prepare image data
            image_data = self._prepare_image_data(image)
            
            # REST API request
            files = {'file': ('image', image_data, f'image/{output_format}')}
            data = {
                'mode': mode,
                'compute_metrics': compute_metrics,
                'output_format': output_format
            }
            
            response = self.session.post(
                f"{self.base_url}/enhance",
                files=files,
                data=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Decode enhanced image
            if 'enhanced_image' in result:
                result['enhanced_image_array'] = self._decode_base64_image(
                    result['enhanced_image']
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            raise
    
    def enhance_batch(self, images: List[Union[str, np.ndarray, bytes]],
                     mode: str = "lightweight",
                     compute_metrics: bool = False) -> Dict[str, Any]:
        """
        Enhance multiple images in batch.
        
        Args:
            images: List of images to enhance
            mode: Enhancement mode
            compute_metrics: Whether to compute metrics
            
        Returns:
            Batch enhancement results
        """
        try:
            # Prepare files
            files = []
            for i, image in enumerate(images):
                image_data = self._prepare_image_data(image)
                files.append(('files', (f'image_{i}', image_data, 'image/png')))
            
            data = {
                'mode': mode,
                'compute_metrics': compute_metrics
            }
            
            response = self.session.post(
                f"{self.base_url}/enhance/batch",
                files=files,
                data=data,
                timeout=self.timeout * len(images)  # Scale timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Decode enhanced images
            if 'results' in result:
                for item in result['results']:
                    if 'enhanced_image' in item:
                        item['enhanced_image_array'] = self._decode_base64_image(
                            item['enhanced_image']
                        )
            
            return result
            
        except Exception as e:
            logger.error(f"Batch enhancement failed: {e}")
            raise
    
    def start_stream(self, source: str, mode: str = "lightweight",
                    rtsp_output: Optional[str] = None) -> str:
        """
        Start video stream enhancement.
        
        Args:
            source: Stream source (RTSP URL, file, camera index)
            mode: Enhancement mode
            rtsp_output: Optional RTSP output URL
            
        Returns:
            Stream ID
        """
        try:
            data = {
                'source': source,
                'mode': mode,
                'rtsp_output': rtsp_output
            }
            
            response = self.session.post(
                f"{self.base_url}/stream/start",
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            return result['stream_id']
            
        except Exception as e:
            logger.error(f"Stream start failed: {e}")
            raise
    
    def stop_stream(self, stream_id: str) -> bool:
        """
        Stop video stream enhancement.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            True if successful
        """
        try:
            response = self.session.post(
                f"{self.base_url}/stream/{stream_id}/stop",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Stream stop failed: {e}")
            return False
    
    def get_stream_status(self, stream_id: str) -> Dict[str, Any]:
        """
        Get stream status information.
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Stream status dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/stream/{stream_id}/status",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Stream status query failed: {e}")
            raise
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current system configuration.
        
        Returns:
            Configuration dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/config",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Config retrieval failed: {e}")
            raise
    
    def update_config(self, config: Dict[str, Any]) -> bool:
        """
        Update system configuration.
        
        Args:
            config: Configuration updates
            
        Returns:
            True if successful
        """
        try:
            response = self.session.put(
                f"{self.base_url}/config",
                json=config,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return True
            
        except Exception as e:
            logger.error(f"Config update failed: {e}")
            return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get system performance metrics.
        
        Returns:
            Metrics dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/metrics",
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.error(f"Metrics retrieval failed: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check.
        
        Returns:
            Health status dictionary
        """
        try:
            response = self.session.get(
                f"{self.base_url}/health",
                timeout=5  # Short timeout for health check
            )
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return {'status': 'unhealthy', 'error': str(e)}
    
    def _prepare_image_data(self, image: Union[str, np.ndarray, bytes]) -> bytes:
        """Prepare image data for API request."""
        if isinstance(image, str):
            # File path
            with open(image, 'rb') as f:
                return f.read()
        elif isinstance(image, np.ndarray):
            # NumPy array
            _, buffer = cv2.imencode('.png', image)
            return buffer.tobytes()
        elif isinstance(image, bytes):
            # Raw bytes
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to NumPy array."""
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
    
    def _validate_connection(self):
        """Validate connection to the service."""
        try:
            health = self.health_check()
            if health.get('status') != 'healthy':
                logger.warning(f"Service health check returned: {health}")
        except Exception as e:
            logger.warning(f"Failed to validate connection: {e}")

class AsyncUnderwaterEnhancementClient:
    """Async version of the enhancement client."""
    
    def __init__(self, base_url: str = "http://localhost:8000",
                 api_key: Optional[str] = None,
                 timeout: int = 30):
        """Initialize async client."""
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self._session = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        import aiohttp
        
        headers = {}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
            
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        self._session = aiohttp.ClientSession(
            headers=headers,
            timeout=timeout
        )
        
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def enhance_image(self, image: Union[str, np.ndarray, bytes],
                           mode: str = "lightweight",
                           compute_metrics: bool = False) -> Dict[str, Any]:
        """Async image enhancement."""
        if not self._session:
            raise RuntimeError("Client not initialized. Use 'async with' statement.")
        
        # Prepare image data
        image_data = self._prepare_image_data(image)
        
        # Create form data
        data = aiohttp.FormData()
        data.add_field('file', image_data, filename='image.png', content_type='image/png')
        data.add_field('mode', mode)
        data.add_field('compute_metrics', str(compute_metrics).lower())
        
        async with self._session.post(f"{self.base_url}/enhance", data=data) as response:
            response.raise_for_status()
            result = await response.json()
            
            # Decode enhanced image
            if 'enhanced_image' in result:
                result['enhanced_image_array'] = self._decode_base64_image(
                    result['enhanced_image']
                )
            
            return result
    
    def _prepare_image_data(self, image: Union[str, np.ndarray, bytes]) -> bytes:
        """Prepare image data for API request."""
        if isinstance(image, str):
            with open(image, 'rb') as f:
                return f.read()
        elif isinstance(image, np.ndarray):
            _, buffer = cv2.imencode('.png', image)
            return buffer.tobytes()
        elif isinstance(image, bytes):
            return image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _decode_base64_image(self, base64_str: str) -> np.ndarray:
        """Decode base64 image to NumPy array."""
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return image
```

### `src/sdk/cpp/include/uie_client.hpp`
```cpp
/**
 * C++ SDK client for Underwater Image Enhancement API
 * Header-only implementation for easy integration
 */

#pragma once

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <future>
#include <opencv2/opencv.hpp>

namespace uie {

enum class EnhancementMode {
    LIGHTWEIGHT,
    HIGH_FIDELITY
};

struct ProcessingStats {
    double processing_time_ms;
    double fps;
    std::pair<int, int> input_resolution;
    double uiqm_score;
    double uciqe_score;
    double memory_usage_mb;
};

struct EnhancementResult {
    cv::Mat enhanced_image;
    ProcessingStats stats;
    bool success;
    std::string error_message;
};

class UnderwaterEnhancementClient {
public:
    explicit UnderwaterEnhancementClient(
        const std::string& base_url = "http://localhost:8000",
        const std::string& api_key = "",
        int timeout_seconds = 30
    );
    
    ~UnderwaterEnhancementClient();
    
    // Single image enhancement
    EnhancementResult enhance_image(
        const cv::Mat& image,
        EnhancementMode mode = EnhancementMode::LIGHTWEIGHT,
        bool compute_metrics = false
    );
    
    // Batch enhancement
    std::vector<EnhancementResult> enhance_batch(
        const std::vector<cv::Mat>& images,
        EnhancementMode mode = EnhancementMode::LIGHTWEIGHT,
        bool compute_metrics = false
    );
    
    // Async enhancement
    std::future<EnhancementResult> enhance_image_async(
        const cv::Mat& image,
        EnhancementMode mode = EnhancementMode::LIGHTWEIGHT,
        bool compute_metrics = false
    );
    
    // Stream management
    std::string start_stream(
        const std::string& source,
        EnhancementMode mode = EnhancementMode::LIGHTWEIGHT,
        const std::string& rtsp_output = ""
    );
    
    bool stop_stream(const std::string& stream_id);
    
    // Configuration
    std::map<std::string, std::string> get_config();
    bool update_config(const std::map<std::string, std::string>& config);
    
    // Health and metrics
    bool health_check();
    std::map<std::string, double> get_metrics();
    
    // Utility functions
    static std::string encode_base64(const cv::Mat& image);
    static cv::Mat decode_base64(const std::string& base64_data);
    static std::string mode_to_string(EnhancementMode mode);
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace uie
```

## Docker Files

### `docker/Dockerfile.base`
```dockerfile
# Multi-stage production Dockerfile for underwater image enhancement
FROM nvcr.io/nvidia/pytorch:23.08-py3 as builder

WORKDIR /build

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Build optimized OpenCV with CUDA support
ARG OPENCV_VERSION=4.8.0
RUN git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git && \
    git clone --depth 1 --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git && \
    cd opencv && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D WITH_CUDA=ON \
          -D CUDA_ARCH_BIN=7.5,8.0,8.6,9.0 \
          -D WITH_CUBLAS=ON \
          -D WITH_CUDNN=ON \
          -D OPENCV_DNN_CUDA=ON \
          -D ENABLE_FAST_MATH=ON \
          -D CUDA_FAST_MATH=ON \
          -D WITH_TBB=ON \
          -D WITH_V4L=ON \
          -D WITH_QT=OFF \
          -D WITH_OPENGL=ON \
          -D OPENCV_GENERATE_PKGCONFIG=ON \
          -D OPENCV_EXTRA_MODULES_PATH=../opencv_contrib/modules \
          -D PYTHON3_EXECUTABLE=$(which python3) \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Production stage
FROM nvcr.io/nvidia/pytorch:23.08-py3 as production

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/conda/bin:${PATH}"

# Create non-root user
ARG UID=1000
ARG GID=1000
RUN groupadd -g ${GID} uie && \
    useradd -u ${UID} -g ${GID} -m -s /bin/bash uie && \
    mkdir -p /app /app/data /app/logs /app/cache && \
    chown -R uie:uie /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy built OpenCV from builder stage
COPY --from=builder /usr/local /usr/local
COPY --from=builder /opt/conda/lib/python3.*/site-packages /opt/conda/lib/python3.*/site-packages

WORKDIR /app

# Copy application files
COPY --chown=uie:uie src/ src/
COPY --chown=uie:uie configs/ configs/
COPY --chown=uie:uie models/ models/
COPY --chown=uie:uie requirements.txt setup.py ./

# Install application
RUN pip install --no-cache-dir -e . && \
    python -c "import cv2; print('OpenCV version:', cv2.__version__)"

# Copy entrypoint script
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Security hardening
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \; && \
    chmod +x /entrypoint.sh

# Switch to non-root user
USER uie

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080 50051

# Labels
LABEL maintainer="DRDO Maritime AI Systems <maritime-ai@drdo.gov.in>"
LABEL version="1.0.0"
LABEL description="Underwater Image Enhancement System for Maritime Security"

ENTRYPOINT ["/entrypoint.sh"]
CMD ["serve"]
```

### `docker/entrypoint.sh`
```bash
#!/bin/bash
set -e

# Underwater Image Enhancement System Entrypoint
# DRDO Maritime AI Systems

echo "Starting Underwater Image Enhancement System v1.0.0"
echo "DRDO Maritime AI Systems"

# Environment setup
export PYTHONPATH="/app/src:${PYTHONPATH}"
export CUDA_CACHE_PATH="/app/cache/cuda"
export TENSORRT_CACHE_PATH="/app/cache/tensorrt"

# Create cache directories
mkdir -p /app/cache/cuda /app/cache/tensorrt /app/logs

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        if nvidia-smi >/dev/null 2>&1; then
            echo "GPU detected:"
            nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
            export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0"}
        else
            echo "NVIDIA driver not properly loaded"
            export CUDA_VISIBLE_DEVICES=""
        fi
    else
        echo "No GPU detected, running in CPU mode"
        export CUDA_VISIBLE_DEVICES=""
    fi
}

# Function to warm up models
warmup_models() {
    echo "Warming up enhancement models..."
    python3 -c "
import sys
sys.path.append('/app/src')
from core.enhancement import UnderwaterImageEnhancer, EnhancementMode
import numpy as np

# Create small test image
test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

# Warm up lightweight mode
enhancer_light = UnderwaterImageEnhancer(mode=EnhancementMode.LIGHTWEIGHT, device='auto')
_, _ = enhancer_light.enhance_frame(test_img)
print('Lightweight mode warmed up')

# Warm up high-fidelity mode
enhancer_hifi = UnderwaterImageEnhancer(mode=EnhancementMode.HIGH_FIDELITY, device='auto')
_, _ = enhancer_hifi.enhance_frame(test_img)
print('High-fidelity mode warmed up')
print('Model warmup complete')
"
}

# Function to check model files
check_models() {
    echo "Checking model files..."
    
    if [ ! -f "/app/models/weights/unet_lite.onnx" ]; then
        echo "Warning: UNet ONNX model not found, using embedded weights"
    fi
    
    if [ ! -f "/app/models/weights/lut_model.onnx" ]; then
        echo "Warning: LUT ONNX model not found, using embedded weights"
    fi
    
    # Check for TensorRT engines
    if [ -n "${CUDA_VISIBLE_DEVICES}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "" ]; then
        if [ -f "/app/models/weights/unet_lite_fp16.trt" ]; then
            echo "Found TensorRT FP16 engine"
        fi
        
        if [ -f "/app/models/weights/unet_lite_int8.trt" ]; then
            echo "Found TensorRT INT8 engine"
        fi
    fi
}

# Function to setup logging
setup_logging() {
    local log_level=${LOG_LEVEL:-INFO}
    local log_format=${LOG_FORMAT:-json}
    
    export UIE_LOG_LEVEL=$log_level
    export UIE_LOG_FORMAT=$log_format
    
    echo "Logging configured: level=$log_level, format=$log_format"
}

# Function to validate configuration
validate_config() {
    local config_file=${UIE_CONFIG_FILE:-/app/configs/default.yaml}
    
    if [ -f "$config_file" ]; then
        echo "Using configuration file: $config_file"
    else
        echo "Configuration file not found: $config_file"
        echo "Using default configuration"
    fi
}

# Main execution logic
main() {
    local command=${1:-serve}
    
    echo "Command: $command"
    
    # Common setup
    check_gpu
    setup_logging
    validate_config
    check_models
    
    case "$command" in
        "serve")
            echo "Starting API server..."
            warmup_models
            exec python3 -m src.api.rest_server --host 0.0.0.0 --port 8000
            ;;
        "grpc")
            echo "Starting gRPC server..."
            warmup_models
            exec python3 -m src.api.grpc_server --host 0.0.0.0 --port 50051
            ;;
        "worker")
            echo "Starting worker process..."
            warmup_models
            exec python3 -m src.core.worker
            ;;
        "cli")
            shift
            exec python3 -m src.cli.main "$@"
            ;;
        "benchmark")
            echo "Running benchmark..."
            exec python3 -m src.cli.main bench --dataset /app/samples/input
            ;;
        "export")
            echo "Exporting models..."
            exec python3 /app/models/scripts/export_onnx.py
            ;;
        "tensorrt")
            echo "Building TensorRT engines..."
            exec python3 /app/models/scripts/build_tensorrt.py
            ;;
        "bash")
            exec /bin/bash
            ;;
        *)
            echo "Unknown command: $command"
            echo "Available commands: serve, grpc, worker, cli, benchmark, export, tensorrt, bash"
            exit 1
            ;;
    esac
}

# Signal handling for graceful shutdown
trap 'echo "Received SIGTERM, shutting down gracefully..."; kill -TERM $PID; wait $PID' TERM
trap 'echo "Received SIGINT, shutting down..."; kill -INT $PID; wait $PID' INT

# Execute main function
main "$@" &
PID=$!
wait $PID
```

### `docker/docker-compose.yml`
```yaml
version: '3.8'

services:
  # Main API service
  uie-api:
    build:
      context: ..
      dockerfile: docker/Dockerfile.base
    image: underwater-image-enhancement:latest
    container_name: uie-api
    ports:
      - "8000:8000"
      - "8080:8080"
      - "50051:50051"
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - UIE_CONFIG_FILE=/app/configs/default.yaml
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./cache:/app/cache
      - ./configs:/app/configs:ro
      - ./models/weights:/app/models/weights:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    networks:
      - uie-network

  # Development service
  uie-dev:
    build:
      context: ..
      dockerfile: docker/Dockerfile.dev
    image: underwater-image-enhancement:dev
    container_name: uie-dev
    ports:
      - "8888:8888"  # Jupyter
      - "8787:8787"  # VS Code Server
      - "6006:6006"  # TensorBoard
    environment:
      - CUDA_VISIBLE_DEVICES=0
      - JUPYTER_ENABLE_LAB=yes
    volumes:
      - ../:/workspace
      - ./data:/workspace/data
      - ./logs:/workspace/logs
      - ./cache:/workspace/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    profiles:
      - dev
    networks:
      - uie-network

  # Monitoring with Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: uie-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped
    profiles:
      - monitoring
    networks:
      - uie-network

  # Grafana for dashboards
  grafana:
    image: grafana/grafana:latest
    container_name: uie-grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana_data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources:ro
    restart: unless-stopped
    profiles:
      - monitoring
    networks:
      - uie-network

volumes:
  prometheus_data:
  grafana_data:

networks:
  uie-network:
    driver: bridge
```

## Configuration Files

### `configs/default.yaml`
```yaml
# Default configuration for Underwater Image Enhancement System
# DRDO Maritime AI Systems

# System settings
system:
  name: "Underwater Image Enhancement System"
  version: "1.0.0"
  environment: "production"

# Processing configuration
processing:
  default_mode: "lightweight"
  device: "auto"  # auto, cpu, cuda, cuda:0
  max_concurrent_requests: 100
  request_timeout_seconds: 30
  enable_gpu_memory_growth: true
  gpu_memory_limit_mb: 4096

# Enhancement parameters
enhancement:
  # Gamma correction
  gamma_value: 1.2
  
  # Color space processing
  use_lab_color: true
  use_clahe: true
  clahe_clip_limit: 3.0
  clahe_tile_grid_size: [8, 8]
  
  # Noise reduction
  denoise: true
  denoise_strength: 0.1
  
  # White balance correction
  white_balance:
    method: "underwater_physics"  # gray_world, white_patch, underwater_physics, adaptive
    adaptation_strength: 0.8
    auto_detect_water_type: true
  
  # Guided filter settings
  guided_filter:
    radius: 8
    eps: 0.01
    use_color_guide: true
  
  # Dehazing parameters
  dehazing:
    enabled: true
    method: "dark_channel"  # dark_channel, color_attenuation
    beta: 1.0  # Attenuation coefficient
    tx: 0.1    # Minimum transmission
    use_dark_channel: true
    omega: 0.95  # Dark channel factor
    patch_size: 15
    water_type: "oceanic"  # oceanic, coastal, turbid

# Model configuration
models:
  # U-Net Lite model
  unet_lite:
    model_path: "models/weights/unet_lite.onnx"
    tensorrt_engine_path: "models/weights/unet_lite_fp16.trt"
    input_size: [224, 224]
    batch_size: 1
    precision: "fp16"  # fp32, fp16, int8
    
  # LUT model
  lut_model:
    model_path: "models/weights/lut_model.onnx"
    lut_size: 33
    trilinear_interpolation: true

# Quality metrics
metrics:
  enabled: true
  compute_on_demand: true
  
  # Available metrics
  uiqm:
    enabled: true
    weight: 1.0
    
  uciqe:
    enabled: true
    weight: 1.0
    
  psnr:
    enabled: false  # Requires reference image
    
  ssim:
    enabled: false  # Requires reference image

# Performance settings
performance:
  # Memory management
  max_memory_cache_mb: 512
  clear_cache_interval_seconds: 300
  
  # Threading
  worker_threads: 4
  io_threads: 2
  
  # Optimization
  enable_tensorrt: true
  enable_cuda_graphs: false  # Experimental
  warmup_iterations: 5

# API server configuration
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_request_size_mb: 50
  
  # CORS settings
  cors:
    enabled: true
    allow_origins: ["*"]
    allow_methods: ["GET", "POST", "PUT", "DELETE"]
    allow_headers: ["*"]
  
  # Rate limiting
  rate_limit:
    enabled: true
    requests_per_minute: 60
    burst_size: 10

# gRPC server configuration  
grpc:
  host: "0.0.0.0"
  port: 50051
  max_workers: 10
  max_message_length: 100  # MB

# Streaming configuration
streaming:
  # Buffer settings
  frame_buffer_size: 10
  max_streams: 5
  stream_timeout_seconds: 300
  
  # RTSP settings
  rtsp_enabled: true
  rtsp_output_format: "h264"
  rtsp_bitrate_kbps: 2000
  
  # WebRTC settings
  webrtc_enabled: false

# Logging configuration
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
  format: "json"  # json, console
  
  # File logging
  file_logging:
    enabled: true
    path: "logs/uie.log"
    max_size_mb: 100
    backup_count: 5
    
  # Performance logging
  performance_logging:
    enabled: true
    interval_seconds: 60
    include_memory_stats: true
    include_gpu_stats: true

# Security settings
security:
  # API authentication
  api_key_required: false
  api_key_header: "X-API-Key"
  
  # TLS settings
  tls:
    enabled: false
    cert_file: ""
    key_file: ""
    
  # Input validation
  max_image_size_mb: 25
  allowed_image_formats: ["jpg", "jpeg", "png", "bmp"]
  allowed_video_formats: ["mp4", "avi", "mov", "webm"]

# Monitoring and metrics
monitoring:
  # Prometheus metrics
  prometheus:
    enabled: true
    port: 8080
    path: "/metrics"
    
  # Health checks
  health_check:
    enabled: true
    path: "/health"
    timeout_seconds: 5
    
  # Performance metrics
  performance_metrics:
    enabled: true
    collection_interval_seconds: 30
    retention_hours: 24

# Data handling
data:
  # Input/output paths
  input_path: "data/input"
  output_path: "data/output"
  temp_path: "data/temp"
  
  # Cleanup settings
  auto_cleanup: true
  cleanup_interval_hours: 24
  max_temp_age_hours: 6
  
  # Privacy settings
  store_processed_images: false
  anonymize_logs: true

# Presets for different missions
presets:
  port_survey:
    gamma_value: 1.3
    dehazing:
      beta: 1.2
      tx: 0.15
    white_balance:
      method: "underwater_physics"
      adaptation_strength: 0.9
      
  diver_assist:
    gamma_value: 1.4
    use_lab_color: true
    denoise: true
    dehazing:
      beta: 0.8
      tx: 0.1
      
  deep_water:
    gamma_value: 1.5
    white_balance:
      method: "underwater_physics"
      adaptation_strength: 1.0
    dehazing:
      beta: 1.5
      tx: 0.2
      
  high_performance:
    processing:
      device: "cuda"
      max_concurrent_requests: 200
    models:
      unet_lite:
        precision: "fp16"
        batch_size: 4
    performance:
      enable_tensorrt: true
      enable_cuda_graphs: true
```

### `configs/presets/port-survey.yaml`
```yaml
# Port Survey Mission Preset
# Optimized for harbor surveillance and port security operations

name: "Port Survey"
description: "Harbor surveillance with enhanced dehazing for port security"

# Override base configuration
enhancement:
  gamma_value: 1.3
  use_lab_color: true
  
  white_balance:
    method: "underwater_physics"
    adaptation_strength: 0.9
    
  dehazing:
    enabled: true
    method: "dark_channel"
    beta: 1.2
    tx: 0.15
    water_type: "coastal"
    
  guided_filter:
    radius: 10
    eps: 0.01

performance:
  # Optimize for real-time processing
  enable_tensorrt: true
  worker_threads: 6
  
models:
  unet_lite:
    precision: "fp16"
    batch_size: 1

# Mission-specific parameters
mission:
  environment: "coastal_port"
  lighting_conditions: "variable"
  expected_turbidity: "moderate"
  priority: "speed_over_quality"
```