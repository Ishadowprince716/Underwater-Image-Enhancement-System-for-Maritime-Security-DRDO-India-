## Kubernetes Deployment Files

### `deployment/k8s/namespace.yaml`
```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: uie-system
  labels:
    name: uie-system
    app.kubernetes.io/name: underwater-image-enhancement
    app.kubernetes.io/version: "1.0.0"
    app.kubernetes.io/component: system
    app.kubernetes.io/part-of: maritime-security
    app.kubernetes.io/managed-by: kubernetes
  annotations:
    description: "Underwater Image Enhancement System for Maritime Security"
```

### `deployment/k8s/secret.yaml`
```yaml
apiVersion: v1
kind: Secret
metadata:
  name: uie-secrets
  namespace: uie-system
  labels:
    app.kubernetes.io/name: underwater-image-enhancement
    app.kubernetes.io/component: secrets
type: Opaque
data:
  # Base64 encoded values - replace with actual secrets
  api-key: ""  # echo -n "your-api-key" | base64
  registry-password: ""  # echo -n "registry-pass" | base64
  tls-cert: ""  # base64 encoded certificate
  tls-key: ""   # base64 encoded private key
---
apiVersion: v1
kind: Secret
metadata:
  name: registry-secret
  namespace: uie-system
type: kubernetes.io/dockerconfigjson
data:
  .dockerconfigjson: eyJhdXRocyI6eyJyZWdpc3RyeS5kcmRvLmdvdi5pbiI6eyJ1c2VybmFtZSI6IiIsInBhc3N3b3JkIjoiIiwiYXV0aCI6IiJ9fX0=
```

### `deployment/k8s/configmap.yaml`
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: uie-config
  namespace: uie-system
  labels:
    app.kubernetes.io/name: underwater-image-enhancement
    app.kubernetes.io/component: configuration
data:
  default.yaml: |
    # Kubernetes production configuration
    processing:
      default_mode: "lightweight"
      device: "cuda"
      max_concurrent_requests: 100
      request_timeout_seconds: 30
      
    enhancement:
      gamma_value: 1.2
      use_lab_color: true
      denoise: true
      
      white_balance:
        method: "underwater_physics"
        adaptation_strength: 0.8
        
      guided_filter:
        radius: 8
        eps: 0.01
        
      dehazing:
        enabled: true
        beta: 1.0
        tx: 0.1
        use_dark_channel: true
        
    models:
      unet_lite:
        tensorrt_engine_path: "/app/cache/tensorrt/unet_lite_fp16.trt"
        precision: "fp16"
        batch_size: 1
        
    server:
      host: "0.0.0.0"
      port: 8000
      workers: 4
      max_request_size_mb: 50
      
    logging:
      level: "INFO"
      format: "json"
      file_logging:
        enabled: false  # Use container logs
        
    monitoring:
      prometheus:
        enabled: true
        port: 8080
        path: "/metrics"
        
  presets.yaml: |
    presets:
      port-survey:
        enhancement:
          gamma_value: 1.3
          dehazing:
            beta: 1.2
            tx: 0.15
      diver-assist:
        enhancement:
          gamma_value: 1.4
          use_lab_color: true
          denoise: true
      deep-water:
        enhancement:
          gamma_value: 1.5
          dehazing:
            beta: 1.5
            tx: 0.2
```

### `deployment/k8s/deployment.yaml`
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: uie-api
  namespace: uie-system
  labels:
    app.kubernetes.io/name: underwater-image-enhancement
    app.kubernetes.io/component: api-server
    app.kubernetes.io/version: "1.0.0"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  selector:
    matchLabels:
      app.kubernetes.io/name: underwater-image-enhancement
      app.kubernetes.io/component: api-server
  template:
    metadata:
      labels:
        app.kubernetes.io/name: underwater-image-enhancement
        app.kubernetes.io/component: api-server
        app.kubernetes.io/version: "1.0.0"
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
        config-hash: "placeholder"  # Will be updated by CD pipeline
    spec:
      serviceAccountName: uie-service-account
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: uie-api
        image: registry.drdo.gov.in/maritime-ai/underwater-image-enhancement:1.0.0
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8000
          protocol: TCP
        - name: metrics
          containerPort: 8080
          protocol: TCP
        - name: grpc
          containerPort: 50051
          protocol: TCP
        env:
        - name: UIE_CONFIG_FILE
          value: "/etc/uie/default.yaml"
        - name: UIE_LOG_LEVEL
          value: "INFO"
        - name: UIE_LOG_FORMAT
          value: "json"
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: PYTHONUNBUFFERED
          value: "1"
        - name: POD_NAME
          valueFrom:
            fieldRef:
              fieldPath: metadata.name
        - name: POD_NAMESPACE
          valueFrom:
            fieldRef:
              fieldPath: metadata.namespace
        volumeMounts:
        - name: config-volume
          mountPath: /etc/uie
          readOnly: true
        - name: cache-volume
          mountPath: /app/cache
        - name: tmp-volume
          mountPath: /tmp
        - name: logs-volume
          mountPath: /app/logs
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
            nvidia.com/gpu: 1
          limits:
            memory: "8Gi"
            cpu: "4000m"
            nvidia.com/gpu: 1
        livenessProbe:
          httpGet:
            path: /health
            port: http
            httpHeaders:
            - name: Accept
              value: application/json
          initialDelaySeconds: 120
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /health
            port: http
            httpHeaders:
            - name: Accept
              value: application/json
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 2
        startupProbe:
          httpGet:
            path: /health
            port: http
            httpHeaders:
            - name: Accept
              value: application/json
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 24  # Allow 4 minutes for startup
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
          runAsNonRoot: true
          runAsUser: 1000
      volumes:
      - name: config-volume
        configMap:
          name: uie-config
          defaultMode: 0644
      - name: cache-volume
        emptyDir:
          sizeLimit: 4Gi
      - name: tmp-volume
        emptyDir:
          sizeLimit: 1Gi
      - name: logs-volume
        emptyDir:
          sizeLimit: 2Gi
      nodeSelector:
        kubernetes.io/arch: amd64
        accelerator: nvidia-gpu
      tolerations:
      - key: nvidia.com/gpu
        operator: Exists
        effect: NoSchedule
      - key: accelerator
        operator: Equal
        value: nvidia-gpu
        effect: NoSchedule
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app.kubernetes.io/name
                  operator: In
                  values:
                  - underwater-image-enhancement
                - key: app.kubernetes.io/component
                  operator: In
                  values:
                  - api-server
              topologyKey: kubernetes.io/hostname
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: accelerator
                operator: In
                values:
                - nvidia-gpu
      imagePullSecrets:
      - name: registry-secret
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: uie-service-account
  namespace: uie-system
  labels:
    app.kubernetes.io/name: underwater-image-enhancement
automountServiceAccountToken: false
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: uie-role
  namespace: uie-system
rules:
- apiGroups: [""]
  resources: ["configmaps"]
  verbs: ["get", "list", "watch"]
- apiGroups: [""]
  resources: ["secrets"]
  verbs: ["get", "list"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: uie-role-binding
  namespace: uie-system
subjects:
- kind: ServiceAccount
  name: uie-service-account
  namespace: uie-system
roleRef:
  kind: Role
  name: uie-role
  apiGroup: rbac.authorization.k8s.io
```

## Helm Chart Files

### `deployment/helm/Chart.yaml`
```yaml
apiVersion: v2
name: underwater-image-enhancement
description: A Helm chart for Underwater Image Enhancement System
version: 1.0.0
appVersion: "1.0.0"
type: application

keywords:
- underwater
- image-enhancement
- maritime
- defense
- ai
- computer-vision

maintainers:
- name: DRDO Maritime AI Team
  email: maritime-ai@drdo.gov.in
  url: https://drdo.gov.in

sources:
- https://github.com/drdo-maritime-ai/underwater-image-enhancement

annotations:
  category: AI/ML
  artifacthub.io/license: Proprietary
  artifacthub.io/operator: "false"
  artifacthub.io/prerelease: "false"
  artifacthub.io/containsSecurityUpdates: "true"
  artifacthub.io/changes: |
    - Initial release of production-ready underwater image enhancement system
    - Support for NVIDIA GPU acceleration with TensorRT
    - REST and gRPC APIs with comprehensive monitoring
    - Mission-specific presets for maritime security operations
```

### `deployment/helm/values.yaml`
```yaml
# Default values for underwater-image-enhancement
# This is a YAML-formatted file.
# Declare variables to be substituted into your templates.

# Global settings
global:
  imageRegistry: "registry.drdo.gov.in/maritime-ai"
  imagePullSecrets:
    - name: "registry-secret"

# Image configuration
image:
  repository: underwater-image-enhancement
  tag: "1.0.0"
  pullPolicy: IfNotPresent
  digest: ""

# Deployment configuration
replicaCount: 3

# Service configuration
service:
  type: ClusterIP
  ports:
    http:
      port: 80
      targetPort: 8000
      protocol: TCP
    metrics:
      port: 8080
      targetPort: 8080
      protocol: TCP
    grpc:
      port: 50051
      targetPort: 50051
      protocol: TCP
  annotations:
    prometheus.io/scrape: "true"
    prometheus.io/port: "8080"
    prometheus.io/path: "/metrics"

# Ingress configuration
ingress:
  enabled: true
  className: "nginx"
  annotations:
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
    - host: uie-api.maritime.drdo.gov.in
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: uie-api-tls
      hosts:
        - uie-api.maritime.drdo.gov.in

# Resource configuration
resources:
  requests:
    memory: "2Gi"
    cpu: "1000m"
    nvidia.com/gpu: 1
  limits:
    memory: "8Gi"
    cpu: "4000m"
    nvidia.com/gpu: 1

# GPU configuration
gpu:
  enabled: true
  count: 1
  nodeSelector:
    accelerator: nvidia-gpu
  tolerations:
    - key: nvidia.com/gpu
      operator: Exists
      effect: NoSchedule
    - key: accelerator
      operator: Equal
      value: nvidia-gpu
      effect: NoSchedule

# Autoscaling configuration
autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
  targetGPUUtilizationPercentage: 85
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 100
        periodSeconds: 30

# Configuration
config:
  # Enhancement settings
  enhancement:
    default_mode: "lightweight"
    gamma_value: 1.2
    use_lab_color: true
    denoise: true
    
  # Processing settings
  processing:
    device: "cuda"
    max_concurrent_requests: 100
    request_timeout_seconds: 30
    
  # Server settings
  server:
    host: "0.0.0.0"
    port: 8000
    workers: 4
    max_request_size_mb: 50
    
  # Logging settings
  logging:
    level: "INFO"
    format: "json"

# Mission presets
presets:
  portSurvey:
    enabled: true
    gamma_value: 1.3
    dehazing_beta: 1.2
    
  diverAssist:
    enabled: true
    gamma_value: 1.4
    use_lab_color: true
    denoise: true
    
  deepWater:
    enabled: true
    gamma_value: 1.5
    dehazing_beta: 1.5

# Security settings
security:
  # Pod Security Context
  podSecurityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
      
  # Container Security Context
  securityContext:
    allowPrivilegeEscalation: false
    readOnlyRootFilesystem: true
    capabilities:
      drop:
      - ALL
    runAsNonRoot: true
    runAsUser: 1000

  # Network policies
  networkPolicy:
    enabled: true
    ingress:
      - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
        ports:
        - protocol: TCP
          port: 8080
      - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
        ports:
        - protocol: TCP
          port: 8000

# Monitoring configuration
monitoring:
  prometheus:
    enabled: true
    port: 8080
    path: "/metrics"
    
  serviceMonitor:
    enabled: true
    namespace: monitoring
    interval: 30s
    scrapeTimeout: 10s
    
  grafana:
    enabled: true
    dashboards:
      - uie-performance
      - uie-quality-metrics
      - uie-system-health

# Storage configuration
storage:
  cache:
    enabled: true
    size: 4Gi
    storageClass: "fast-ssd"
    
  logs:
    enabled: true
    size: 2Gi
    storageClass: "standard"

# Probes configuration
probes:
  liveness:
    enabled: true
    httpGet:
      path: /health
      port: http
    initialDelaySeconds: 120
    periodSeconds: 30
    timeoutSeconds: 10
    failureThreshold: 3
    
  readiness:
    enabled: true
    httpGet:
      path: /health
      port: http
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 2
    
  startup:
    enabled: true
    httpGet:
      path: /health
      port: http
    initialDelaySeconds: 30
    periodSeconds: 10
    timeoutSeconds: 5
    failureThreshold: 24

# Node selection and affinity
nodeSelector:
  kubernetes.io/arch: amd64

affinity:
  podAntiAffinity:
    preferredDuringSchedulingIgnoredDuringExecution:
    - weight: 100
      podAffinityTerm:
        labelSelector:
          matchExpressions:
          - key: app.kubernetes.io/name
            operator: In
            values:
            - underwater-image-enhancement
        topologyKey: kubernetes.io/hostname
  nodeAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      nodeSelectorTerms:
      - matchExpressions:
        - key: accelerator
          operator: In
          values:
          - nvidia-gpu

tolerations: []

# Service account
serviceAccount:
  create: true
  name: ""
  annotations: {}
  automountServiceAccountToken: false

# RBAC configuration
rbac:
  create: true
  rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["secrets"]
    verbs: ["get", "list"]

# Environment variables
env:
  - name: CUDA_VISIBLE_DEVICES
    value: "0"
  - name: PYTHONUNBUFFERED
    value: "1"
  - name: UIE_LOG_LEVEL
    value: "INFO"
  - name: UIE_LOG_FORMAT
    value: "json"

# Volume mounts
volumeMounts:
  - name: cache-volume
    mountPath: /app/cache
  - name: tmp-volume
    mountPath: /tmp
  - name: logs-volume
    mountPath: /app/logs

volumes:
  - name: cache-volume
    emptyDir:
      sizeLimit: 4Gi
  - name: tmp-volume
    emptyDir:
      sizeLimit: 1Gi
  - name: logs-volume
    emptyDir:
      sizeLimit: 2Gi

# Additional labels
labels: {}

# Additional annotations
annotations: {}

# Development settings (disabled by default)
development:
  enabled: false
  jupyter:
    enabled: false
    port: 8888
  tensorboard:
    enabled: false
    port: 6006
```

## Test Files

### `tests/conftest.py`
```python
#!/usr/bin/env python3
"""
Pytest configuration and fixtures for underwater image enhancement tests.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import sys
from pathlib import Path
from typing import Generator
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import test modules
from core.enhancement import UnderwaterImageEnhancer, EnhancementMode
from core.metrics import ImageQualityMetrics

# Configure logging for tests
logging.basicConfig(level=logging.INFO)

@pytest.fixture(scope="session")
def test_data_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test data."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture
def sample_underwater_image() -> np.ndarray:
    """Generate a sample underwater-like image for testing."""
    # Create realistic underwater image with blue cast
    height, width = 480, 640
    
    # Base image with some structure
    image = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add some geometric patterns to simulate underwater structures
    cv2.rectangle(image, (100, 100), (200, 200), (80, 120, 150), -1)
    cv2.circle(image, (400, 300), 80, (60, 100, 140), -1)
    
    # Apply underwater color cast (blue dominant, reduced red)
    image[:, :, 0] = np.clip(image[:, :, 0] * 1.4, 0, 255)  # Enhance blue
    image[:, :, 1] = np.clip(image[:, :, 1] * 1.1, 0, 255)  # Slight green boost
    image[:, :, 2] = np.clip(image[:, :, 2] * 0.6, 0, 255)  # Reduce red significantly
    
    # Add some noise to simulate real conditions
    noise = np.random.normal(0, 5, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    return image

@pytest.fixture
def sample_clear_image() -> np.ndarray:
    """Generate a sample clear (reference) image."""
    height, width = 480, 640
    
    # Create well-balanced image
    image = np.random.randint(80, 220, (height, width, 3), dtype=np.uint8)
    
    # Add structures
    cv2.rectangle(image, (100, 100), (200, 200), (120, 150, 180), -1)
    cv2.circle(image, (400, 300), 80, (140, 180, 200), -1)
    
    return image

@pytest.fixture
def multiple_test_images() -> list:
    """Generate multiple test images with different characteristics."""
    images = []
    
    # Dark underwater image
    dark_img = np.random.randint(20, 80, (240, 320, 3), dtype=np.uint8)
    dark_img[:, :, 0] *= 1.5  # Blue cast
    dark_img[:, :, 2] *= 0.4  # Reduce red
    images.append(dark_img)
    
    # Bright underwater image  
    bright_img = np.random.randint(150, 240, (240, 320, 3), dtype=np.uint8)
    bright_img[:, :, 0] *= 1.2  # Blue cast
    bright_img[:, :, 2] *= 0.7  # Reduce red
    images.append(bright_img)
    
    # Medium contrast underwater image
    medium_img = np.random.randint(80, 160, (240, 320, 3), dtype=np.uint8)
    medium_img[:, :, 0] *= 1.3  # Blue cast
    medium_img[:, :, 2] *= 0.5  # Reduce red
    images.append(medium_img)
    
    return images

@pytest.fixture
def enhancer_lightweight() -> UnderwaterImageEnhancer:
    """Create lightweight mode enhancer for testing."""
    return UnderwaterImageEnhancer(
        mode=EnhancementMode.LIGHTWEIGHT,
        device='cpu',  # Use CPU for testing
        config={'batch_size': 1}
    )

@pytest.fixture
def enhancer_hifi() -> UnderwaterImageEnhancer:
    """Create high-fidelity mode enhancer for testing."""
    return UnderwaterImageEnhancer(
        mode=EnhancementMode.HIGH_FIDELITY,
        device='cpu',  # Use CPU for testing
        config={'batch_size': 1}
    )

@pytest.fixture
def metrics_calculator() -> ImageQualityMetrics:
    """Create image quality metrics calculator."""
    return ImageQualityMetrics()

@pytest.fixture
def temp_image_file(sample_underwater_image, test_data_dir) -> Path:
    """Create temporary image file for testing."""
    temp_file = test_data_dir / "test_image.jpg"
    cv2.imwrite(str(temp_file), sample_underwater_image)
    return temp_file

@pytest.fixture
def temp_video_file(test_data_dir) -> Path:
    """Create temporary video file for testing."""
    temp_file = test_data_dir / "test_video.mp4"
    
    # Create simple test video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(temp_file), fourcc, 10.0, (320, 240))
    
    for i in range(30):  # 30 frames
        # Create frame with changing color cast
        frame = np.random.randint(50, 200, (240, 320, 3), dtype=np.uint8)
        frame[:, :, 0] = np.clip(frame[:, :, 0] * (1.2 + 0.1 * np.sin(i/10)), 0, 255)
        frame[:, :, 2] = np.clip(frame[:, :, 2] * 0.6, 0, 255)
        writer.write(frame)
    
    writer.release()
    return temp_file

# Performance testing fixtures
@pytest.fixture
def performance_config():
    """Configuration for performance tests."""
    return {
        'target_fps_lightweight': 30,
        'target_fps_hifi': 15,
        'max_memory_growth_mb': 100,
        'max_processing_time_ms': 100
    }

# API testing fixtures
@pytest.fixture
def mock_api_client():
    """Mock API client for testing."""
    from unittest.mock import MagicMock
    
    client = MagicMock()
    client.enhance_image.return_value = {
        'request_id': 'test-123',
        'processing_stats': {
            'processing_time_ms': 50.0,
            'fps': 20.0,
            'input_resolution': (640, 480)
        }
    }
    
    return client

# Test data generators
def generate_test_dataset(output_dir: Path, num_images: int = 10):
    """Generate synthetic test dataset."""
    output_dir.mkdir(exist_ok=True)
    
    for i in range(num_images):
        # Create varied underwater images
        height, width = np.random.randint(200, 600, 2)
        
        image = np.random.randint(30, 200, (height, width, 3), dtype=np.uint8)
        
        # Vary underwater conditions
        blue_factor = np.random.uniform(1.2, 1.8)
        red_factor = np.random.uniform(0.3, 0.8)
        
        image[:, :, 0] = np.clip(image[:, :, 0] * blue_factor, 0, 255)
        image[:, :, 2] = np.clip(image[:, :, 2] * red_factor, 0, 255)
        
        # Save image
        filename = output_dir / f"underwater_{i:03d}.jpg"
        cv2.imwrite(str(filename), image)

@pytest.fixture
def test_dataset(test_data_dir) -> Path:
    """Create test dataset."""
    dataset_dir = test_data_dir / "dataset"
    generate_test_dataset(dataset_dir, 5)
    return dataset_dir

# Pytest configuration
pytest_plugins = ['pytest_benchmark']

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "gpu: marks tests that require GPU")
    config.addinivalue_line("markers", "api: marks API tests")
    config.addinivalue_line("markers", "performance: marks performance tests")

def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    # Add slow marker to performance tests
    for item in items:
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        if "gpu" in item.nodeid:
            item.add_marker(pytest.mark.gpu)
            
        if "api" in item.nodeid:
            item.add_marker(pytest.mark.api)

# Cleanup fixtures
@pytest.fixture(scope="session", autouse=True)
def cleanup_session():
    """Cleanup resources after test session."""
    yield
    
    # Cleanup any global resources
    import gc
    gc.collect()
    
    # Clear OpenCV windows if any
    try:
        cv2.destroyAllWindows()
    except:
        pass
```

### `tests/test_enhancement.py`
```python
#!/usr/bin/env python3
"""
Unit tests for underwater image enhancement functionality.
"""

import pytest
import numpy as np
import cv2
import time
from unittest.mock import patch, MagicMock

from core.enhancement import UnderwaterImageEnhancer, EnhancementMode, ProcessingStats
from core.metrics import ImageQualityMetrics

class TestUnderwaterImageEnhancer:
    """Test suite for the main enhancement engine."""
    
    def test_initialization_lightweight(self):
        """Test lightweight mode initialization."""
        enhancer = UnderwaterImageEnhancer(mode=EnhancementMode.LIGHTWEIGHT, device='cpu')
        
        assert enhancer.mode == EnhancementMode.LIGHTWEIGHT
        assert enhancer.device == 'cpu'
        assert enhancer.config is not None
    
    def test_initialization_hifi(self):
        """Test high-fidelity mode initialization."""
        enhancer = UnderwaterImageEnhancer(mode=EnhancementMode.HIGH_FIDELITY, device='cpu')
        
        assert enhancer.mode == EnhancementMode.HIGH_FIDELITY
        assert enhancer.device == 'cpu'
    
    def test_single_frame_enhancement(self, enhancer_lightweight, sample_underwater_image):
        """Test single frame enhancement."""
        enhanced, stats = enhancer_lightweight.enhance_frame(sample_underwater_image)
        
        # Check output properties
        assert enhanced.shape == sample_underwater_image.shape
        assert enhanced.dtype == np.uint8
        assert isinstance(stats, ProcessingStats)
        assert stats.processing_time_ms > 0
        assert stats.fps > 0
        assert stats.input_resolution == (640, 480)
    
    def test_enhancement_with_metrics(self, enhancer_lightweight, sample_underwater_image):
        """Test enhancement with quality metrics computation."""
        enhanced, stats = enhancer_lightweight.enhance_frame(
            sample_underwater_image, 
            compute_metrics=True
        )
        
        assert stats.uiqm_score is not None
        assert stats.uciqe_score is not None
        assert stats.uiqm_score >= 0
        assert stats.uciqe_score >= 0
    
    def test_mode_switching(self, sample_underwater_image):
        """Test switching between enhancement modes."""
        enhancer = UnderwaterImageEnhancer(mode=EnhancementMode.LIGHTWEIGHT, device='cpu')
        
        # Test lightweight mode
        enhanced1, stats1 = enhancer.enhance_frame(sample_underwater_image)
        assert enhancer.mode == EnhancementMode.LIGHTWEIGHT
        
        # Switch to high-fidelity mode
        enhancer.switch_mode(EnhancementMode.HIGH_FIDELITY)
        enhanced2, stats2 = enhancer.enhance_frame(sample_underwater_image)
        assert enhancer.mode == EnhancementMode.HIGH_FIDELITY
        
        # Results should be different between modes
        assert not np.array_equal(enhanced1, enhanced2)
    
    def test_performance_consistency(self, enhancer_lightweight, sample_underwater_image):
        """Test performance consistency across multiple runs."""
        processing_times = []
        
        # Warmup
        for _ in range(3):
            enhancer_lightweight.enhance_frame(sample_underwater_image)
        
        # Measure performance
        for _ in range(10):
            _, stats = enhancer_lightweight.enhance_frame(sample_underwater_image)
            processing_times.append(stats.processing_time_ms)
        
        # Check consistency (coefficient of variation < 0.3)
        mean_time = np.mean(processing_times)
        std_time = np.std(processing_times)
        cv = std_time / mean_time
        
        assert cv < 0.3, f"Performance too inconsistent: CV={cv}"
        assert mean_time > 0
    
    def test_different_resolutions(self, enhancer_lightweight):
        """Test enhancement with different image resolutions."""
        resolutions = [(240, 320), (480, 640), (720, 1280)]
        
        for height, width in resolutions:
            img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
            enhanced, stats = enhancer_lightweight.enhance_frame(img)
            
            assert enhanced.shape == img.shape
            assert stats.input_resolution == (width, height)
            assert stats.processing_time_ms > 0
    
    def test_edge_cases(self, enhancer_lightweight):
        """Test edge cases and error handling."""
        # Test with very small image
        tiny_img = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        enhanced, stats = enhancer_lightweight.enhance_frame(tiny_img)
        assert enhanced.shape == tiny_img.shape
        
        # Test with large image
        large_img = np.random.randint(0, 255, (1000, 1000, 3), dtype=np.uint8)
        enhanced, stats = enhancer_lightweight.enhance_frame(large_img)
        assert enhanced.shape == large_img.shape
        
        # Test with invalid input
        with pytest.raises((ValueError, TypeError)):
            enhancer_lightweight.enhance_frame(None)
    
    def test_configuration_override(self):
        """Test configuration parameter override."""
        custom_config = {
            'gamma_value': 1.5,
            'use_lab_color': False,
            'denoise': False
        }
        
        enhancer = UnderwaterImageEnhancer(
            mode=EnhancementMode.LIGHTWEIGHT,
            config=custom_config,
            device='cpu'
        )
        
        assert enhancer.config['gamma_value'] == 1.5
        assert enhancer.config['use_lab_color'] == False
        assert enhancer.config['denoise'] == False
    
    @pytest.mark.slow
    def test_memory_usage(self, enhancer_lightweight, sample_underwater_image):
        """Test memory usage stays reasonable."""
        import psutil
        process = psutil.Process()
        
        # Baseline memory
        baseline_memory = process.memory_info().rss / 1024 / 1024
        
        # Process multiple frames
        for _ in range(50):
            enhancer_lightweight.enhance_frame(sample_underwater_image)
        
        # Check memory after processing
        final_memory = process.memory_info().rss / 1024 / 1024
        memory_growth = final_memory - baseline_memory
        
        # Memory growth should be reasonable (< 200MB for 50 frames)
        assert memory_growth < 200, f"Memory growth too high: {memory_growth}MB"
    
    def test_batch_processing(self, enhancer_lightweight, multiple_test_images):
        """Test batch processing of multiple images."""
        results = []
        total_time = 0
        
        for image in multiple_test_images:
            enhanced, stats = enhancer_lightweight.enhance_frame(image)
            results.append((enhanced, stats))
            total_time += stats.processing_time_ms
        
        # Verify all images processed
        assert len(results) == len(multiple_test_images)
        
        # Verify reasonable performance
        avg_time = total_time / len(multiple_test_images)
        assert avg_time > 0
    
    def test_error_handling(self, enhancer_lightweight):
        """Test error handling with invalid inputs."""
        # Test with wrong data type
        with pytest.raises((ValueError, TypeError)):
            enhancer_lightweight.enhance_frame("not_an_image")
        
        # Test with wrong shape
        wrong_shape = np.random.randint(0, 255, (100, 100), dtype=np.uint8)  # Missing channel dim
        with pytest.raises((ValueError, IndexError)):
            enhancer_lightweight.enhance_frame(wrong_shape)
        
        # Test with wrong dtype
        wrong_dtype = np.random.rand(100, 100, 3).astype(np.float64)
        # Should either work or raise appropriate error
        try:
            result, stats = enhancer_lightweight.enhance_frame(wrong_dtype)
            assert result.dtype == np.uint8
        except (ValueError, TypeError):
            pass  # Expected error

class TestEnhancementModes:
    """Test different enhancement modes."""
    
    def test_mode_comparison(self, sample_underwater_image):
        """Test that different modes produce different results."""
        lightweight = UnderwaterImageEnhancer(EnhancementMode.LIGHTWEIGHT, device='cpu')
        hifi = UnderwaterImageEnhancer(EnhancementMode.HIGH_FIDELITY, device='cpu')
        
        result_light, _ = lightweight.enhance_frame(sample_underwater_image)
        result_hifi, _ = hifi.enhance_frame(sample_underwater_image)
        
        # Results should be different
        assert not np.array_equal(result_light, result_hifi)
        
        # Both should have same shape as input
        assert result_light.shape == sample_underwater_image.shape
        assert result_hifi.shape == sample_underwater_image.shape
    
    @pytest.mark.performance
    def test_performance_targets(self, sample_underwater_image, performance_config):
        """Test that performance targets are met."""
        # Test lightweight mode
        lightweight = UnderwaterImageEnhancer(EnhancementMode.LIGHTWEIGHT, device='cpu')
        
        # Warmup
        for _ in range(3):
            lightweight.enhance_frame(sample_underwater_image)
        
        # Measure performance
        start_time = time.time()
        for _ in range(10):
            _, stats = lightweight.enhance_frame(sample_underwater_image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) * 1000 / 10  # ms per frame
        avg_fps = 1000 / avg_time
        
        # On CPU, we expect lower FPS than GPU targets
        assert avg_fps > 5, f"FPS too low: {avg_fps}"
        assert avg_time < 500, f"Processing time too high: {avg_time}ms"

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_lightweight_performance_benchmark(self, benchmark, sample_underwater_image):
        """Benchmark lightweight mode performance."""
        enhancer = UnderwaterImageEnhancer(EnhancementMode.LIGHTWEIGHT, device='cpu')
        
        def enhance_frame():
            enhanced, stats = enhancer.enhance_frame(sample_underwater_image)
            return enhanced, stats
        
        result = benchmark(enhance_frame)
        enhanced, stats = result
        
        # Verify results
        assert enhanced.shape == sample_underwater_image.shape
        assert stats.fps > 0
    
    def test_hifi_performance_benchmark(self, benchmark, sample_underwater_image):
        """Benchmark high-fidelity mode performance."""
        enhancer = UnderwaterImageEnhancer(EnhancementMode.HIGH_FIDELITY, device='cpu')
        
        def enhance_frame():
            enhanced, stats = enhancer.enhance_frame(sample_underwater_image)
            return enhanced, stats
        
        result = benchmark(enhance_frame)
        enhanced, stats = result
        
        # Verify results  
        assert enhanced.shape == sample_underwater_image.shape
        assert stats.fps > 0
```