### `tests/integration/test_rest_api.py`
```python
#!/usr/bin/env python3
"""
Integration tests for REST API endpoints.
"""

import pytest
import requests
import base64
import json
import numpy as np
import cv2
import io
from pathlib import Path
import time

class TestRestAPIIntegration:
    """Integration tests for REST API."""
    
    @pytest.fixture(scope="class")
    def api_base_url(self):
        """Base URL for API testing."""
        return "http://localhost:8000"
    
    @pytest.fixture(scope="class")
    def api_client(self, api_base_url):
        """HTTP client for API testing."""
        session = requests.Session()
        session.timeout = 30
        return session
    
    def test_health_endpoint(self, api_client, api_base_url):
        """Test health check endpoint."""
        response = api_client.get(f"{api_base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "performance" in data
    
    def test_metrics_endpoint(self, api_client, api_base_url):
        """Test metrics endpoint."""
        response = api_client.get(f"{api_base_url}/metrics")
        
        assert response.status_code == 200
        # Should return Prometheus-format metrics
        assert "uie_" in response.text
    
    def test_config_endpoints(self, api_client, api_base_url):
        """Test configuration endpoints."""
        # Get current config
        response = api_client.get(f"{api_base_url}/config")
        assert response.status_code == 200
        
        config = response.json()
        assert "mode" in config
        assert "device" in config
        
        # Update config
        update_data = {
            "config": {
                "gamma_value": 1.3
            }
        }
        
        response = api_client.put(
            f"{api_base_url}/config",
            json=update_data
        )
        assert response.status_code == 200
    
    def test_single_image_enhancement(self, api_client, api_base_url, sample_underwater_image):
        """Test single image enhancement endpoint."""
        # Encode image
        _, buffer = cv2.imencode('.png', sample_underwater_image)
        image_data = buffer.tobytes()
        
        # Prepare request
        files = {'file': ('test.png', image_data, 'image/png')}
        data = {
            'mode': 'lightweight',
            'compute_metrics': 'true',
            'output_format': 'png'
        }
        
        response = api_client.post(
            f"{api_base_url}/enhance",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "request_id" in result
        assert "enhanced_image" in result
        assert "processing_stats" in result
        
        # Verify processing stats
        stats = result["processing_stats"]
        assert stats["processing_time_ms"] > 0
        assert stats["fps"] > 0
        assert stats["input_resolution"] == [640, 480]
        
        # Verify enhanced image can be decoded
        enhanced_data = base64.b64decode(result["enhanced_image"])
        enhanced_array = np.frombuffer(enhanced_data, np.uint8)
        enhanced_img = cv2.imdecode(enhanced_array, cv2.IMREAD_COLOR)
        
        assert enhanced_img is not None
        assert enhanced_img.shape == sample_underwater_image.shape
    
    def test_batch_enhancement(self, api_client, api_base_url, multiple_test_images):
        """Test batch image enhancement endpoint."""
        files = []
        
        for i, image in enumerate(multiple_test_images):
            _, buffer = cv2.imencode('.png', image)
            image_data = buffer.tobytes()
            files.append(('files', (f'test_{i}.png', image_data, 'image/png')))
        
        data = {
            'mode': 'lightweight',
            'compute_metrics': 'true'
        }
        
        response = api_client.post(
            f"{api_base_url}/enhance/batch",
            files=files,
            data=data
        )
        
        assert response.status_code == 200
        result = response.json()
        
        # Verify response structure
        assert "request_id" in result
        assert "results" in result
        assert "total_processing_time_ms" in result
        assert "success_count" in result
        assert "total_count" in result
        
        # Verify batch results
        assert len(result["results"]) == len(multiple_test_images)
        assert result["success_count"] == len(multiple_test_images)
        assert result["total_count"] == len(multiple_test_images)
        
        # Verify individual results
        for item in result["results"]:
            assert "enhanced_image" in item
            assert "processing_stats" in item
    
    def test_stream_endpoints(self, api_client, api_base_url):
        """Test stream management endpoints."""
        # Start stream (this may fail without actual video source)
        stream_data = {
            'source': '/dev/video0',  # Test with webcam
            'mode': 'lightweight',
            'enable_metrics': False
        }
        
        response = api_client.post(
            f"{api_base_url}/stream/start",
            json=stream_data
        )
        
        # May fail due to no camera, check for appropriate error
        if response.status_code == 200:
            result = response.json()
            stream_id = result["stream_id"]
            
            # Get stream status
            response = api_client.get(f"{api_base_url}/stream/{stream_id}/status")
            assert response.status_code == 200
            
            # Stop stream
            response = api_client.post(f"{api_base_url}/stream/{stream_id}/stop")
            assert response.status_code == 200
        else:
            # Expected if no camera available
            assert response.status_code == 400
    
    def test_error_handling(self, api_client, api_base_url):
        """Test error handling for invalid requests."""
        # Test invalid image format
        response = api_client.post(
            f"{api_base_url}/enhance",
            files={'file': ('test.txt', b'not an image', 'text/plain')}
        )
        assert response.status_code == 400
        
        # Test missing file
        response = api_client.post(f"{api_base_url}/enhance")
        assert response.status_code == 422  # Validation error
        
        # Test invalid mode
        valid_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', valid_image)
        
        response = api_client.post(
            f"{api_base_url}/enhance",
            files={'file': ('test.png', buffer.tobytes(), 'image/png')},
            data={'mode': 'invalid_mode'}
        )
        assert response.status_code == 422
    
    def test_request_limits(self, api_client, api_base_url):
        """Test request size limits and rate limiting."""
        # Create large image (should be within limits)
        large_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', large_image)
        
        response = api_client.post(
            f"{api_base_url}/enhance",
            files={'file': ('large.png', buffer.tobytes(), 'image/png')},
            data={'mode': 'lightweight'}
        )
        
        # Should either succeed or fail with appropriate error
        assert response.status_code in [200, 413, 400]
    
    @pytest.mark.slow
    def test_concurrent_requests(self, api_client, api_base_url, sample_underwater_image):
        """Test handling of concurrent requests."""
        import concurrent.futures
        import threading
        
        def make_request():
            _, buffer = cv2.imencode('.png', sample_underwater_image)
            files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
            data = {'mode': 'lightweight'}
            
            response = api_client.post(f"{api_base_url}/enhance", files=files, data=data)
            return response.status_code
        
        # Make multiple concurrent requests
        num_requests = 5
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(num_requests)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        # All requests should succeed
        success_count = sum(1 for code in results if code == 200)
        assert success_count >= num_requests * 0.8  # Allow some failures under load

class TestAPIAuthentication:
    """Test API authentication and authorization."""
    
    def test_api_without_auth(self, api_client, api_base_url):
        """Test API access without authentication."""
        # Most endpoints should work without auth in test environment
        response = api_client.get(f"{api_base_url}/health")
        assert response.status_code == 200
    
    def test_api_with_auth(self, api_base_url):
        """Test API access with authentication headers."""
        headers = {'Authorization': 'Bearer test-token'}
        
        session = requests.Session()
        session.headers.update(headers)
        
        response = session.get(f"{api_base_url}/health")
        # Should still work (auth may not be enabled in test environment)
        assert response.status_code in [200, 401, 403]

class TestAPIPerformance:
    """Performance tests for API endpoints."""
    
    @pytest.mark.performance
    def test_response_times(self, api_client, api_base_url, sample_underwater_image):
        """Test API response times meet requirements."""
        # Prepare image
        _, buffer = cv2.imencode('.png', sample_underwater_image)
        files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
        data = {'mode': 'lightweight'}
        
        # Warmup
        for _ in range(3):
            api_client.post(f"{api_base_url}/enhance", files=files, data=data)
        
        # Measure response times
        response_times = []
        for _ in range(10):
            start_time = time.time()
            response = api_client.post(f"{api_base_url}/enhance", files=files, data=data)
            end_time = time.time()
            
            assert response.status_code == 200
            response_times.append((end_time - start_time) * 1000)  # ms
        
        avg_response_time = np.mean(response_times)
        max_response_time = np.max(response_times)
        
        # API should respond quickly
        assert avg_response_time < 5000, f"Average response time too high: {avg_response_time}ms"
        assert max_response_time < 10000, f"Max response time too high: {max_response_time}ms"
    
    @pytest.mark.slow
    def test_throughput(self, api_client, api_base_url, sample_underwater_image):
        """Test API throughput under sustained load."""
        # Prepare image
        _, buffer = cv2.imencode('.png', sample_underwater_image)
        files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
        data = {'mode': 'lightweight'}
        
        # Sustained requests
        num_requests = 20
        start_time = time.time()
        
        successful_requests = 0
        for _ in range(num_requests):
            try:
                response = api_client.post(
                    f"{api_base_url}/enhance", 
                    files=files, 
                    data=data,
                    timeout=30
                )
                if response.status_code == 200:
                    successful_requests += 1
            except requests.exceptions.Timeout:
                pass  # Count as failed request
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Calculate throughput
        throughput = successful_requests / duration
        success_rate = successful_requests / num_requests
        
        assert throughput > 0.1, f"Throughput too low: {throughput} req/s"
        assert success_rate > 0.8, f"Success rate too low: {success_rate}"
```

### `.github/workflows/ci.yml`
```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main, develop]
    tags: [v*]
  pull_request:
    branches: [main]

env:
  REGISTRY: registry.drdo.gov.in/maritime-ai
  IMAGE_NAME: underwater-image-enhancement

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.11]
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        fetch-depth: 0
        lfs: true
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ matrix.python-version }}-${{ hashFiles('**/requirements*.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-${{ matrix.python-version }}-
          ${{ runner.os }}-pip-
    
    - name: Install system dependencies
      run: |
        sudo apt-get update
        sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1
    
    - name: Install Python dependencies
      run: |
        python -m pip install --upgrade pip setuptools wheel
        pip install -r requirements-dev.txt
        pip install -e .
    
    - name: Lint with flake8
      run: |
        flake8 src/ tests/ --count --select=E9,F63,F7,F82 --show-source --statistics
        flake8 src/ tests/ --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
    
    - name: Format check with black
      run: black --check src/ tests/
    
    - name: Import sort check with isort
      run: isort --check-only src/ tests/
    
    - name: Type checking with mypy
      run: mypy src/ --ignore-missing-imports
    
    - name: Security check with bandit
      run: bandit -r src/ -f json -o bandit-report.json
      continue-on-error: true
    
    - name: Dependency vulnerability check
      run: safety check --json --output safety-report.json
      continue-on-error: true
    
    - name: Run unit tests with coverage
      run: |
        pytest tests/ -v \
          --cov=src \
          --cov-report=xml \
          --cov-report=term-missing \
          --cov-report=html \
          --junitxml=pytest-results.xml \
          -m "not slow and not integration and not gpu"
    
    - name: Upload test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: test-results-${{ matrix.python-version }}
        path: |
          pytest-results.xml
          htmlcov/
          bandit-report.json
          safety-report.json
    
    - name: Upload coverage to Codecov
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella
        fail_ci_if_error: false

  build-and-scan:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' || github.event_name == 'pull_request'
    
    permissions:
      contents: read
      packages: write
      security-events: write
    
    strategy:
      matrix:
        platform: [linux/amd64, linux/arm64]
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
      with:
        lfs: true
    
    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v3
    
    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ secrets.REGISTRY_USERNAME }}
        password: ${{ secrets.REGISTRY_PASSWORD }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
        tags: |
          type=ref,event=branch
          type=ref,event=pr
          type=semver,pattern={{version}}
          type=semver,pattern={{major}}.{{minor}}
          type=sha,prefix={{branch}}-
          type=raw,value=latest,enable={{is_default_branch}}
    
    - name: Build Docker image
      uses: docker/build-push-action@v5
      id: build
      with:
        context: .
        file: docker/Dockerfile.base
        platforms: ${{ matrix.platform }}
        push: ${{ github.event_name != 'pull_request' }}
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
        cache-from: type=gha
        cache-to: type=gha,mode=max
        outputs: type=image,name=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }},push-by-digest=true,name-canonical=true,push=${{ github.event_name != 'pull_request' }}
    
    - name: Export digest
      run: |
        mkdir -p /tmp/digests
        digest="${{ steps.build.outputs.digest }}"
        touch "/tmp/digests/${digest#sha256:}"
    
    - name: Upload digest
      uses: actions/upload-artifact@v3
      with:
        name: digests-${{ matrix.platform }}
        path: /tmp/digests/*
        if-no-files-found: error
        retention-days: 1
    
    - name: Run Trivy vulnerability scanner
      uses: aquasecurity/trivy-action@master
      if: matrix.platform == 'linux/amd64'
      with:
        image-ref: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:${{ steps.meta.outputs.version }}
        format: 'sarif'
        output: 'trivy-results.sarif'
    
    - name: Upload Trivy scan results to GitHub Security
      uses: github/codeql-action/upload-sarif@v2
      if: matrix.platform == 'linux/amd64'
      with:
        sarif_file: 'trivy-results.sarif'

  generate-sbom:
    needs: [lint-and-test, build-and-scan]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Download all artifacts
      uses: actions/download-artifact@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install SBOM tools
      run: |
        pip install cyclonedx-bom
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    
    - name: Generate Python SBOM
      run: |
        cyclonedx-bom -o sbom/python-sbom.json .
    
    - name: Generate container SBOM
      run: |
        syft ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest -o spdx-json > sbom/container-sbom.spdx.json
        syft ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest -o cyclonedx-json > sbom/container-sbom.cdx.json
    
    - name: Upload SBOM artifacts
      uses: actions/upload-artifact@v3
      with:
        name: sbom-files
        path: sbom/
    
    - name: Sign SBOM with Cosign
      uses: sigstore/cosign-installer@v3
    
    - name: Sign SBOM files
      env:
        COSIGN_EXPERIMENTAL: 1
      run: |
        for file in sbom/*.json; do
          cosign sign-blob --yes "$file" --output-signature "${file}.sig"
        done

  integration-tests:
    needs: [build-and-scan]
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/develop'
    
    services:
      uie-api:
        image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop
        ports:
          - 8000:8000
        env:
          CUDA_VISIBLE_DEVICES: ""
          UIE_LOG_LEVEL: INFO
        options: --health-cmd "curl -f http://localhost:8000/health || exit 1" --health-interval 30s --health-timeout 10s --health-retries 5
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install test dependencies
      run: |
        pip install -r requirements-dev.txt
        pip install requests pytest-xdist
    
    - name: Wait for API to be ready
      run: |
        timeout 60s bash -c 'until curl -f http://localhost:8000/health; do sleep 5; done'
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v --tb=short -x
    
    - name: Run API load tests
      run: |
        python tests/load/test_load_rest.py --base-url http://localhost:8000
    
    - name: Upload integration test results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: integration-test-results
        path: integration-test-results.xml

  security-scan:
    needs: [build-and-scan]
    runs-on: ubuntu-latest
    if: github.event_name == 'push'
    
    permissions:
      security-events: write
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Run comprehensive security scan
      run: |
        # Create security scan script
        cat > security_scan.sh << 'EOF'
        #!/bin/bash
        set -e
        
        echo "Running comprehensive security scan..."
        
        # Container image scanning
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
          -v $PWD:/workspace \
          aquasec/trivy:latest image \
          --format json --output /workspace/trivy-report.json \
          ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:latest
        
        # Source code scanning
        docker run --rm -v $PWD:/workspace \
          securecodewarrior/docker-semgrep \
          --config=auto --json --output=/workspace/semgrep-report.json /workspace
        
        echo "Security scans completed"
        EOF
        
        chmod +x security_scan.sh
        ./security_scan.sh
    
    - name: Upload security scan results
      uses: actions/upload-artifact@v3
      with:
        name: security-scan-results
        path: |
          trivy-report.json
          semgrep-report.json

  deploy-staging:
    needs: [integration-tests, security-scan]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    environment: staging
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_STAGING }}" | base64 -d > kubeconfig
        echo "KUBECONFIG=kubeconfig" >> $GITHUB_ENV
    
    - name: Deploy to staging
      run: |
        # Update image tag in deployment
        sed -i "s|image: .*|image: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:develop|g" deployment/k8s/deployment.yaml
        
        # Apply manifests
        kubectl apply -f deployment/k8s/namespace.yaml
        kubectl apply -f deployment/k8s/configmap.yaml
        kubectl apply -f deployment/k8s/secret.yaml
        kubectl apply -f deployment/k8s/deployment.yaml
        kubectl apply -f deployment/k8s/service.yaml
        
        # Wait for deployment
        kubectl rollout status deployment/uie-api -n uie-system --timeout=300s
    
    - name: Run smoke tests
      run: |
        # Get service endpoint
        STAGING_IP=$(kubectl get svc uie-api-service -n uie-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        # Wait for service to be ready
        sleep 30
        
        # Run smoke tests
        curl -f http://$STAGING_IP/health
        
        # Test basic image enhancement
        python -c "
        import requests
        import numpy as np
        import cv2
        
        # Create test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', img)
        
        # Test API
        files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
        data = {'mode': 'lightweight'}
        
        response = requests.post(f'http://$STAGING_IP/enhance', files=files, data=data, timeout=30)
        assert response.status_code == 200
        print('Smoke test passed!')
        "

  deploy-production:
    needs: [deploy-staging]
    runs-on: ubuntu-latest
    if: startsWith(github.ref, 'refs/tags/v')
    environment: production
    
    steps:
    - name: Checkout repository
      uses: actions/checkout@v4
    
    - name: Set up Helm
      uses: azure/setup-helm@v3
      with:
        version: '3.12.0'
    
    - name: Configure kubectl
      run: |
        echo "${{ secrets.KUBE_CONFIG_PROD }}" | base64 -d > kubeconfig
        echo "KUBECONFIG=kubeconfig" >> $GITHUB_ENV
    
    - name: Extract version
      id: version
      run: echo "VERSION=${GITHUB_REF#refs/tags/v}" >> $GITHUB_OUTPUT
    
    - name: Deploy to production using Helm
      run: |
        helm upgrade --install uie-system deployment/helm/ \
          --namespace uie-system --create-namespace \
          --set image.tag=${{ steps.version.outputs.VERSION }} \
          --set image.repository=${{ env.REGISTRY }}/${{ env.IMAGE_NAME }} \
          --set replicaCount=5 \
          --set autoscaling.enabled=true \
          --set autoscaling.minReplicas=3 \
          --set autoscaling.maxReplicas=20 \
          --wait --timeout=600s
    
    - name: Verify production deployment
      run: |
        kubectl get pods -n uie-system
        kubectl get svc -n uie-system
        
        # Wait for all pods to be ready
        kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=underwater-image-enhancement -n uie-system --timeout=300s
    
    - name: Run production health checks
      run: |
        PROD_IP=$(kubectl get svc uie-api-service -n uie-system -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
        
        # Comprehensive health check
        curl -f http://$PROD_IP/health
        curl -f http://$PROD_IP/metrics
        
        # Performance validation
        python -c "
        import requests
        import time
        import numpy as np
        import cv2
        
        # Performance test
        img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        _, buffer = cv2.imencode('.png', img)
        
        files = {'file': ('test.png', buffer.tobytes(), 'image/png')}
        data = {'mode': 'lightweight'}
        
        # Warm up
        for _ in range(3):
            requests.post(f'http://$PROD_IP/enhance', files=files, data=data, timeout=30)
        
        # Measure performance
        times = []
        for _ in range(5):
            start = time.time()
            response = requests.post(f'http://$PROD_IP/enhance', files=files, data=data, timeout=30)
            end = time.time()
            assert response.status_code == 200
            times.append(end - start)
        
        avg_time = sum(times) / len(times)
        print(f'Average response time: {avg_time:.2f}s')
        assert avg_time < 5.0, f'Response time too high: {avg_time}s'
        print('Production health check passed!')
        "

  notify:
    needs: [deploy-production]
    runs-on: ubuntu-latest
    if: always()
    
    steps:
    - name: Notify deployment status
      run: |
        if [[ "${{ needs.deploy-production.result }}" == "success" ]]; then
          echo "✅ Production deployment successful"
          # Add notification logic (Slack, email, etc.)
        else
          echo "❌ Production deployment failed"
          # Add error notification logic
        fi
```

### `scripts/setup.sh`
```bash
#!/bin/bash
set -euo pipefail

# Setup script for Underwater Image Enhancement System
# DRDO Maritime AI Systems

echo "🌊 Setting up Underwater Image Enhancement System"
echo "=================================================="

# Configuration
PYTHON_VERSION=${PYTHON_VERSION:-3.9}
CUDA_VERSION=${CUDA_VERSION:-12.1}
INSTALL_DEV=${INSTALL_DEV:-false}
INSTALL_GPU=${INSTALL_GPU:-auto}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check system requirements
check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_success "Operating System: Linux"
        
        # Check Ubuntu version
        if command_exists lsb_release; then
            UBUNTU_VERSION=$(lsb_release -rs)
            log_info "Ubuntu version: $UBUNTU_VERSION"
            
            if [[ $(echo "$UBUNTU_VERSION >= 20.04" | bc -l) -eq 1 ]]; then
                log_success "Ubuntu version is supported"
            else
                log_warn "Ubuntu version might not be fully supported"
            fi
        fi
    else
        log_warn "Non-Linux OS detected. Some features may not work correctly."
    fi
    
    # Check architecture
    ARCH=$(uname -m)
    log_info "Architecture: $ARCH"
    
    if [[ "$ARCH" == "x86_64" ]] || [[ "$ARCH" == "aarch64" ]]; then
        log_success "Architecture is supported"
    else
        log_warn "Architecture might not be fully supported"
    fi
    
    # Check available memory
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    log_info "Total memory: ${TOTAL_MEM}GB"
    
    if [[ $TOTAL_MEM -lt 8 ]]; then
        log_warn "System has less than 8GB RAM. Performance may be impacted."
    else
        log_success "Memory requirements met"
    fi
}

# Function to detect GPU
detect_gpu() {
    log_info "Detecting GPU..."
    
    if command_exists nvidia-smi; then
        if nvidia-smi >/dev/null 2>&1; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits)
            log_success "NVIDIA GPU detected:"
            echo "$GPU_INFO" | while IFS= read -r line; do
                log_info "  $line"
            done
            return 0
        else
            log_warn "nvidia-smi found but not working properly"
        fi
    else
        log_info "No NVIDIA GPU detected"
    fi
    
    return 1
}

# Function to install system dependencies
install_system_dependencies() {
    log_info "Installing system dependencies..."
    
    # Update package list
    sudo apt-get update
    
    # Install basic dependencies
    sudo apt-get install -y \
        curl \
        wget \
        git \
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
        python3-dev \
        python3-pip \
        python3-venv \
        ffmpeg \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1
    
    log_success "System dependencies installed"
}

# Function to setup Python environment
setup_python_environment() {
    log_info "Setting up Python environment..."
    
    # Check Python version
    if command_exists python3; then
        CURRENT_PYTHON=$(python3 --version | awk '{print $2}' | cut -d. -f1,2)
        log_info "Current Python version: $CURRENT_PYTHON"
        
        REQUIRED_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        REQUIRED_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        CURRENT_MAJOR=$(echo $CURRENT_PYTHON | cut -d. -f1)
        CURRENT_MINOR=$(echo $CURRENT_PYTHON | cut -d. -f2)
        
        if [[ $CURRENT_MAJOR -eq $REQUIRED_MAJOR ]] && [[ $CURRENT_MINOR -ge $REQUIRED_MINOR ]]; then
            log_success "Python version is compatible"
        else
            log_warn "Python version might not be fully compatible"
        fi
    else
        log_error "Python 3 not found"
        return 1
    fi
    
    # Create virtual environment
    log_info "Creating virtual environment..."
    if [[ ! -d "venv" ]]; then
        python3 -m venv venv
        log_success "Virtual environment created"
    else
        log_info "Virtual environment already exists"
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    log_success "Python environment ready"
}

# Function to install CUDA (if requested)
install_cuda() {
    if [[ "$INSTALL_GPU" != "true" ]]; then
        return 0
    fi
    
    log_info "Installing CUDA $CUDA_VERSION..."
    
    # Check if CUDA is already installed
    if command_exists nvcc; then
        CURRENT_CUDA=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
        log_info "Current CUDA version: $CURRENT_CUDA"
        
        if [[ "$CURRENT_CUDA" == "$CUDA_VERSION"* ]]; then
            log_success "CUDA is already installed"
            return 0
        fi
    fi
    
    # Download and install CUDA
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    CUDA_PACKAGE="cuda-${CUDA_MAJOR}-${CUDA_MINOR}"
    
    # Add NVIDIA package repository
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    sudo dpkg -i cuda-keyring_1.0-1_all.deb
    sudo apt-get update
    
    # Install CUDA
    sudo apt-get install -y $CUDA_PACKAGE
    
    # Set up environment variables
    echo "export PATH=/usr/local/cuda-$CUDA_VERSION/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda-$CUDA_VERSION/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
    
    log_success "CUDA installed"
}

# Function to install Python dependencies
install_python_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Ensure virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Install PyTorch with CUDA support if GPU available
    if detect_gpu && [[ "$INSTALL_GPU" != "false" ]]; then
        log_info "Installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        log_info "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install main requirements
    log_info "Installing main requirements..."
    pip install -r requirements.txt
    
    # Install development dependencies if requested
    if [[ "$INSTALL_DEV" == "true" ]]; then
        log_info "Installing development dependencies..."
        pip install -r requirements-dev.txt
    fi
    
    # Install package in development mode
    log_info "Installing package in development mode..."
    pip install -e .
    
    log_success "Python dependencies installed"
}

# Function to download and setup models
setup_models() {
    log_info "Setting up models..."
    
    # Create model directories
    mkdir -p models/weights
    mkdir -p models/cache
    
    # Download or generate embedded weights
    log_info "Setting up embedded model weights..."
    python3 -c "
import sys
sys.path.append('src')
from core.models import UNetLite, LUTModel

# Initialize models to trigger weight creation
unet = UNetLite(device='cpu')
lut = LUTModel(device='cpu')

# Load embedded weights
unet.load_embedded_weights()
lut.load_embedded_weights()

print('Model weights initialized')
"
    
    log_success "Models setup complete"
}

# Function to run tests
run_tests() {
    log_info "Running tests..."
    
    # Ensure virtual environment is activated
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        source venv/bin/activate
    fi
    
    # Run basic tests
    pytest tests/ -v -m "not slow and not integration and not gpu" --tb=short
    
    log_success "Tests completed"
}

# Function to create sample data
create_sample_data() {
    log_info "Creating sample data..."
    
    # Create sample directories
    mkdir -p samples/input
    mkdir -p samples/output
    mkdir -p samples/calibration/images
    
    # Generate sample underwater images
    python3 -c "
import numpy as np
import cv2
import os

# Create sample underwater images
for i in range(5):
    # Create realistic underwater-looking image
    height, width = np.random.randint(400, 800, 2)
    
    # Base image
    img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
    
    # Add underwater color cast
    img[:, :, 0] = np.clip(img[:, :, 0] * 1.4, 0, 255)  # More blue
    img[:, :, 1] = np.clip(img[:, :, 1] * 1.1, 0, 255)  # Slight green
    img[:, :, 2] = np.clip(img[:, :, 2] * 0.6, 0, 255)  # Less red
    
    # Add some structure
    cv2.rectangle(img, (50, 50), (150, 150), (80, 120, 160), -1)
    cv2.circle(img, (width//2, height//2), 50, (60, 100, 140), -1)
    
    # Save sample image
    cv2.imwrite(f'samples/input/underwater_sample_{i+1:02d}.jpg', img)

print('Sample data created')
"
    
    # Create README for samples
    cat > samples/README.md << EOF
# Sample Data

This directory contains sample underwater images for testing the enhancement system.

## Input Images
- \`input/\`: Sample underwater images with typical underwater characteristics
  - Blue/green color cast
  - Reduced contrast
  - Hazy appearance

## Usage

### CLI Enhancement
\`\`\`bash
# Enhance single image
uie enhance -i samples/input/underwater_sample_01.jpg -o samples/output/enhanced_01.jpg --metrics

# Enhance all samples
for img in samples/input/*.jpg; do
    basename=\$(basename "\$img" .jpg)
    uie enhance -i "\$img" -o "samples/output/\${basename}_enhanced.jpg" --metrics
done
\`\`\`

### API Enhancement
\`\`\`python
from src.sdk.python.client import UnderwaterEnhancementClient
import cv2

client = UnderwaterEnhancementClient()

# Load and enhance image
image = cv2.imread('samples/input/underwater_sample_01.jpg')
result = client.enhance_image(image, mode='lightweight', compute_metrics=True)

print(f"UIQM Score: {result['processing_stats']['uiqm_score']}")
\`\`\`
EOF
    
    log_success "Sample data created"
}

# Function to setup development environment
setup_development() {
    if [[ "$INSTALL_DEV" != "true" ]]; then
        return 0
    fi
    
    log_info "Setting up development environment..."
    
    # Install pre-commit hooks
    if command_exists pre-commit; then
        pre-commit install
        log_success "Pre-commit hooks installed"
    fi
    
    # Create VS Code settings
    mkdir -p .vscode
    cat > .vscode/settings.json << EOF
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.mypyEnabled": true,
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests",
        "-v"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
EOF
    
    # Create development aliases
    cat >> ~/.bashrc << EOF

# Underwater Image Enhancement System aliases
alias uie-activate='source $(pwd)/venv/bin/activate'
alias uie-test='pytest tests/ -v'
alias uie-lint='flake8 src/ tests/ && black --check src/ tests/ && isort --check-only src/ tests/'
alias uie-format='black src/ tests/ && isort src/ tests/'
alias uie-serve='python -m src.api.rest_server'
EOF
    
    log_success "Development environment configured"
}

# Function to display final instructions
display_final_instructions() {
    log_success "🎉 Setup completed successfully!"
    echo ""
    echo "==============================================="
    echo "🌊 Underwater Image Enhancement System Ready"
    echo "==============================================="
    echo ""
    echo "📋 Next Steps:"
    echo "1. Activate virtual environment: source venv/bin/activate"
    echo "2. Test installation: uie --help"
    echo "3. Try sample enhancement: uie enhance -i samples/input/underwater_sample_01.jpg -o enhanced.jpg --metrics"
    echo "4. Start API server: uie serve"
    echo ""
    echo "🔗 Quick Links:"
    echo "- Documentation: docs/"
    echo "- Sample data: samples/"
    echo "- Configuration: configs/"
    echo "- Tests: pytest tests/"
    echo ""
    
    if detect_gpu; then
        echo "🚀 GPU acceleration is available!"
        echo "- CUDA version: $(nvcc --version | grep "release" | awk '{print $6}' | cut -c2- || echo 'N/A')"
        echo "- GPU info: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || echo 'N/A')"
    else
        echo "💻 Running in CPU mode"
        echo "- For GPU acceleration, install NVIDIA drivers and CUDA"
    fi
    
    echo ""
    echo "📚 For detailed documentation, see: docs/quickstart/"
    echo "🔧 For configuration options, see: configs/default.yaml"
    echo ""
}

# Main setup function
main() {
    log_info "Starting setup process..."
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            --gpu)
                INSTALL_GPU=true
                shift
                ;;
            --no-gpu)
                INSTALL_GPU=false
                shift
                ;;
            --cuda-version)
                CUDA_VERSION="$2"
                shift 2
                ;;
            --python-version)
                PYTHON_VERSION="$2"
                shift 2
                ;;
            -h|--help)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  --dev              Install development dependencies"
                echo "  --gpu              Force GPU installation"
                echo "  --no-gpu           Disable GPU installation"
                echo "  --cuda-version     Specify CUDA version (default: 12.1)"
                echo "  --python-version   Specify Python version (default: 3.9)"
                echo "  -h, --help         Show this help message"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Auto-detect GPU if not specified
    if [[ "$INSTALL_GPU" == "auto" ]]; then
        if detect_gpu; then
            INSTALL_GPU=true
            log_info "GPU detected, enabling GPU support"
        else
            INSTALL_GPU=false
            log_info "No GPU detected, using CPU only"
        fi
    fi
    
    # Run setup steps
    check_system_requirements
    install_system_dependencies
    setup_python_environment
    
    if [[ "$INSTALL_GPU" == "true" ]]; then
        install_cuda
    fi
    
    install_python_dependencies
    setup_models
    create_sample_data
    
    if [[ "$INSTALL_DEV" == "true" ]]; then
        setup_development
    fi
    
    run_tests
    display_final_instructions
}

# Run main function
main "$@"
```