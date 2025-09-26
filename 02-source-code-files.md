## Source Code Files

### `src/__init__.py`
```python
"""
Underwater Image Enhancement System for Maritime Security
Copyright (c) 2025 DRDO India
"""

__version__ = "1.0.0"
__author__ = "DRDO Maritime AI Systems"
__email__ = "maritime-ai@drdo.gov.in"

from .core import UnderwaterImageEnhancer, EnhancementMode
from .core.metrics import ImageQualityMetrics

__all__ = ["UnderwaterImageEnhancer", "EnhancementMode", "ImageQualityMetrics"]
```

### `src/core/__init__.py`
```python
"""Core underwater image enhancement modules."""

from .enhancement import UnderwaterImageEnhancer, EnhancementMode, ProcessingStats
from .models import UNetLite, LUTModel
from .metrics import ImageQualityMetrics
from .streaming import StreamProcessor
from .utils import setup_logging, load_config

__all__ = [
    "UnderwaterImageEnhancer",
    "EnhancementMode", 
    "ProcessingStats",
    "UNetLite",
    "LUTModel",
    "ImageQualityMetrics",
    "StreamProcessor",
    "setup_logging",
    "load_config"
]
```

### `src/core/utils.py`
```python
#!/usr/bin/env python3
"""
Utility functions for underwater image enhancement system.
"""

import os
import yaml
import json
import logging
import cv2
import numpy as np
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import structlog

def setup_logging(level: str = "INFO", format: str = "json") -> None:
    """
    Setup structured logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        format: Log format (json, console)
    """
    log_level = getattr(logging, level.upper())
    
    if format == "json":
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        config_path = os.environ.get('UIE_CONFIG_FILE', 'configs/default.yaml')
    
    config_file = Path(config_path)
    
    # Default configuration
    default_config = {
        'default_mode': 'lightweight',
        'device': 'auto',
        'gamma_value': 1.2,
        'use_lab_color': True,
        'denoise': True,
        'white_balance': {
            'method': 'underwater_physics',
            'adaptation_strength': 0.8
        },
        'guided_filter': {
            'radius': 8,
            'eps': 0.01
        },
        'dehazing': {
            'beta': 1.0,
            'tx': 0.1,
            'use_dark_channel': True
        }
    }
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)
                default_config.update(file_config)
        except Exception as e:
            logging.warning(f"Failed to load config file {config_file}: {e}")
    
    return default_config

def validate_image(image: np.ndarray) -> bool:
    """
    Validate input image format and properties.
    
    Args:
        image: Input image array
        
    Returns:
        True if valid, False otherwise
    """
    if image is None:
        return False
        
    if not isinstance(image, np.ndarray):
        return False
        
    if len(image.shape) != 3:
        return False
        
    if image.shape[2] != 3:
        return False
        
    if image.dtype != np.uint8:
        return False
        
    return True

def resize_image(image: np.ndarray, target_size: Tuple[int, int], 
                 maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image while optionally maintaining aspect ratio.
    
    Args:
        image: Input image
        target_size: Target (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    if maintain_aspect:
        # Calculate scaling factor
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        
        # Calculate new dimensions
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        
        # Create canvas and center image
        canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        
        y_offset = (target_height - new_height) // 2
        x_offset = (target_width - new_width) // 2
        
        canvas[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized
        
        return canvas
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def create_batches(items: list, batch_size: int):
    """
    Create batches from list of items.
    
    Args:
        items: List of items to batch
        batch_size: Size of each batch
        
    Yields:
        Batches of items
    """
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes to human readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} PB"

def get_device_info() -> Dict[str, Any]:
    """
    Get information about available compute devices.
    
    Returns:
        Device information dictionary
    """
    device_info = {
        'cpu_count': os.cpu_count(),
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': []
    }
    
    try:
        import torch
        if torch.cuda.is_available():
            device_info['cuda_available'] = True
            device_info['gpu_count'] = torch.cuda.device_count()
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                device_info['gpu_names'].append(props.name)
                device_info['gpu_memory'].append(props.total_memory)
    except ImportError:
        pass
    
    return device_info

def save_image_with_metadata(image: np.ndarray, filepath: str, 
                            metadata: Dict[str, Any] = None) -> bool:
    """
    Save image with optional metadata.
    
    Args:
        image: Image to save
        filepath: Output file path
        metadata: Optional metadata dictionary
        
    Returns:
        True if successful
    """
    try:
        # Save image
        success = cv2.imwrite(filepath, image)
        
        if success and metadata:
            # Save metadata as sidecar JSON file
            metadata_path = Path(filepath).with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return success
    except Exception as e:
        logging.error(f"Failed to save image {filepath}: {e}")
        return False

def create_grid_image(images: list, grid_size: Tuple[int, int] = None) -> np.ndarray:
    """
    Create a grid image from list of images.
    
    Args:
        images: List of images
        grid_size: (rows, cols) for grid, auto-calculated if None
        
    Returns:
        Grid image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Calculate grid size if not provided
    if grid_size is None:
        n_images = len(images)
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
        grid_size = (rows, cols)
    
    rows, cols = grid_size
    
    # Ensure all images have same size
    reference_shape = images[0].shape
    resized_images = []
    
    for img in images:
        if img.shape != reference_shape:
            img = cv2.resize(img, (reference_shape[1], reference_shape[0]))
        resized_images.append(img)
    
    # Pad with blank images if needed
    while len(resized_images) < rows * cols:
        blank = np.zeros(reference_shape, dtype=np.uint8)
        resized_images.append(blank)
    
    # Create grid
    grid_rows = []
    for row in range(rows):
        start_idx = row * cols
        end_idx = start_idx + cols
        row_images = resized_images[start_idx:end_idx]
        grid_row = np.hstack(row_images)
        grid_rows.append(grid_row)
    
    grid_image = np.vstack(grid_rows)
    return grid_image

class Timer:
    """Context manager for timing operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        self.elapsed = self.end_time - self.start_time
        
    def __str__(self):
        if self.elapsed:
            return f"{self.name}: {self.elapsed:.3f}s"
        return f"{self.name}: Not completed"

class MemoryTracker:
    """Track memory usage during operations."""
    
    def __init__(self):
        self.initial_memory = None
        self.peak_memory = None
        self.current_memory = None
    
    def start(self):
        """Start memory tracking."""
        import psutil
        process = psutil.Process()
        self.initial_memory = process.memory_info().rss
        self.peak_memory = self.initial_memory
    
    def update(self):
        """Update memory tracking."""
        import psutil
        process = psutil.Process()
        self.current_memory = process.memory_info().rss
        self.peak_memory = max(self.peak_memory, self.current_memory)
    
    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics in MB."""
        mb = 1024 * 1024
        return {
            'initial_mb': self.initial_memory / mb if self.initial_memory else 0,
            'current_mb': self.current_memory / mb if self.current_memory else 0,
            'peak_mb': self.peak_memory / mb if self.peak_memory else 0,
            'growth_mb': (self.current_memory - self.initial_memory) / mb if self.initial_memory and self.current_memory else 0
        }
```

### `src/core/streaming.py`
```python
#!/usr/bin/env python3
"""
Streaming and video processing utilities for underwater image enhancement.
Handles RTSP streams, USB cameras, and video files.
"""

import cv2
import numpy as np
import threading
import queue
import time
import logging
from typing import Optional, Callable, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import subprocess
import os

logger = logging.getLogger(__name__)

class StreamState(Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

@dataclass
class StreamInfo:
    """Information about video stream."""
    width: int
    height: int
    fps: float
    frame_count: Optional[int] = None
    duration: Optional[float] = None
    codec: Optional[str] = None

class StreamProcessor:
    """
    High-performance video stream processor for underwater imagery.
    Supports RTSP streams, USB cameras, and video files.
    """
    
    def __init__(self, source: Union[str, int], buffer_size: int = 10):
        """
        Initialize stream processor.
        
        Args:
            source: Video source (file path, RTSP URL, camera index)
            buffer_size: Frame buffer size
        """
        self.source = source
        self.buffer_size = buffer_size
        self.state = StreamState.STOPPED
        
        # Threading components
        self.capture_thread = None
        self.processing_thread = None
        self.frame_queue = queue.Queue(maxsize=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        
        # Stream info
        self.stream_info = None
        self.capture = None
        
        # Statistics
        self.frames_captured = 0
        self.frames_processed = 0
        self.frames_dropped = 0
        self.start_time = None
        
        # Callbacks
        self.frame_callback = None
        self.error_callback = None
        
    def open(self) -> bool:
        """
        Open video stream.
        
        Returns:
            True if successful
        """
        try:
            # Configure OpenCV capture
            self.capture = cv2.VideoCapture(self.source)
            
            if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                # RTSP stream optimizations
                self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
            
            if not self.capture.isOpened():
                logger.error(f"Failed to open video source: {self.source}")
                return False
            
            # Get stream information
            width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Handle invalid FPS values
            if fps <= 0 or fps > 1000:
                fps = 30.0  # Default FPS
                
            self.stream_info = StreamInfo(
                width=width,
                height=height,
                fps=fps,
                frame_count=frame_count if frame_count > 0 else None
            )
            
            logger.info(f"Stream opened: {width}x{height}@{fps}fps")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open stream: {e}")
            return False
    
    def start_capture(self, frame_callback: Callable = None, 
                     enhancement_function: Callable = None) -> bool:
        """
        Start capturing and processing frames.
        
        Args:
            frame_callback: Callback for processed frames
            enhancement_function: Function to enhance frames
            
        Returns:
            True if started successfully
        """
        if not self.capture or not self.capture.isOpened():
            logger.error("Stream not opened")
            return False
        
        self.frame_callback = frame_callback
        self.enhancement_function = enhancement_function
        self.stop_event.clear()
        self.state = StreamState.STARTING
        
        # Start capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Start processing thread if enhancement function provided
        if enhancement_function:
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        
        self.start_time = time.time()
        self.state = StreamState.RUNNING
        
        logger.info("Stream capture started")
        return True
    
    def stop_capture(self):
        """Stop capturing frames."""
        self.state = StreamState.STOPPED
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5.0)
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=5.0)
        
        logger.info("Stream capture stopped")
    
    def close(self):
        """Close video stream and cleanup."""
        self.stop_capture()
        
        if self.capture:
            self.capture.release()
            self.capture = None
        
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.result_queue)
        
        logger.info("Stream closed")
    
    def _capture_loop(self):
        """Main capture loop running in separate thread."""
        logger.debug("Capture loop started")
        
        while not self.stop_event.is_set():
            try:
                ret, frame = self.capture.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    # Try to reconnect for RTSP streams
                    if isinstance(self.source, str) and self.source.startswith(('rtsp://', 'http://')):
                        self._reconnect_stream()
                    continue
                
                self.frames_captured += 1
                
                # Add frame to queue
                try:
                    if self.enhancement_function:
                        # Add to processing queue
                        self.frame_queue.put_nowait((frame, time.time()))
                    else:
                        # Direct callback
                        if self.frame_callback:
                            self.frame_callback(frame, None)
                            
                except queue.Full:
                    # Drop frame if queue is full
                    self.frames_dropped += 1
                    
            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                self.state = StreamState.ERROR
                if self.error_callback:
                    self.error_callback(e)
                break
        
        logger.debug("Capture loop ended")
    
    def _processing_loop(self):
        """Frame processing loop running in separate thread."""
        logger.debug("Processing loop started")
        
        while not self.stop_event.is_set():
            try:
                # Get frame from queue
                frame, timestamp = self.frame_queue.get(timeout=1.0)
                
                # Process frame
                enhanced_frame, stats = self.enhancement_function(frame)
                self.frames_processed += 1
                
                # Add processing timestamp
                processing_time = time.time() - timestamp
                
                # Callback with enhanced frame
                if self.frame_callback:
                    self.frame_callback(enhanced_frame, stats)
                
                self.frame_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Processing loop error: {e}")
                break
        
        logger.debug("Processing loop ended")
    
    def _reconnect_stream(self):
        """Attempt to reconnect RTSP stream."""
        logger.info("Attempting to reconnect stream...")
        
        if self.capture:
            self.capture.release()
        
        time.sleep(1.0)  # Wait before reconnecting
        
        try:
            self.capture = cv2.VideoCapture(self.source)
            if self.capture.isOpened():
                logger.info("Stream reconnected successfully")
            else:
                logger.error("Failed to reconnect stream")
        except Exception as e:
            logger.error(f"Reconnection error: {e}")
    
    def _clear_queue(self, q: queue.Queue):
        """Clear all items from queue."""
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
    
    def get_stats(self) -> dict:
        """Get capture and processing statistics."""
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'state': self.state.value,
            'frames_captured': self.frames_captured,
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'capture_fps': self.frames_captured / elapsed_time if elapsed_time > 0 else 0,
            'processing_fps': self.frames_processed / elapsed_time if elapsed_time > 0 else 0,
            'queue_size': self.frame_queue.qsize(),
            'elapsed_time': elapsed_time
        }
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class RTSPStreamer:
    """RTSP streaming server using GStreamer or FFmpeg."""
    
    def __init__(self, output_url: str, width: int, height: int, fps: int = 30):
        """
        Initialize RTSP streamer.
        
        Args:
            output_url: RTSP output URL
            width: Frame width
            height: Frame height
            fps: Frames per second
        """
        self.output_url = output_url
        self.width = width
        self.height = height
        self.fps = fps
        self.process = None
        self.is_streaming = False
    
    def start_streaming(self) -> bool:
        """
        Start RTSP streaming.
        
        Returns:
            True if started successfully
        """
        if self.is_streaming:
            return True
        
        try:
            # FFmpeg command for RTSP streaming
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.width}x{self.height}',
                '-r', str(self.fps),
                '-i', '-',  # Input from stdin
                '-c:v', 'libx264',
                '-preset', 'ultrafast',
                '-tune', 'zerolatency',
                '-g', str(self.fps),  # Keyframe interval
                '-f', 'rtsp',
                self.output_url
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            
            self.is_streaming = True
            logger.info(f"RTSP streaming started: {self.output_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start RTSP streaming: {e}")
            return False
    
    def stream_frame(self, frame: np.ndarray) -> bool:
        """
        Stream a single frame.
        
        Args:
            frame: BGR frame to stream
            
        Returns:
            True if successful
        """
        if not self.is_streaming or not self.process:
            return False
        
        try:
            # Resize frame if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            # Write frame to FFmpeg stdin
            self.process.stdin.write(frame.tobytes())
            self.process.stdin.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to stream frame: {e}")
            return False
    
    def stop_streaming(self):
        """Stop RTSP streaming."""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()
            self.process = None
        
        self.is_streaming = False
        logger.info("RTSP streaming stopped")

class VideoWriter:
    """Enhanced video writer with multiple codec support."""
    
    def __init__(self, output_path: str, width: int, height: int, fps: float, 
                 codec: str = 'mp4v', quality: int = 95):
        """
        Initialize video writer.
        
        Args:
            output_path: Output file path
            width: Frame width
            height: Frame height
            fps: Frames per second
            codec: Video codec (mp4v, h264, h265)
            quality: Video quality (0-100)
        """
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.codec = codec
        self.quality = quality
        
        self.writer = None
        self.frames_written = 0
    
    def open(self) -> bool:
        """
        Open video writer.
        
        Returns:
            True if successful
        """
        try:
            # Get codec fourcc
            if self.codec.lower() == 'h264':
                fourcc = cv2.VideoWriter_fourcc(*'H264')
            elif self.codec.lower() == 'h265':
                fourcc = cv2.VideoWriter_fourcc(*'HEVC')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            self.writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            
            if not self.writer.isOpened():
                logger.error(f"Failed to open video writer: {self.output_path}")
                return False
            
            logger.info(f"Video writer opened: {self.output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to open video writer: {e}")
            return False
    
    def write_frame(self, frame: np.ndarray) -> bool:
        """
        Write frame to video.
        
        Args:
            frame: BGR frame to write
            
        Returns:
            True if successful
        """
        if not self.writer:
            return False
        
        try:
            # Resize frame if needed
            if frame.shape[:2] != (self.height, self.width):
                frame = cv2.resize(frame, (self.width, self.height))
            
            self.writer.write(frame)
            self.frames_written += 1
            return True
            
        except Exception as e:
            logger.error(f"Failed to write frame: {e}")
            return False
    
    def close(self):
        """Close video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
        
        logger.info(f"Video writer closed. Frames written: {self.frames_written}")
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
```

### `src/classical/__init__.py`
```python
"""Classical image enhancement algorithms for underwater imagery."""

from .white_balance import WhiteBalancer
from .gamma_correction import GammaCorrector
from .guided_filter import GuidedFilter
from .dehazing import PhysicsDehazer
from .color_correction import ColorCorrector

__all__ = [
    "WhiteBalancer",
    "GammaCorrector", 
    "GuidedFilter",
    "PhysicsDehazer",
    "ColorCorrector"
]
```

### `src/classical/gamma_correction.py`
```python
#!/usr/bin/env python3
"""
Gamma correction for underwater images.
Addresses low-light conditions common in underwater environments.
"""

import cv2
import numpy as np
from typing import Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class GammaCorrector:
    """
    Gamma correction for underwater image enhancement.
    
    Implements adaptive and fixed gamma correction with various strategies:
    - Fixed gamma correction
    - Adaptive gamma based on image statistics
    - Per-channel gamma correction
    - Histogram-based gamma selection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize gamma corrector.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.default_gamma = self.config.get('default_gamma', 1.2)
        self.adaptive_mode = self.config.get('adaptive_mode', 'brightness')
        self.per_channel = self.config.get('per_channel', False)
        
        # Build lookup tables for common gamma values for efficiency
        self._lut_cache = {}
        self._build_common_luts()
    
    def adjust(self, image: np.ndarray, gamma: Optional[float] = None, 
               method: str = 'fixed') -> np.ndarray:
        """
        Apply gamma correction to image.
        
        Args:
            image: Input BGR image
            gamma: Gamma value (if None, uses adaptive or default)
            method: Correction method ('fixed', 'adaptive', 'histogram')
            
        Returns:
            Gamma-corrected image
        """
        try:
            if method == 'fixed':
                gamma_value = gamma or self.default_gamma
                return self._apply_gamma(image, gamma_value)
                
            elif method == 'adaptive':
                return self._adaptive_gamma_correction(image)
                
            elif method == 'histogram':
                return self._histogram_based_gamma(image)
                
            elif method == 'per_channel':
                return self._per_channel_gamma(image, gamma)
                
            else:
                logger.warning(f"Unknown gamma correction method: {method}")
                return self._apply_gamma(image, self.default_gamma)
                
        except Exception as e:
            logger.error(f"Gamma correction failed: {e}")
            return image.copy()
    
    def _apply_gamma(self, image: np.ndarray, gamma: float) -> np.ndarray:
        """
        Apply fixed gamma correction using lookup table.
        
        Args:
            image: Input image
            gamma: Gamma value
            
        Returns:
            Gamma-corrected image
        """
        # Use cached LUT if available
        if gamma in self._lut_cache:
            lut = self._lut_cache[gamma]
        else:
            lut = self._build_gamma_lut(gamma)
            # Cache if common value
            if len(self._lut_cache) < 10:
                self._lut_cache[gamma] = lut
        
        # Apply LUT
        corrected = cv2.LUT(image, lut)
        return corrected
    
    def _build_gamma_lut(self, gamma: float) -> np.ndarray:
        """
        Build gamma correction lookup table.
        
        Args:
            gamma: Gamma value
            
        Returns:
            256-element LUT array
        """
        inv_gamma = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv_gamma) * 255 
                       for i in range(256)]).astype(np.uint8)
        return lut
    
    def _build_common_luts(self):
        """Pre-build LUTs for common gamma values."""
        common_gammas = [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2]
        for gamma in common_gammas:
            self._lut_cache[gamma] = self._build_gamma_lut(gamma)
    
    def _adaptive_gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive gamma correction based on image statistics.
        
        Args:
            image: Input image
            
        Returns:
            Gamma-corrected image
        """
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        if self.adaptive_mode == 'brightness':
            # Gamma based on mean brightness
            mean_brightness = np.mean(gray) / 255.0
            
            if mean_brightness < 0.3:
                # Dark image - reduce gamma to brighten
                gamma = 0.7
            elif mean_brightness > 0.7:
                # Bright image - increase gamma to darken
                gamma = 1.4
            else:
                # Normal brightness
                gamma = 1.0
                
        elif self.adaptive_mode == 'histogram':
            # Gamma based on histogram distribution
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_norm = hist / hist.sum()
            
            # Calculate histogram moments
            cumsum = np.cumsum(hist_norm)
            
            # Find percentiles
            p25_idx = np.where(cumsum >= 0.25)[0][0]
            p75_idx = np.where(cumsum >= 0.75)[0][0]
            
            # Adaptive gamma based on histogram spread
            spread = (p75_idx - p25_idx) / 255.0
            
            if spread < 0.3:
                # Low contrast - stronger gamma correction
                gamma = 0.8 if np.mean(gray) < 128 else 1.3
            else:
                # Good contrast - moderate correction
                gamma = 1.1 if np.mean(gray) < 128 else 1.2
                
        elif self.adaptive_mode == 'entropy':
            # Gamma based on image entropy
            entropy = self._calculate_entropy(gray)
            
            if entropy < 6.0:
                # Low entropy - stronger correction
                gamma = 0.8 if np.mean(gray) < 128 else 1.4
            else:
                # High entropy - mild correction
                gamma = 1.1
                
        else:
            gamma = self.default_gamma
        
        return self._apply_gamma(image, gamma)
    
    def _histogram_based_gamma(self, image: np.ndarray) -> np.ndarray:
        """
        Apply histogram-based gamma correction.
        
        Args:
            image: Input image
            
        Returns:
            Gamma-corrected image
        """
        # Convert to LAB for better perceptual uniformity
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel = lab[:, :, 0]
        
        # Calculate target histogram (more uniform distribution)
        hist, bins = np.histogram(l_channel.flatten(), 256, [0, 256])
        
        # Calculate cumulative distribution
        cdf = hist.cumsum()
        cdf_normalized = cdf / cdf.max()
        
        # Find optimal gamma that makes distribution more uniform
        target_cdf = np.linspace(0, 1, 256)
        
        # Minimize difference between actual and target CDF
        best_gamma = 1.0
        min_error = float('inf')
        
        for gamma_test in np.arange(0.5, 2.5, 0.1):
            # Create gamma-corrected CDF
            gamma_lut = self._build_gamma_lut(gamma_test)
            gamma_corrected = cv2.LUT(l_channel, gamma_lut)
            
            hist_test, _ = np.histogram(gamma_corrected.flatten(), 256, [0, 256])
            cdf_test = hist_test.cumsum()
            cdf_test_norm = cdf_test / cdf_test.max()
            
            # Calculate error
            error = np.mean((cdf_test_norm - target_cdf) ** 2)
            
            if error < min_error:
                min_error = error
                best_gamma = gamma_test
        
        return self._apply_gamma(image, best_gamma)
    
    def _per_channel_gamma(self, image: np.ndarray, 
                          gamma: Optional[Union[float, tuple]] = None) -> np.ndarray:
        """
        Apply per-channel gamma correction.
        
        Args:
            image: Input BGR image
            gamma: Single gamma or tuple of (gamma_b, gamma_g, gamma_r)
            
        Returns:
            Gamma-corrected image
        """
        if gamma is None:
            # Calculate per-channel gamma based on channel statistics
            gammas = []
            for channel in range(3):
                ch_mean = np.mean(image[:, :, channel]) / 255.0
                
                if ch_mean < 0.4:
                    ch_gamma = 0.8  # Brighten dark channels
                elif ch_mean > 0.6:
                    ch_gamma = 1.3  # Darken bright channels  
                else:
                    ch_gamma = 1.0
                    
                gammas.append(ch_gamma)
        else:
            if isinstance(gamma, (int, float)):
                gammas = [gamma] * 3
            else:
                gammas = list(gamma)
        
        # Apply gamma to each channel
        corrected = image.copy()
        for channel in range(3):
            lut = self._build_gamma_lut(gammas[channel])
            corrected[:, :, channel] = cv2.LUT(image[:, :, channel], lut)
        
        return corrected
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """
        Calculate image entropy.
        
        Args:
            image: Grayscale image
            
        Returns:
            Entropy value
        """
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        
        # Remove zero entries
        hist_norm = hist_norm[hist_norm > 0]
        
        # Calculate entropy
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        return entropy
    
    def estimate_optimal_gamma(self, image: np.ndarray) -> float:
        """
        Estimate optimal gamma value for given image.
        
        Args:
            image: Input image
            
        Returns:
            Estimated optimal gamma
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate image statistics
        mean_val = np.mean(gray) / 255.0
        std_val = np.std(gray) / 255.0
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.flatten() / hist.sum()
        
        # Find mode (peak) of histogram
        mode_idx = np.argmax(hist_norm)
        mode_val = mode_idx / 255.0
        
        # Estimate gamma based on statistics
        if mean_val < 0.3:
            # Dark image
            if std_val < 0.2:
                gamma = 0.6  # Very dark, low contrast
            else:
                gamma = 0.8  # Dark but with some contrast
        elif mean_val > 0.7:
            # Bright image
            if std_val < 0.2:
                gamma = 1.6  # Very bright, low contrast
            else:
                gamma = 1.3  # Bright with contrast
        else:
            # Normal brightness
            if std_val < 0.15:
                gamma = 1.4  # Low contrast
            elif mode_val < mean_val:
                gamma = 1.1  # Slightly dark bias
            else:
                gamma = 1.0  # Well balanced
        
        return gamma
    
    def validate_correction(self, original: np.ndarray, 
                          corrected: np.ndarray) -> Dict[str, float]:
        """
        Validate gamma correction quality.
        
        Args:
            original: Original image
            corrected: Gamma-corrected image
            
        Returns:
            Validation metrics
        """
        # Convert to grayscale for analysis
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        corr_gray = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        # Brightness metrics
        orig_mean = np.mean(orig_gray)
        corr_mean = np.mean(corr_gray)
        brightness_change = (corr_mean - orig_mean) / orig_mean
        
        # Contrast metrics
        orig_std = np.std(orig_gray)
        corr_std = np.std(corr_gray)
        contrast_change = (corr_std - orig_std) / orig_std if orig_std > 0 else 0
        
        # Entropy metrics
        orig_entropy = self._calculate_entropy(orig_gray)
        corr_entropy = self._calculate_entropy(corr_gray)
        entropy_change = corr_entropy - orig_entropy
        
        # Dynamic range
        orig_range = np.max(orig_gray) - np.min(orig_gray)
        corr_range = np.max(corr_gray) - np.min(corr_gray)
        range_change = (corr_range - orig_range) / orig_range if orig_range > 0 else 0
        
        return {
            'brightness_change': float(brightness_change),
            'contrast_change': float(contrast_change),
            'entropy_change': float(entropy_change),
            'range_change': float(range_change),
            'original_mean': float(orig_mean),
            'corrected_mean': float(corr_mean),
            'original_std': float(orig_std),
            'corrected_std': float(corr_std)
        }
```