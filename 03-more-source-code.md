### `src/classical/guided_filter.py`
```python
#!/usr/bin/env python3
"""
Guided filter implementation for underwater image enhancement.
Provides edge-preserving smoothing while maintaining important details.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class GuidedFilter:
    """
    Guided filter for edge-preserving image smoothing.
    
    Particularly effective for underwater images where fine details
    need to be preserved while reducing noise and artifacts.
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize guided filter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.radius = self.config.get('radius', 8)
        self.eps = self.config.get('eps', 0.01)
        self.use_color_guide = self.config.get('use_color_guide', True)
    
    def filter(self, input_image: np.ndarray, 
               guide_image: Optional[np.ndarray] = None,
               radius: Optional[int] = None,
               eps: Optional[float] = None) -> np.ndarray:
        """
        Apply guided filter to input image.
        
        Args:
            input_image: Image to be filtered
            guide_image: Guide image (if None, uses input as guide)
            radius: Filter radius (if None, uses default)
            eps: Regularization parameter (if None, uses default)
            
        Returns:
            Filtered image
        """
        try:
            # Use provided parameters or defaults
            r = radius or self.radius
            epsilon = eps or self.eps
            
            # Use input as guide if no guide provided
            if guide_image is None:
                guide_image = input_image.copy()
            
            # Ensure images are float32
            I = input_image.astype(np.float32) / 255.0
            p = input_image.astype(np.float32) / 255.0
            
            if len(guide_image.shape) == 3 and self.use_color_guide:
                # Color guided filter
                return self._guided_filter_color(I, p, guide_image, r, epsilon)
            else:
                # Grayscale guided filter
                if len(guide_image.shape) == 3:
                    guide_gray = cv2.cvtColor(guide_image, cv2.COLOR_BGR2GRAY)
                else:
                    guide_gray = guide_image
                
                return self._guided_filter_gray(I, p, guide_gray, r, epsilon)
                
        except Exception as e:
            logger.error(f"Guided filter failed: {e}")
            return input_image.copy()
    
    def _guided_filter_gray(self, I: np.ndarray, p: np.ndarray, 
                           guide: np.ndarray, r: int, eps: float) -> np.ndarray:
        """Apply grayscale guided filter."""
        # Ensure guide is float32
        guide = guide.astype(np.float32) / 255.0
        
        # Handle multi-channel input
        if len(p.shape) == 3:
            # Filter each channel separately
            filtered_channels = []
            for c in range(p.shape[2]):
                filtered_ch = self._guided_filter_single_channel(
                    I[:,:,c], p[:,:,c], guide, r, eps
                )
                filtered_channels.append(filtered_ch)
            
            result = np.stack(filtered_channels, axis=2)
        else:
            result = self._guided_filter_single_channel(I, p, guide, r, eps)
        
        # Convert back to uint8
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    def _guided_filter_single_channel(self, I: np.ndarray, p: np.ndarray,
                                    guide: np.ndarray, r: int, eps: float) -> np.ndarray:
        """Apply guided filter to single channel."""
        # Mean filters
        mean_I = cv2.blur(guide, (2*r+1, 2*r+1))
        mean_p = cv2.blur(p, (2*r+1, 2*r+1))
        
        # Correlation and covariance
        corr_Ip = cv2.blur(guide * p, (2*r+1, 2*r+1))
        cov_Ip = corr_Ip - mean_I * mean_p
        
        # Variance of guide
        mean_II = cv2.blur(guide * guide, (2*r+1, 2*r+1))
        var_I = mean_II - mean_I * mean_I
        
        # Linear coefficients
        a = cov_Ip / (var_I + eps)
        b = mean_p - a * mean_I
        
        # Smooth coefficients
        mean_a = cv2.blur(a, (2*r+1, 2*r+1))
        mean_b = cv2.blur(b, (2*r+1, 2*r+1))
        
        # Output
        q = mean_a * guide + mean_b
        return q
    
    def _guided_filter_color(self, I: np.ndarray, p: np.ndarray,
                            guide: np.ndarray, r: int, eps: float) -> np.ndarray:
        """Apply color guided filter."""
        guide = guide.astype(np.float32) / 255.0
        height, width = guide.shape[:2]
        
        # Handle multi-channel input
        if len(p.shape) == 3:
            output_channels = []
            for c in range(p.shape[2]):
                filtered_ch = self._color_guided_filter_channel(
                    I[:,:,c], p[:,:,c], guide, r, eps
                )
                output_channels.append(filtered_ch)
            result = np.stack(output_channels, axis=2)
        else:
            result = self._color_guided_filter_channel(I, p, guide, r, eps)
        
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)
    
    def _color_guided_filter_channel(self, I_ch: np.ndarray, p_ch: np.ndarray,
                                   guide: np.ndarray, r: int, eps: float) -> np.ndarray:
        """Apply color guided filter to single channel."""
        height, width = guide.shape[:2]
        
        # Box filter size
        box_size = 2 * r + 1
        
        # Number of pixels in each local patch
        N = cv2.blur(np.ones((height, width)), (box_size, box_size))
        
        # Mean of guide and input
        mean_I_r = cv2.blur(guide[:,:,0], (box_size, box_size))
        mean_I_g = cv2.blur(guide[:,:,1], (box_size, box_size)) 
        mean_I_b = cv2.blur(guide[:,:,2], (box_size, box_size))
        mean_p = cv2.blur(p_ch, (box_size, box_size))
        
        # Covariance of (I, p) in each local patch
        cov_Ip_r = cv2.blur(guide[:,:,0] * p_ch, (box_size, box_size)) - mean_I_r * mean_p
        cov_Ip_g = cv2.blur(guide[:,:,1] * p_ch, (box_size, box_size)) - mean_I_g * mean_p
        cov_Ip_b = cv2.blur(guide[:,:,2] * p_ch, (box_size, box_size)) - mean_I_b * mean_p
        
        # Variance of I in each local patch: the matrix Sigma
        var_I_rr = cv2.blur(guide[:,:,0] * guide[:,:,0], (box_size, box_size)) - mean_I_r * mean_I_r + eps
        var_I_rg = cv2.blur(guide[:,:,0] * guide[:,:,1], (box_size, box_size)) - mean_I_r * mean_I_g
        var_I_rb = cv2.blur(guide[:,:,0] * guide[:,:,2], (box_size, box_size)) - mean_I_r * mean_I_b
        var_I_gg = cv2.blur(guide[:,:,1] * guide[:,:,1], (box_size, box_size)) - mean_I_g * mean_I_g + eps
        var_I_gb = cv2.blur(guide[:,:,1] * guide[:,:,2], (box_size, box_size)) - mean_I_g * mean_I_b
        var_I_bb = cv2.blur(guide[:,:,2] * guide[:,:,2], (box_size, box_size)) - mean_I_b * mean_I_b + eps
        
        # Inverse of Sigma + eps*I
        a = np.zeros((height, width, 3))
        
        for y in range(height):
            for x in range(width):
                # 3x3 covariance matrix
                Sigma = np.array([
                    [var_I_rr[y,x], var_I_rg[y,x], var_I_rb[y,x]],
                    [var_I_rg[y,x], var_I_gg[y,x], var_I_gb[y,x]],
                    [var_I_rb[y,x], var_I_gb[y,x], var_I_bb[y,x]]
                ])
                
                cov_Ip = np.array([cov_Ip_r[y,x], cov_Ip_g[y,x], cov_Ip_b[y,x]])
                
                try:
                    a[y,x,:] = np.linalg.solve(Sigma, cov_Ip)
                except:
                    a[y,x,:] = 0
        
        # Calculate b
        b = mean_p - a[:,:,0] * mean_I_r - a[:,:,1] * mean_I_g - a[:,:,2] * mean_I_b
        
        # Smooth a and b
        mean_a_r = cv2.blur(a[:,:,0], (box_size, box_size))
        mean_a_g = cv2.blur(a[:,:,1], (box_size, box_size))
        mean_a_b = cv2.blur(a[:,:,2], (box_size, box_size))
        mean_b = cv2.blur(b, (box_size, box_size))
        
        # Output
        q = (mean_a_r * guide[:,:,0] + 
             mean_a_g * guide[:,:,1] + 
             mean_a_b * guide[:,:,2] + 
             mean_b)
        
        return q

class FastGuidedFilter:
    """Fast approximation of guided filter for real-time applications."""
    
    def __init__(self, config: dict = None):
        """Initialize fast guided filter."""
        self.config = config or {}
        self.radius = self.config.get('radius', 4)
        self.eps = self.config.get('eps', 0.01)
        self.subsample_ratio = self.config.get('subsample_ratio', 4)
    
    def filter(self, input_image: np.ndarray,
               guide_image: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply fast guided filter.
        
        Args:
            input_image: Input image to filter
            guide_image: Guide image (optional)
            
        Returns:
            Filtered image
        """
        try:
            if guide_image is None:
                guide_image = input_image.copy()
            
            # Downsample for efficiency
            h, w = input_image.shape[:2]
            sub_h = h // self.subsample_ratio
            sub_w = w // self.subsample_ratio
            
            # Downsample images
            I_sub = cv2.resize(guide_image, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            p_sub = cv2.resize(input_image, (sub_w, sub_h), interpolation=cv2.INTER_NEAREST)
            
            # Apply guided filter on subsampled images
            guided_filter = GuidedFilter({'radius': self.radius, 'eps': self.eps})
            q_sub = guided_filter.filter(p_sub, I_sub)
            
            # Upsample result
            result = cv2.resize(q_sub, (w, h), interpolation=cv2.INTER_LINEAR)
            
            return result
            
        except Exception as e:
            logger.error(f"Fast guided filter failed: {e}")
            return input_image.copy()
```

### `src/classical/dehazing.py`
```python
#!/usr/bin/env python3
"""
Physics-informed dehazing algorithms for underwater image enhancement.
Implements dark channel prior and underwater light attenuation models.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class PhysicsDehazer:
    """
    Physics-informed underwater dehazing using optical models.
    
    Implements multiple dehazing approaches:
    - Dark channel prior adapted for underwater
    - Color attenuation prior
    - Depth-based transmission estimation
    - Backscatter removal
    """
    
    def __init__(self, config: dict = None):
        """
        Initialize physics-based dehazer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.beta = self.config.get('beta', 1.0)  # Attenuation coefficient
        self.tx = self.config.get('tx', 0.1)      # Minimum transmission
        self.use_dark_channel = self.config.get('use_dark_channel', True)
        self.omega = self.config.get('omega', 0.95)  # Dark channel factor
        self.patch_size = self.config.get('patch_size', 15)
        
        # Underwater-specific parameters
        self.water_type = self.config.get('water_type', 'oceanic')  # oceanic, coastal, turbid
        self._setup_water_parameters()
    
    def remove_backscatter(self, image: np.ndarray) -> np.ndarray:
        """
        Remove backscatter from underwater image.
        
        Args:
            image: Input underwater image
            
        Returns:
            Dehazed image
        """
        try:
            if self.use_dark_channel:
                return self._dark_channel_dehazing(image)
            else:
                return self._color_attenuation_dehazing(image)
                
        except Exception as e:
            logger.error(f"Backscatter removal failed: {e}")
            return image.copy()
    
    def _setup_water_parameters(self):
        """Setup water-specific optical parameters."""
        if self.water_type == 'oceanic':
            # Clear oceanic water
            self.attenuation_coeffs = {
                'red': 0.8,    # High red attenuation
                'green': 0.4,  # Moderate green attenuation
                'blue': 0.1    # Low blue attenuation
            }
            self.backscatter_coeff = 0.05
            
        elif self.water_type == 'coastal':
            # Coastal water with some turbidity
            self.attenuation_coeffs = {
                'red': 1.2,
                'green': 0.6,
                'blue': 0.2
            }
            self.backscatter_coeff = 0.1
            
        elif self.water_type == 'turbid':
            # Turbid water with high scattering
            self.attenuation_coeffs = {
                'red': 2.0,
                'green': 1.0,
                'blue': 0.5
            }
            self.backscatter_coeff = 0.2
        
        else:
            # Default parameters
            self.attenuation_coeffs = {
                'red': 1.0,
                'green': 0.5,
                'blue': 0.2
            }
            self.backscatter_coeff = 0.1
    
    def _dark_channel_dehazing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply dark channel prior dehazing adapted for underwater images.
        
        Args:
            image: Input image
            
        Returns:
            Dehazed image
        """
        # Convert to float
        I = image.astype(np.float32) / 255.0
        
        # Compute dark channel
        dark_channel = self._get_dark_channel(I, self.patch_size)
        
        # Estimate atmospheric light (background light in water)
        A = self._estimate_background_light(I, dark_channel)
        
        # Estimate transmission map
        transmission = self._estimate_transmission(I, A, dark_channel)
        
        # Refine transmission map using guided filter
        from .guided_filter import GuidedFilter
        guide_filter = GuidedFilter({'radius': 60, 'eps': 0.0001})
        gray_I = cv2.cvtColor((I * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
        transmission = guide_filter.filter(
            (transmission * 255).astype(np.uint8),
            gray_I
        ).astype(np.float32) / 255.0
        
        # Recover scene radiance
        J = self._recover_scene_radiance(I, A, transmission)
        
        # Convert back to uint8
        result = (np.clip(J, 0, 1) * 255).astype(np.uint8)
        return result
    
    def _get_dark_channel(self, image: np.ndarray, patch_size: int) -> np.ndarray:
        """
        Compute dark channel of image.
        
        Args:
            image: Input image (float, 0-1)
            patch_size: Size of local patch
            
        Returns:
            Dark channel map
        """
        # Take minimum across color channels
        min_channel = np.min(image, axis=2)
        
        # Apply minimum filter to get local minimum in patch
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
        dark_channel = cv2.erode(min_channel, kernel)
        
        return dark_channel
    
    def _estimate_background_light(self, image: np.ndarray, 
                                  dark_channel: np.ndarray) -> np.ndarray:
        """
        Estimate background light (atmospheric light in water).
        
        Args:
            image: Input image
            dark_channel: Dark channel map
            
        Returns:
            Background light for each channel
        """
        height, width = dark_channel.shape
        num_pixels = height * width
        num_brightest = int(max(num_pixels * 0.001, 1))  # Top 0.1% brightest pixels
        
        # Find brightest pixels in dark channel
        dark_vec = dark_channel.reshape(num_pixels)
        image_vec = image.reshape(num_pixels, 3)
        
        indices = dark_vec.argsort()[-num_brightest:]
        
        # Background light is max intensity in brightest dark channel pixels
        brightest_pixels = image_vec[indices]
        A = np.max(brightest_pixels, axis=0)
        
        # Ensure minimum background light
        A = np.maximum(A, 0.05)
        
        return A
    
    def _estimate_transmission(self, image: np.ndarray, A: np.ndarray,
                              dark_channel: np.ndarray) -> np.ndarray:
        """
        Estimate transmission map.
        
        Args:
            image: Input image
            A: Background light
            dark_channel: Dark channel map
            
        Returns:
            Transmission map
        """
        # Normalize image by background light
        norm_image = image / A
        
        # Compute dark channel of normalized image
        norm_dark = self._get_dark_channel(norm_image, self.patch_size)
        
        # Transmission estimation
        transmission = 1 - self.omega * norm_dark
        
        # Enforce minimum transmission
        transmission = np.maximum(transmission, self.tx)
        
        return transmission
    
    def _recover_scene_radiance(self, image: np.ndarray, A: np.ndarray,
                               transmission: np.ndarray) -> np.ndarray:
        """
        Recover scene radiance using transmission map.
        
        Args:
            image: Hazy input image
            A: Background light
            transmission: Transmission map
            
        Returns:
            Recovered scene radiance
        """
        # Expand transmission to 3 channels
        t = transmission[:, :, np.newaxis].repeat(3, axis=2)
        
        # Scene radiance recovery
        J = (image - A) / t + A
        
        return J
    
    def _color_attenuation_dehazing(self, image: np.ndarray) -> np.ndarray:
        """
        Apply color attenuation prior dehazing for underwater images.
        
        Args:
            image: Input image
            
        Returns:
            Dehazed image
        """
        # Convert to float
        I = image.astype(np.float32) / 255.0
        
        # Estimate depth map using color attenuation
        depth_map = self._estimate_depth_map(I)
        
        # Estimate transmission for each channel
        transmission_r = np.exp(-self.attenuation_coeffs['red'] * depth_map)
        transmission_g = np.exp(-self.attenuation_coeffs['green'] * depth_map)
        transmission_b = np.exp(-self.attenuation_coeffs['blue'] * depth_map)
        
        # Stack transmissions
        transmission = np.stack([transmission_b, transmission_g, transmission_r], axis=2)
        
        # Enforce minimum transmission
        transmission = np.maximum(transmission, self.tx)
        
        # Estimate background light
        A = self._estimate_background_light_color(I)
        
        # Recover scene radiance
        J = (I - A) / transmission + A
        
        # Convert back to uint8
        result = (np.clip(J, 0, 1) * 255).astype(np.uint8)
        return result
    
    def _estimate_depth_map(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate relative depth map from color attenuation.
        
        Args:
            image: Input image (float, 0-1)
            
        Returns:
            Relative depth map
        """
        # Extract color channels
        B, G, R = image[:,:,0], image[:,:,1], image[:,:,2]
        
        # Red channel attenuates most in water
        # Use ratio of blue to red as depth indicator
        red_blue_ratio = np.divide(R, B + 1e-6)
        
        # Normalize and invert (higher ratio = less depth)
        depth_map = 1.0 - (red_blue_ratio - np.min(red_blue_ratio)) / (np.max(red_blue_ratio) - np.min(red_blue_ratio) + 1e-6)
        
        # Apply smoothing
        depth_map = cv2.GaussianBlur(depth_map, (5, 5), 0)
        
        # Scale depth map
        depth_map = depth_map * self.beta
        
        return depth_map
    
    def _estimate_background_light_color(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate background light for color attenuation method.
        
        Args:
            image: Input image
            
        Returns:
            Background light per channel
        """
        # For underwater images, background light varies per channel
        # Blue light penetrates deepest, red is most attenuated
        
        if self.water_type == 'oceanic':
            A = np.array([0.8, 0.6, 0.4])  # Blue dominant
        elif self.water_type == 'coastal':
            A = np.array([0.7, 0.7, 0.5])  # More balanced
        elif self.water_type == 'turbid':
            A = np.array([0.6, 0.6, 0.6])  # Uniform scattering
        else:
            # Adaptive estimation
            A = np.array([
                np.percentile(image[:,:,0], 95),  # Blue
                np.percentile(image[:,:,1], 90),  # Green  
                np.percentile(image[:,:,2], 85)   # Red
            ])
        
        return A
    
    def estimate_water_type(self, image: np.ndarray) -> str:
        """
        Automatically estimate water type from image characteristics.
        
        Args:
            image: Input underwater image
            
        Returns:
            Estimated water type ('oceanic', 'coastal', 'turbid')
        """
        # Convert to float
        I = image.astype(np.float32) / 255.0
        
        # Analyze color distribution
        mean_blue = np.mean(I[:,:,0])
        mean_green = np.mean(I[:,:,1])
        mean_red = np.mean(I[:,:,2])
        
        # Calculate color ratios
        blue_red_ratio = mean_blue / (mean_red + 1e-6)
        green_red_ratio = mean_green / (mean_red + 1e-6)
        
        # Analyze contrast and clarity
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contrast = np.std(gray)
        
        # Classification logic
        if blue_red_ratio > 3.0 and contrast > 50:
            return 'oceanic'
        elif blue_red_ratio > 2.0 and green_red_ratio > 1.5:
            return 'coastal'  
        else:
            return 'turbid'
    
    def validate_dehazing(self, original: np.ndarray, 
                         dehazed: np.ndarray) -> dict:
        """
        Validate dehazing quality using various metrics.
        
        Args:
            original: Original hazy image
            dehazed: Dehazed image
            
        Returns:
            Validation metrics
        """
        # Convert to float
        orig = original.astype(np.float32) / 255.0
        deha = dehazed.astype(np.float32) / 255.0
        
        # Contrast improvement
        orig_std = np.std(orig)
        deha_std = np.std(deha)
        contrast_improvement = (deha_std - orig_std) / orig_std if orig_std > 0 else 0
        
        # Color saturation improvement
        orig_hsv = cv2.cvtColor(orig, cv2.COLOR_BGR2HSV)
        deha_hsv = cv2.cvtColor(deha, cv2.COLOR_BGR2HSV)
        
        orig_saturation = np.mean(orig_hsv[:,:,1])
        deha_saturation = np.mean(deha_hsv[:,:,1])
        saturation_improvement = (deha_saturation - orig_saturation) / orig_saturation if orig_saturation > 0 else 0
        
        # Edge sharpness (using Laplacian variance)
        orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
        deha_gray = cv2.cvtColor(deha, cv2.COLOR_BGR2GRAY)
        
        orig_sharpness = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
        deha_sharpness = cv2.Laplacian(deha_gray, cv2.CV_64F).var()
        sharpness_improvement = (deha_sharpness - orig_sharpness) / orig_sharpness if orig_sharpness > 0 else 0
        
        return {
            'contrast_improvement': float(contrast_improvement),
            'saturation_improvement': float(saturation_improvement),
            'sharpness_improvement': float(sharpness_improvement),
            'original_contrast': float(orig_std),
            'dehazed_contrast': float(deha_std),
            'original_saturation': float(orig_saturation),
            'dehazed_saturation': float(deha_saturation)
        }
```

### `src/api/schemas.py`
```python
#!/usr/bin/env python3
"""
Pydantic schemas for REST API request/response models.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from datetime import datetime

class EnhancementModeEnum(str, Enum):
    """Enhancement mode enumeration."""
    LIGHTWEIGHT = "lightweight"
    HIFI = "hifi"

class OutputFormatEnum(str, Enum):
    """Output format enumeration."""
    JPEG = "jpeg"
    PNG = "png"

class ProcessingStats(BaseModel):
    """Processing statistics model."""
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")
    fps: float = Field(..., description="Frames per second")
    input_resolution: tuple = Field(..., description="Input image resolution (width, height)")
    uiqm_score: Optional[float] = Field(None, description="UIQM quality score")
    uciqe_score: Optional[float] = Field(None, description="UCIQE quality score")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")

class QualityMetrics(BaseModel):
    """Image quality metrics model."""
    uiqm: Optional[float] = Field(None, description="Underwater Image Quality Measure")
    uciqe: Optional[float] = Field(None, description="Underwater Color Image Quality Evaluation")
    psnr: Optional[float] = Field(None, description="Peak Signal-to-Noise Ratio")
    ssim: Optional[float] = Field(None, description="Structural Similarity Index Measure")
    colorfulness: Optional[float] = Field(None, description="Colorfulness metric")
    contrast: Optional[float] = Field(None, description="Contrast metric")
    sharpness: Optional[float] = Field(None, description="Sharpness metric")

class EnhanceResponse(BaseModel):
    """Single image enhancement response."""
    request_id: str = Field(..., description="Unique request identifier")
    enhanced_image: str = Field(..., description="Base64 encoded enhanced image")
    processing_stats: ProcessingStats = Field(..., description="Processing statistics")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality metrics")

class BatchEnhanceResponse(BaseModel):
    """Batch enhancement response."""
    request_id: str = Field(..., description="Batch request identifier")
    results: List[EnhanceResponse] = Field(..., description="Individual enhancement results")
    total_processing_time_ms: float = Field(..., description="Total processing time")
    success_count: int = Field(..., description="Number of successful enhancements")
    total_count: int = Field(..., description="Total number of images")

class StreamRequest(BaseModel):
    """Video stream processing request."""
    source: str = Field(..., description="Stream source (RTSP URL, file path, camera index)")
    mode: EnhancementModeEnum = Field(EnhancementModeEnum.LIGHTWEIGHT, description="Enhancement mode")
    output_format: OutputFormatEnum = Field(OutputFormatEnum.JPEG, description="Output format")
    rtsp_output: Optional[str] = Field(None, description="RTSP output URL for restreaming")
    enable_metrics: bool = Field(False, description="Enable quality metrics computation")
    
    @validator('source')
    def validate_source(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Source cannot be empty')
        return v.strip()

class SourceInfo(BaseModel):
    """Video source information."""
    width: int = Field(..., description="Frame width")
    height: int = Field(..., description="Frame height")
    fps: float = Field(..., description="Frames per second")

class StreamResponse(BaseModel):
    """Stream processing response."""
    stream_id: str = Field(..., description="Unique stream identifier")
    status: str = Field(..., description="Stream status")
    source_info: SourceInfo = Field(..., description="Source information")

class StreamStatus(BaseModel):
    """Stream status information."""
    stream_id: str = Field(..., description="Stream identifier")
    status: str = Field(..., description="Current status")
    start_time: datetime = Field(..., description="Stream start time")
    source_info: SourceInfo = Field(..., description="Source information")

class ConfigUpdate(BaseModel):
    """Configuration update request."""
    mode: Optional[EnhancementModeEnum] = Field(None, description="Enhancement mode")
    config: Optional[Dict[str, Any]] = Field(None, description="Configuration parameters")

class ConfigResponse(BaseModel):
    """Configuration response."""
    mode: str = Field(..., description="Current enhancement mode")
    device: str = Field(..., description="Processing device")
    config: Dict[str, Any] = Field(..., description="Current configuration")

class PerformanceMetrics(BaseModel):
    """System performance metrics."""
    avg_fps: float = Field(..., description="Average frames per second")
    max_fps: Optional[float] = Field(None, description="Maximum FPS achieved")
    min_fps: Optional[float] = Field(None, description="Minimum FPS achieved")
    avg_processing_time_ms: float = Field(..., description="Average processing time")
    frame_count: Optional[int] = Field(None, description="Total frames processed")
    memory_usage_mb: float = Field(..., description="Current memory usage in MB")

class SystemMetrics(BaseModel):
    """Complete system metrics."""
    performance: PerformanceMetrics = Field(..., description="Performance metrics")
    active_streams: int = Field(..., description="Number of active streams")
    total_requests: int = Field(..., description="Total requests processed")
    uptime_seconds: float = Field(..., description="System uptime in seconds")

class HealthStatus(BaseModel):
    """Health check status."""
    status: str = Field(..., description="Health status (healthy, unhealthy)")
    timestamp: datetime = Field(..., description="Check timestamp")
    performance: PerformanceMetrics = Field(..., description="Performance metrics")

class ServiceInfo(BaseModel):
    """Service information."""
    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    status: str = Field(..., description="Service status")
    mode: str = Field(..., description="Current enhancement mode")
    device: str = Field(..., description="Processing device")

class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
```

### `src/cli/enhance.py`
```python
#!/usr/bin/env python3
"""
CLI enhancement commands for underwater image processing.
"""

import click
import cv2
import numpy as np
import os
import logging
from pathlib import Path
from typing import Optional

from ..core.enhancement import UnderwaterImageEnhancer, EnhancementMode
from ..core.metrics import ImageQualityMetrics

logger = logging.getLogger(__name__)

@click.command()
@click.option('--input', '-i', required=True, 
              help='Input image/video file or RTSP URL')
@click.option('--output', '-o', 
              help='Output file path')
@click.option('--mode', '-m', default='lightweight',
              type=click.Choice(['lightweight', 'hifi']),
              help='Enhancement mode')
@click.option('--metrics', is_flag=True,
              help='Compute and display quality metrics')
@click.option('--overlay', is_flag=True,
              help='Overlay metrics on output image')
@click.option('--device', default='auto',
              help='Compute device (auto/cpu/cuda)')
@click.option('--preset', 
              type=click.Choice(['port-survey', 'diver-assist', 'deep-water']),
              help='Use mission-specific preset')
@click.option('--batch-size', default=1, type=int,
              help='Batch size for video processing')
@click.option('--quality', default=95, type=int,
              help='Output JPEG quality (1-100)')
@click.pass_context
def enhance_command(ctx, input, output, mode, metrics, overlay, device, 
                   preset, batch_size, quality):
    """
    Enhance underwater images or videos.
    
    Examples:
    \b
    # Single image with metrics
    uie enhance -i underwater.jpg -o enhanced.jpg --metrics
    
    # Video with high-fidelity mode
    uie enhance -i video.mp4 -o enhanced.mp4 --mode hifi
    
    # RTSP stream with preset
    uie enhance -i rtsp://camera:8554/stream --preset port-survey
    """
    try:
        # Load configuration and preset
        config = ctx.obj.get('config', {})
        if preset:
            config.update(load_preset_config(preset))
        
        # Initialize enhancer
        enhancement_mode = EnhancementMode(mode)
        enhancer = UnderwaterImageEnhancer(
            mode=enhancement_mode,
            config=config,
            device=device
        )
        
        # Determine input type and process
        if input.lower().startswith(('rtsp://', 'http://', 'https://')):
            process_stream(enhancer, input, output, metrics, overlay, batch_size)
        elif os.path.isfile(input):
            if is_image_file(input):
                process_single_image(enhancer, input, output, metrics, overlay, quality)
            elif is_video_file(input):
                process_video_file(enhancer, input, output, metrics, batch_size, quality)
            else:
                raise click.ClickException(f"Unsupported file format: {input}")
        else:
            raise click.ClickException(f"Input not found: {input}")
            
    except Exception as e:
        logger.error(f"Enhancement failed: {e}")
        raise click.ClickException(str(e))

def process_single_image(enhancer: UnderwaterImageEnhancer, 
                        input_path: str, output_path: Optional[str],
                        compute_metrics: bool, overlay_metrics: bool,
                        quality: int):
    """Process a single image."""
    logger.info(f"Processing image: {input_path}")
    
    # Load image
    image = cv2.imread(input_path)
    if image is None:
        raise click.ClickException(f"Cannot read image: {input_path}")
    
    # Enhance image
    enhanced, stats = enhancer.enhance_frame(image, compute_metrics=compute_metrics)
    
    # Add metrics overlay if requested
    if overlay_metrics and compute_metrics:
        enhanced = add_metrics_overlay(enhanced, stats)
    
    # Save output
    if output_path:
        save_params = [cv2.IMWRITE_JPEG_QUALITY, quality] if output_path.lower().endswith('.jpg') else []
        success = cv2.imwrite(output_path, enhanced, save_params)
        if success:
            click.echo(f"Enhanced image saved: {output_path}")
        else:
            raise click.ClickException(f"Failed to save image: {output_path}")
    
    # Display metrics
    if compute_metrics:
        display_image_metrics(stats, input_path)
    
    # Display processing stats
    click.echo(f"\nProcessing Statistics:")
    click.echo(f"  Resolution: {stats.input_resolution}")
    click.echo(f"  Processing time: {stats.processing_time_ms:.2f} ms")
    click.echo(f"  Theoretical FPS: {stats.fps:.1f}")

def process_video_file(enhancer: UnderwaterImageEnhancer,
                      input_path: str, output_path: Optional[str],
                      compute_metrics: bool, batch_size: int, quality: int):
    """Process a video file."""
    logger.info(f"Processing video: {input_path}")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise click.ClickException(f"Cannot open video: {input_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process frames
    frame_count = 0
    total_processing_time = 0
    quality_scores = []
    
    with click.progressbar(length=total_frames, label="Processing frames") as bar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Enhance frame
            enhanced, stats = enhancer.enhance_frame(frame, compute_metrics=compute_metrics)
            
            # Write to output
            if writer:
                writer.write(enhanced)
            
            # Collect statistics
            frame_count += 1
            total_processing_time += stats.processing_time_ms
            
            if compute_metrics and stats.uiqm_score:
                quality_scores.append(stats.uiqm_score)
            
            bar.update(1)
            
            # Log progress
            if frame_count % 100 == 0:
                avg_fps = frame_count * 1000 / total_processing_time if total_processing_time > 0 else 0
                logger.info(f"Processed {frame_count}/{total_frames} frames, avg FPS: {avg_fps:.1f}")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    
    # Display final statistics
    avg_processing_time = total_processing_time / frame_count if frame_count > 0 else 0
    avg_fps = frame_count * 1000 / total_processing_time if total_processing_time > 0 else 0
    
    click.echo(f"\nVideo Processing Complete:")
    click.echo(f"  Frames processed: {frame_count}")
    click.echo(f"  Average processing time: {avg_processing_time:.2f} ms/frame")
    click.echo(f"  Average FPS: {avg_fps:.1f}")
    
    if quality_scores:
        click.echo(f"  Average UIQM: {np.mean(quality_scores):.4f}")
        click.echo(f"  UIQM std: {np.std(quality_scores):.4f}")
    
    if output_path:
        click.echo(f"  Output saved: {output_path}")

def process_stream(enhancer: UnderwaterImageEnhancer,
                  source: str, output_path: Optional[str],
                  compute_metrics: bool, overlay_metrics: bool,
                  batch_size: int):
    """Process RTSP stream or live camera."""
    logger.info(f"Processing stream: {source}")
    
    # Stream processing callback
    def stream_callback(enhanced_frame, stats):
        try:
            # Add overlay if requested
            if overlay_metrics and stats and stats.uiqm_score:
                enhanced_frame = add_metrics_overlay(enhanced_frame, stats)
            
            # Display frame (if GUI available)
            if hasattr(cv2, 'imshow'):
                cv2.imshow('Enhanced Stream', enhanced_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    return False
            
            # Log statistics
            if stats and stats.fps > 0:
                logger.info(f"FPS: {stats.fps:.1f}, Processing: {stats.processing_time_ms:.1f}ms")
                if stats.uiqm_score:
                    logger.info(f"UIQM: {stats.uiqm_score:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Stream callback error: {e}")
            return False
    
    try:
        # Process stream
        enhancer.enhance_video_stream(source, output_path, stream_callback)
        
    except KeyboardInterrupt:
        click.echo("\nStream processing interrupted by user")
    except Exception as e:
        raise click.ClickException(f"Stream processing failed: {e}")
    finally:
        if hasattr(cv2, 'destroyAllWindows'):
            cv2.destroyAllWindows()

def add_metrics_overlay(image: np.ndarray, stats) -> np.ndarray:
    """Add metrics overlay to image."""
    overlay = image.copy()
    
    # Prepare text
    texts = [
        f"FPS: {stats.fps:.1f}",
        f"Time: {stats.processing_time_ms:.1f}ms"
    ]
    
    if stats.uiqm_score:
        texts.append(f"UIQM: {stats.uiqm_score:.4f}")
    if stats.uciqe_score:
        texts.append(f"UCIQE: {stats.uciqe_score:.4f}")
    
    # Draw text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    color = (0, 255, 0)  # Green
    thickness = 2
    
    y_offset = 30
    for i, text in enumerate(texts):
        y_pos = y_offset + i * 25
        cv2.putText(overlay, text, (10, y_pos), font, font_scale, color, thickness)
    
    return overlay

def display_image_metrics(stats, filename: str):
    """Display image quality metrics."""
    click.echo(f"\nQuality Metrics for {filename}:")
    
    if stats.uiqm_score is not None:
        click.echo(f"  UIQM: {stats.uiqm_score:.4f}")
    if stats.uciqe_score is not None:
        click.echo(f"  UCIQE: {stats.uciqe_score:.4f}")

def load_preset_config(preset: str) -> dict:
    """Load mission-specific preset configuration."""
    presets = {
        'port-survey': {
            'gamma_value': 1.3,
            'dehazing': {'beta': 1.2, 'tx': 0.15},
            'white_balance': {'method': 'underwater_physics', 'adaptation_strength': 0.9}
        },
        'diver-assist': {
            'gamma_value': 1.4,
            'use_lab_color': True,
            'denoise': True,
            'dehazing': {'beta': 0.8, 'tx': 0.1}
        },
        'deep-water': {
            'gamma_value': 1.5,
            'white_balance': {'method': 'underwater_physics', 'adaptation_strength': 1.0},
            'dehazing': {'beta': 1.5, 'tx': 0.2}
        }
    }
    
    return presets.get(preset, {})

def is_image_file(filename: str) -> bool:
    """Check if file is an image."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    return Path(filename).suffix.lower() in image_extensions

def is_video_file(filename: str) -> bool:
    """Check if file is a video."""
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}
    return Path(filename).suffix.lower() in video_extensions
```