"""
Canny Edge Detection for Boundary Extraction

Implementation of Stage 3 of the customized k-means pipeline.

Paper Reference: Islam et al. (2021), Section III.C
Parameters: low_threshold=50, high_threshold=150

This module applies Canny edge detection to cluster labels from level set
evolution to extract final segmentation boundaries.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np
import cv2


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CannyConfig:
    """
    Configuration for Canny edge detection.

    Parameters match those in config/default_params.yaml and the paper.
    """
    low_threshold: int = 50
    """Low threshold for Canny hysteresis. Paper: 50."""

    high_threshold: int = 150
    """High threshold for Canny hysteresis. Paper: 150."""

    gaussian_kernel_size: tuple[int, int] = (3, 3)
    """Gaussian blur kernel size for noise reduction before Canny."""

    gaussian_sigma: float = 0.0
    """Gaussian blur sigma. 0 = auto-calculate from kernel size."""

    scale_labels: bool = True
    """Whether to scale cluster labels to [0, 255] for better contrast."""

    apply_morphology: bool = False
    """Whether to apply morphological operations for cleanup."""

    morph_kernel_size: tuple[int, int] = (2, 2)
    """Morphological operation kernel size."""

    morph_operation: str = 'close'
    """Morphological operation type: 'close', 'dilate', 'erode', 'open'."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.low_threshold < 0:
            raise ValueError(f"low_threshold must be >= 0, got {self.low_threshold}")
        if self.high_threshold <= self.low_threshold:
            raise ValueError(
                f"high_threshold must be > low_threshold, "
                f"got low={self.low_threshold}, high={self.high_threshold}"
            )
        if len(self.gaussian_kernel_size) != 2:
            raise ValueError(
                f"gaussian_kernel_size must be (height, width), "
                f"got {self.gaussian_kernel_size}"
            )
        if self.gaussian_kernel_size[0] % 2 == 0 or self.gaussian_kernel_size[1] % 2 == 0:
            raise ValueError(
                f"gaussian_kernel_size must be odd, got {self.gaussian_kernel_size}"
            )
        if self.morph_operation not in ['close', 'dilate', 'erode', 'open']:
            raise ValueError(
                f"morph_operation must be 'close', 'dilate', 'erode', or 'open', "
                f"got '{self.morph_operation}'"
            )


# ============================================================================
# Results
# ============================================================================

@dataclass
class EdgeDetectionResult:
    """
    Results from edge detection.

    Contains edge map and optional intermediate results.
    """
    edges: np.ndarray
    """Binary edge map. Shape: (H, W), values {0, 255}"""

    labels_preprocessed: Optional[np.ndarray] = None
    """Preprocessed labels before edge detection. Shape: (H, W)"""

    labels_blurred: Optional[np.ndarray] = None
    """Blurred labels after Gaussian smoothing. Shape: (H, W)"""

    def get_edge_count(self) -> int:
        """
        Get number of edge pixels.

        Returns:
            count: Number of non-zero pixels in edge map
        """
        return np.count_nonzero(self.edges)

    def get_edge_density(self) -> float:
        """
        Get edge density (fraction of image that is edges).

        Returns:
            density: Edge pixel count / total pixels
        """
        return self.get_edge_count() / self.edges.size


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BaseEdgeDetector(ABC):
    """
    Abstract base class for edge detection implementations.

    This interface allows for different edge detection methods:
    - CannyEdgeDetector: Standard Canny edge detection (production)
    - SobelEdgeDetector: Sobel gradient-based detection (alternative)
    - CustomEdgeDetector: From-scratch implementation (future)
    """

    def __init__(self, config: CannyConfig):
        """
        Initialize edge detector.

        Args:
            config: Configuration parameters
        """
        self.config = config

    @abstractmethod
    def detect_edges(
        self,
        labels: np.ndarray,
        return_intermediates: bool = False
    ) -> EdgeDetectionResult:
        """
        Detect edges from cluster labels.

        Args:
            labels: Cluster labels from level set evolution, shape (H, W)
            return_intermediates: Whether to return intermediate results

        Returns:
            result: EdgeDetectionResult with edge map
        """
        pass


# ============================================================================
# Canny Implementation
# ============================================================================

class CannyEdgeDetector(BaseEdgeDetector):
    """
    Canny edge detection for cluster boundary extraction.

    This is the primary implementation that applies Canny edge detection
    to refined cluster labels from level set evolution.

    Based on: research/custom_k_means/paper_results_v2.py (lines 309-343)
              research/intuition/custom_kmeans/customized_kmeans.py (lines 241-281)

    Algorithm:
    1. Scale cluster labels to [0, 255] for maximum contrast
    2. Apply Gaussian blur to reduce noise
    3. Apply Canny edge detection with paper's parameters (50, 150)
    4. Optionally apply morphological operations for cleanup
    """

    def __init__(self, config: CannyConfig):
        """
        Initialize Canny edge detector.

        Args:
            config: Canny configuration
        """
        super().__init__(config)

    def detect_edges(
        self,
        labels: np.ndarray,
        return_intermediates: bool = False
    ) -> EdgeDetectionResult:
        """
        Apply Canny edge detection to extract cluster boundaries.

        Implements the paper's Stage 3: edge detection with Canny
        using low_threshold=50, high_threshold=150.

        Args:
            labels: Cluster labels from level set, shape (H, W)
            return_intermediates: If True, include preprocessed/blurred labels

        Returns:
            result: EdgeDetectionResult with binary edge map
        """
        # Validate input
        if labels.ndim != 2:
            raise ValueError(
                f"labels must be 2D array (H, W), got shape {labels.shape}"
            )

        # 1. Preprocess labels: convert to uint8 and optionally scale
        labels_preprocessed = self._preprocess_labels(labels)

        # 2. Apply Gaussian blur to reduce noise
        labels_blurred = cv2.GaussianBlur(
            labels_preprocessed,
            self.config.gaussian_kernel_size,
            self.config.gaussian_sigma
        )

        # 3. Apply Canny edge detection with paper's parameters
        edges = cv2.Canny(
            labels_blurred,
            self.config.low_threshold,
            self.config.high_threshold
        )

        # 4. Optionally apply morphological operations for cleanup
        if self.config.apply_morphology:
            edges = self._apply_morphological_operations(edges)

        # Build result
        result = EdgeDetectionResult(
            edges=edges,
            labels_preprocessed=labels_preprocessed if return_intermediates else None,
            labels_blurred=labels_blurred if return_intermediates else None
        )

        return result

    def _preprocess_labels(self, labels: np.ndarray) -> np.ndarray:
        """
        Preprocess cluster labels for edge detection.

        Converts to uint8 and optionally scales to [0, 255] range to
        maximize contrast between different clusters.

        Args:
            labels: Cluster labels (H, W) with integer values [0, n_clusters-1]

        Returns:
            preprocessed: Labels as uint8, optionally scaled to [0, 255]
        """
        labels_uint8 = labels.astype(np.uint8)

        if self.config.scale_labels:
            n_clusters = labels.max() + 1

            if n_clusters > 1:
                # Scale to [0, 255] to maximize contrast
                labels_scaled = (labels_uint8 * (255.0 / (n_clusters - 1))).astype(np.uint8)
            else:
                # Single cluster - no scaling needed
                labels_scaled = labels_uint8

            return labels_scaled
        else:
            return labels_uint8

    def _apply_morphological_operations(self, edges: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up edge map.

        Operations:
        - 'close': Fill small gaps in edges (dilation + erosion)
        - 'open': Remove small noise (erosion + dilation)
        - 'dilate': Thicken edges
        - 'erode': Thin edges

        Args:
            edges: Binary edge map (H, W)

        Returns:
            cleaned: Edge map after morphological operation
        """
        kernel = np.ones(self.config.morph_kernel_size, np.uint8)

        if self.config.morph_operation == 'close':
            cleaned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        elif self.config.morph_operation == 'open':
            cleaned = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        elif self.config.morph_operation == 'dilate':
            cleaned = cv2.dilate(edges, kernel, iterations=1)
        elif self.config.morph_operation == 'erode':
            cleaned = cv2.erode(edges, kernel, iterations=1)
        else:
            # Should never reach here due to __post_init__ validation
            cleaned = edges

        return cleaned

    def detect_edges_from_image(
        self,
        image: np.ndarray,
        return_intermediates: bool = False
    ) -> EdgeDetectionResult:
        """
        Convenience method: Apply Canny directly to grayscale image.

        This is useful for detecting edges in the original image rather
        than cluster labels. Not used in the paper's pipeline but provided
        for experimentation.

        Args:
            image: RGB or grayscale image
            return_intermediates: If True, include preprocessed/blurred image

        Returns:
            result: EdgeDetectionResult with edge map
        """
        # Convert to grayscale if needed
        if image.ndim == 3:
            # RGB to grayscale
            if image.max() <= 1.0:
                gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = image.astype(np.uint8)

        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            gray,
            self.config.gaussian_kernel_size,
            self.config.gaussian_sigma
        )

        # Apply Canny
        edges = cv2.Canny(
            blurred,
            self.config.low_threshold,
            self.config.high_threshold
        )

        # Optionally apply morphology
        if self.config.apply_morphology:
            edges = self._apply_morphological_operations(edges)

        result = EdgeDetectionResult(
            edges=edges,
            labels_preprocessed=gray if return_intermediates else None,
            labels_blurred=blurred if return_intermediates else None
        )

        return result


# ============================================================================
# Alternative Implementation (Sobel-based)
# ============================================================================

class SobelEdgeDetector(BaseEdgeDetector):
    """
    Sobel gradient-based edge detection for cluster boundaries.

    Alternative implementation that uses Sobel operators to detect
    boundaries directly from label gradients. Useful when Canny
    produces too many fragmented edges.

    Based on: research/custom_k_means/paper_results_v2.py extract_contours_from_labels
    """

    def detect_edges(
        self,
        labels: np.ndarray,
        return_intermediates: bool = False
    ) -> EdgeDetectionResult:
        """
        Detect edges using Sobel gradient operators.

        Computes gradient magnitude of label map and thresholds to
        find cluster boundaries.

        Args:
            labels: Cluster labels, shape (H, W)
            return_intermediates: If True, include gradient magnitude

        Returns:
            result: EdgeDetectionResult with edge map
        """
        from scipy.ndimage import sobel

        # Compute gradients in x and y directions
        grad_x = np.abs(sobel(labels.astype(float), axis=1))
        grad_y = np.abs(sobel(labels.astype(float), axis=0))

        # Gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Threshold to get binary edge map
        # Any non-zero gradient indicates a boundary between clusters
        edges = (grad_magnitude > 0).astype(np.uint8) * 255

        # Optionally apply morphology
        if self.config.apply_morphology:
            kernel = np.ones(self.config.morph_kernel_size, np.uint8)
            if self.config.morph_operation == 'close':
                edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
            elif self.config.morph_operation == 'open':
                edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

        result = EdgeDetectionResult(
            edges=edges,
            labels_preprocessed=grad_magnitude if return_intermediates else None
        )

        return result
