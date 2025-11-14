"""
Canny Edge Detection for Segmentation Boundaries.

Implements the paper's approach: "Finally, we utilize the Canny edge detector
to detect the edge information of the boundary of each cluster."

This module extracts edges from segmentation labels using Canny edge detection.
"""

from dataclasses import dataclass
import numpy as np
import cv2


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class CannyConfig:
    """
    Configuration for Canny edge detection on segmentation boundaries.

    Attributes:
        low_threshold: Lower threshold for Canny edge detection (default: 50)
        high_threshold: Upper threshold for Canny edge detection (default: 150)
        morph_kernel_size: Morphological closing kernel size (default: 3, must be odd)
    """
    low_threshold: int = 50
    high_threshold: int = 150
    morph_kernel_size: int = 3

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.low_threshold >= self.high_threshold:
            raise ValueError(
                f"low_threshold ({self.low_threshold}) must be < "
                f"high_threshold ({self.high_threshold})"
            )
        if self.morph_kernel_size % 2 == 0 and self.morph_kernel_size > 0:
            raise ValueError(
                f"morph_kernel_size must be odd, got {self.morph_kernel_size}"
            )


# ============================================================================
# Results
# ============================================================================

@dataclass
class CannyResult:
    """
    Results from Canny edge detection.

    Attributes:
        edges: Binary edge image (H, W) with uint8 values [0, 255]
               White (255) = edge, Black (0) = non-edge
        method: Description of the method used
        config: Configuration used for detection
    """
    edges: np.ndarray
    method: str = "Canny Edge Detection on Segmentation Boundaries"
    config: CannyConfig = None

    def get_edge_pixel_count(self) -> int:
        """Get number of edge pixels."""
        return int((self.edges > 0).sum())

    def get_edge_ratio(self) -> float:
        """Get ratio of edge pixels to total pixels."""
        total = self.edges.size
        return self.get_edge_pixel_count() / total if total > 0 else 0.0

    def __str__(self) -> str:
        """String representation of results."""
        edge_pct = self.get_edge_ratio() * 100
        return (
            f"CannyResult(edges={self.get_edge_pixel_count()} pixels, "
            f"{edge_pct:.2f}% of image)"
        )


# ============================================================================
# Edge Detector
# ============================================================================

class CannyEdgeDetector:
    """
    Extract edges from segmentation labels using Canny edge detection.

    This implements the paper's approach: "we utilize the Canny edge detector
    to detect the edge information of the boundary of each cluster."

    The process:
    1. Find boundaries where different segments/clusters meet
    2. Apply Canny edge detection to refine these boundaries
    3. Return binary edge image (white edges on black background)

    Example:
        >>> config = CannyConfig(low_threshold=50, high_threshold=150)
        >>> detector = CannyEdgeDetector(config)
        >>> result = detector.detect_edges(evolved_labels)
        >>> edge_image = result.edges  # Binary edge map
    """

    def __init__(self, config: CannyConfig):
        """
        Initialize Canny edge detector.

        Args:
            config: Configuration for edge detection
        """
        self.config = config

    def detect_edges(self, labels: np.ndarray) -> CannyResult:
        """
        Extract edges from segmentation labels using Canny edge detection.

        This implements the paper's approach: "we utilize the Canny edge detector
        to detect the edge information of the boundary of each cluster."

        The process:
        1. Find boundaries where different segments/clusters meet
        2. Apply Canny edge detection to refine these boundaries
        3. Return binary edge image (white edges on black background)

        Args:
            labels: Segmentation label array (H, W) with cluster/segment IDs

        Returns:
            result: CannyResult with binary edge image

        Raises:
            ValueError: If labels is not 2D
        """
        # Ensure labels is 2D
        if labels.ndim == 1:
            raise ValueError(f"Labels must be 2D array (H, W), got shape {labels.shape}")

        # Convert labels to uint8 for edge detection
        # Normalize to 0-255 range for better Canny performance
        labels_normalized = (
            (labels - labels.min()) / max(labels.max() - labels.min(), 1) * 255
        ).astype(np.uint8)

        # Method 1: Create boundary map by detecting label changes
        # Compute gradients to find where labels change (cluster boundaries)
        grad_x = np.abs(np.diff(labels, axis=1, prepend=labels[:, :1]))
        grad_y = np.abs(np.diff(labels, axis=0, prepend=labels[:1, :]))
        boundaries = ((grad_x > 0) | (grad_y > 0)).astype(np.uint8) * 255

        # Method 2: Apply Canny to normalized labels
        # This detects edges in the label transitions
        canny_edges = cv2.Canny(
            labels_normalized,
            self.config.low_threshold,
            self.config.high_threshold
        )

        # Combine both methods: use boundary map OR Canny edges
        # This captures both sharp transitions and refined edge locations
        combined_edges = cv2.bitwise_or(boundaries, canny_edges)

        # Optional: morphological operations to clean up edges
        # Dilate slightly to make edges more visible, then erode to thin them
        if self.config.morph_kernel_size > 0:
            kernel = np.ones(
                (self.config.morph_kernel_size, self.config.morph_kernel_size),
                np.uint8
            )
            edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        else:
            edges = combined_edges

        return CannyResult(
            edges=edges,
            method="Canny Edge Detection on Segmentation Boundaries",
            config=self.config
        )
