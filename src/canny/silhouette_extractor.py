"""
Silhouette Extraction and Mask Filling

This module provides methods for extracting clean object silhouettes from
segmentation results and converting them to filled binary masks suitable
for PRI evaluation.

Three extraction methods are provided:
1. From Canny Edges: Morphological closing on edge maps
2. From Labels: Direct extraction from segmentation (RECOMMENDED)
3. Convex Hull: Smoothest possible outline

Based on: custom_k_means/src/experiments/42044/42044.py (lines 806-1034)
          custom_k_means/src/experiments/12074/12074.py (lines 806-1040)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum
import time
import numpy as np
import cv2


# ============================================================================
# Enums
# ============================================================================

class SilhouetteMethod(Enum):
    """Available silhouette extraction methods."""
    CANNY_EDGES = "canny_edges"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SilhouetteConfig:
    """
    Configuration for silhouette extraction using Canny edges.

    Parameters:
        method: Silhouette extraction method (only CANNY_EDGES supported)
        h: Image height (required for mask filling)
        w: Image width (required for mask filling)
        canny_closing_kernel_size: Morphological closing kernel size
        canny_closing_iterations: Morphological closing iterations
        canny_min_area: Minimum contour area to keep
        canny_line_thickness: Line thickness for drawing silhouette
        fill_closing_kernel_size: Kernel size for closing gaps before filling
        fill_closing_iterations: Iterations of closing before filling
    """
    method: SilhouetteMethod = SilhouetteMethod.CANNY_EDGES
    """Silhouette extraction method."""

    h: int = 0
    """Image height (required for mask filling)."""

    w: int = 0
    """Image width (required for mask filling)."""

    # Canny Edges method parameters
    canny_closing_kernel_size: int = 5
    """Morphological closing kernel size."""

    canny_closing_iterations: int = 2
    """Morphological closing iterations."""

    canny_min_area: int = 1000
    """Minimum contour area to keep."""

    canny_line_thickness: int = 1
    """Line thickness for drawing silhouette."""

    # Mask filling parameters
    fill_closing_kernel_size: int = 5
    """Kernel size for closing gaps before filling."""

    fill_closing_iterations: int = 2
    """Iterations of closing before filling."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.h <= 0 or self.w <= 0:
            raise ValueError(f"Image dimensions must be positive, got h={self.h}, w={self.w}")
        if self.canny_closing_kernel_size % 2 == 0:
            raise ValueError(f"Kernel sizes must be odd, got {self.canny_closing_kernel_size}")
        if self.fill_closing_kernel_size % 2 == 0:
            raise ValueError(f"Kernel sizes must be odd, got {self.fill_closing_kernel_size}")


# ============================================================================
# Results
# ============================================================================

@dataclass
class SilhouetteResult:
    """
    Results from silhouette extraction.

    Contains both the visual silhouette (RGB image with black contours on white)
    and the filled binary mask suitable for PRI evaluation.
    """
    silhouette_image: np.ndarray
    """Silhouette as RGB image: black contours on white background (H, W, 3)."""

    binary_mask: np.ndarray
    """Filled binary mask: 0=background, 1=foreground (H, W)."""

    method: SilhouetteMethod
    """Method used for extraction."""

    contour_count: int
    """Number of valid contours extracted."""

    extraction_time: float
    """Time taken to extract silhouette (seconds)."""

    metadata: Optional[dict] = None
    """Optional metadata about the extraction process."""

    def get_foreground_pixels(self) -> int:
        """Get number of foreground pixels in binary mask."""
        return int(self.binary_mask.sum())

    def get_foreground_ratio(self) -> float:
        """Get ratio of foreground to total pixels."""
        total = self.binary_mask.size
        return self.get_foreground_pixels() / total if total > 0 else 0.0

    def __str__(self) -> str:
        """String representation of results."""
        fg_pct = self.get_foreground_ratio() * 100
        return (
            f"SilhouetteResult(method={self.method.value}, "
            f"contours={self.contour_count}, "
            f"foreground={fg_pct:.1f}%, "
            f"time={self.extraction_time*1000:.1f}ms)"
        )


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BaseSilhouetteExtractor(ABC):
    """
    Abstract base class for silhouette extraction implementations.

    This interface allows for different silhouette extraction methods:
    - CannyEdgesSilhouetteExtractor: From Canny edge maps
    - LabelsSilhouetteExtractor: From segmentation labels (recommended)
    - ConvexHullSilhouetteExtractor: Convex hull outline
    """

    def __init__(self, config: SilhouetteConfig):
        """
        Initialize silhouette extractor.

        Args:
            config: Extraction configuration
        """
        self.config = config

    @abstractmethod
    def extract(
        self,
        edge_map: Optional[np.ndarray] = None,
        labels_2d: Optional[np.ndarray] = None
    ) -> SilhouetteResult:
        """
        Extract silhouette and convert to filled binary mask.

        Args:
            edge_map: Binary edge map from Canny detector, shape (H, W)
            labels_2d: Cluster labels from level set, shape (H, W)

        Returns:
            result: SilhouetteResult with silhouette image and binary mask

        Notes:
            Different methods require different inputs:
            - CANNY_EDGES: requires edge_map
            - LABELS: requires labels_2d
            - CONVEX_HULL: requires labels_2d
        """
        pass


# ============================================================================
# Method 1: From Canny Edges
# ============================================================================

class CannyEdgesSilhouetteExtractor(BaseSilhouetteExtractor):
    """
    Extract silhouette from Canny edge map using morphological closing.

    This method:
    1. Closes gaps in edges using morphological operations
    2. Fills holes using flood fill
    3. Finds external contours
    4. Draws clean outlines on white background
    5. Fills contours to create binary mask

    Based on: extract_silhouette_method1() in notebooks
    """

    def extract(
        self,
        edge_map: Optional[np.ndarray] = None,
        labels_2d: Optional[np.ndarray] = None
    ) -> SilhouetteResult:
        """
        Extract silhouette from Canny edges.

        Args:
            edge_map: Binary edge map from Canny, shape (H, W) - REQUIRED
            labels_2d: Not used by this method

        Returns:
            result: SilhouetteResult with silhouette and filled mask
        """
        if edge_map is None:
            raise ValueError("CannyEdgesSilhouetteExtractor requires edge_map")

        start_time = time.time()
        h, w = self.config.h, self.config.w

        # 1. Close gaps in edges
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.canny_closing_kernel_size, self.config.canny_closing_kernel_size)
        )
        closed = cv2.morphologyEx(
            edge_map,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.canny_closing_iterations
        )

        # 2. Fill holes using flood fill
        inverted = cv2.bitwise_not(closed)
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(inverted, mask, (0, 0), 255)
        filled = cv2.bitwise_not(inverted)

        # 3. Find external contours
        contours, _ = cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Draw clean outlines on white background
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0
        for contour in contours:
            if cv2.contourArea(contour) > self.config.canny_min_area:
                cv2.drawContours(
                    silhouette,
                    [contour],
                    -1,
                    (0, 0, 0),
                    self.config.canny_line_thickness
                )
                valid_contours += 1

        elapsed_time = time.time() - start_time

        # 5. Convert to filled binary mask
        binary_mask = silhouette_to_mask(
            silhouette,
            h, w,
            self.config.fill_closing_kernel_size,
            self.config.fill_closing_iterations
        )

        return SilhouetteResult(
            silhouette_image=silhouette,
            binary_mask=binary_mask,
            method=SilhouetteMethod.CANNY_EDGES,
            contour_count=valid_contours,
            extraction_time=elapsed_time,
            metadata={
                'closing_kernel': self.config.canny_closing_kernel_size,
                'closing_iterations': self.config.canny_closing_iterations,
                'min_area': self.config.canny_min_area
            }
        )


# ============================================================================
# Helper Functions
# ============================================================================

def silhouette_to_mask(
    silhouette_rgb: np.ndarray,
    h: int,
    w: int,
    closing_kernel_size: int = 5,
    closing_iterations: int = 2
) -> np.ndarray:
    """
    Convert silhouette (contours on white background) to filled binary mask.

    This is the CORE preprocessing step that bridges silhouettes to PRI evaluation.
    It takes a visual silhouette (black contours on white) and fills it to create
    a binary segmentation mask suitable for comparison with ground truth.

    Args:
        silhouette_rgb: RGB image with black contours on white, shape (H, W, 3)
        h: Image height
        w: Image width
        closing_kernel_size: Kernel size for closing gaps (must be odd)
        closing_iterations: Number of closing iterations

    Returns:
        mask: Binary mask where 1=foreground, 0=background, shape (H, W)

    Example:
        >>> silhouette = extract_silhouette(...)  # Black contours on white
        >>> mask = silhouette_to_mask(silhouette, h, w)
        >>> pri = evaluator.evaluate(mask, [ground_truth])
    """
    # 1. Convert to grayscale
    gray = cv2.cvtColor(silhouette_rgb, cv2.COLOR_RGB2GRAY)

    # 2. Threshold: anything not pure white is a contour (black lines)
    _, contour_binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)

    # 3. Close gaps in contours to ensure they're continuous
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (closing_kernel_size, closing_kernel_size))
    closed = cv2.morphologyEx(contour_binary, cv2.MORPH_CLOSE, kernel, iterations=closing_iterations)

    # 4. Find contours from the closed edge map
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 5. Create empty mask and fill all contours
    filled_mask = np.zeros((h, w), dtype=np.uint8)

    # KEY STEP: Draw filled contours (thickness=-1 means fill completely)
    cv2.drawContours(filled_mask, contours, -1, 255, thickness=-1)

    # 6. Fallback: If contours aren't closed, use flood fill from corners
    if filled_mask.sum() == 0 or filled_mask.sum() < (h * w * 0.01):
        # Flood fill from background
        filled_mask = np.ones((h, w), dtype=np.uint8) * 255
        # Flood fill from all four corners to identify background
        flood_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(filled_mask, flood_mask, (0, 0), 0)
        cv2.floodFill(filled_mask, flood_mask, (w-1, 0), 0)
        cv2.floodFill(filled_mask, flood_mask, (0, h-1), 0)
        cv2.floodFill(filled_mask, flood_mask, (w-1, h-1), 0)

    # 7. Convert to binary labels (0 or 1)
    binary_mask = (filled_mask > 127).astype(np.uint8)

    return binary_mask


# ============================================================================
# Factory Function
# ============================================================================

def create_silhouette_extractor(config: SilhouetteConfig) -> BaseSilhouetteExtractor:
    """
    Factory function to create silhouette extractor.

    Only CANNY_EDGES method is supported.

    Args:
        config: Silhouette extraction configuration

    Returns:
        extractor: CannyEdgesSilhouetteExtractor instance

    Raises:
        ValueError: If method is not CANNY_EDGES

    Example:
        >>> config = SilhouetteConfig(
        ...     method=SilhouetteMethod.CANNY_EDGES,
        ...     h=481, w=321
        ... )
        >>> extractor = create_silhouette_extractor(config)
        >>> result = extractor.extract(edge_map=canny_edges)
        >>> # result.binary_mask is ready for PRI!
        >>> pri = evaluator.evaluate(result.binary_mask, [ground_truth])
    """
    if config.method == SilhouetteMethod.CANNY_EDGES:
        return CannyEdgesSilhouetteExtractor(config)
    else:
        raise ValueError(f"Only CANNY_EDGES method is supported, got: {config.method}")
