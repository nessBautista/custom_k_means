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
    LABELS = "labels"
    CONVEX_HULL = "convex_hull"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class SilhouetteConfig:
    """
    Configuration for silhouette extraction.

    Different parameters are used depending on the selected method.
    """
    method: SilhouetteMethod = SilhouetteMethod.LABELS
    """Silhouette extraction method."""

    h: int = 0
    """Image height (required for mask filling)."""

    w: int = 0
    """Image width (required for mask filling)."""

    # Method 1: From Canny Edges - Parameters
    canny_closing_kernel_size: int = 5
    """Morphological closing kernel size (Method 1)."""

    canny_closing_iterations: int = 2
    """Morphological closing iterations (Method 1)."""

    canny_min_area: int = 1000
    """Minimum contour area to keep (Method 1)."""

    canny_line_thickness: int = 1
    """Line thickness for drawing silhouette (Method 1)."""

    # Method 2: From Labels - Parameters (RECOMMENDED)
    labels_kernel_size: int = 3
    """Cleanup kernel size for morphological operations (Method 2)."""

    labels_smoothing_epsilon: float = 0.001
    """Douglas-Peucker smoothing epsilon factor (Method 2)."""

    labels_min_area: int = 500
    """Minimum contour area to keep (Method 2)."""

    labels_line_thickness: int = 1
    """Line thickness for drawing silhouette (Method 2)."""

    # Method 3: Convex Hull - Parameters
    hull_line_thickness: int = 2
    """Line thickness for drawing convex hull (Method 3)."""

    # Mask filling parameters (common to all methods)
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
        if self.labels_kernel_size % 2 == 0:
            raise ValueError(f"Kernel sizes must be odd, got {self.labels_kernel_size}")
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
# Method 2: From Segmentation Labels (RECOMMENDED)
# ============================================================================

class LabelsSilhouetteExtractor(BaseSilhouetteExtractor):
    """
    Extract silhouette from segmentation labels - cleanest method.

    This method:
    1. Identifies background label (most common at borders)
    2. Creates foreground mask
    3. Cleans up mask with morphological operations
    4. Finds external contours
    5. Smooths contours with Douglas-Peucker
    6. Draws clean outlines on white background
    7. Fills contours to create binary mask

    This is the RECOMMENDED method as it produces the cleanest results
    and works directly with semantic segmentation.

    Based on: extract_silhouette_method2() in notebooks
    """

    def extract(
        self,
        edge_map: Optional[np.ndarray] = None,
        labels_2d: Optional[np.ndarray] = None
    ) -> SilhouetteResult:
        """
        Extract silhouette from segmentation labels.

        Args:
            edge_map: Not used by this method
            labels_2d: Cluster labels from level set, shape (H, W) - REQUIRED

        Returns:
            result: SilhouetteResult with silhouette and filled mask
        """
        if labels_2d is None:
            raise ValueError("LabelsSilhouetteExtractor requires labels_2d")

        start_time = time.time()
        h, w = self.config.h, self.config.w

        # 1. Find background label (most common at borders)
        border_pixels = np.concatenate([
            labels_2d[0, :], labels_2d[-1, :],
            labels_2d[:, 0], labels_2d[:, -1]
        ])
        background_label = np.bincount(border_pixels).argmax()

        # 2. Create foreground mask
        fg_mask = (labels_2d != background_label).astype(np.uint8) * 255

        # 3. Clean up mask
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.labels_kernel_size, self.config.labels_kernel_size)
        )
        cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        # 4. Find external contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 5. Draw clean outlines with smoothing
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0

        for contour in contours:
            if cv2.contourArea(contour) > self.config.labels_min_area:
                # Smooth the contour with Douglas-Peucker
                arc_length = cv2.arcLength(contour, True)
                smoothed = cv2.approxPolyDP(
                    contour,
                    self.config.labels_smoothing_epsilon * arc_length,
                    True
                )
                cv2.drawContours(
                    silhouette,
                    [smoothed],
                    -1,
                    (0, 0, 0),
                    self.config.labels_line_thickness
                )
                valid_contours += 1

        elapsed_time = time.time() - start_time

        # 6. Convert to filled binary mask
        binary_mask = silhouette_to_mask(
            silhouette,
            h, w,
            self.config.fill_closing_kernel_size,
            self.config.fill_closing_iterations
        )

        return SilhouetteResult(
            silhouette_image=silhouette,
            binary_mask=binary_mask,
            method=SilhouetteMethod.LABELS,
            contour_count=valid_contours,
            extraction_time=elapsed_time,
            metadata={
                'background_label': int(background_label),
                'kernel_size': self.config.labels_kernel_size,
                'smoothing_epsilon': self.config.labels_smoothing_epsilon,
                'min_area': self.config.labels_min_area
            }
        )


# ============================================================================
# Method 3: Convex Hull
# ============================================================================

class ConvexHullSilhouetteExtractor(BaseSilhouetteExtractor):
    """
    Extract convex hull silhouette - smoothest possible outline.

    This method:
    1. Identifies background label
    2. Creates foreground mask
    3. Finds largest contour
    4. Computes convex hull
    5. Draws hull on white background
    6. Fills hull to create binary mask

    This produces the simplest, smoothest outline but loses concave features.

    Based on: extract_silhouette_method3() in notebooks
    """

    def extract(
        self,
        edge_map: Optional[np.ndarray] = None,
        labels_2d: Optional[np.ndarray] = None
    ) -> SilhouetteResult:
        """
        Extract convex hull silhouette from segmentation labels.

        Args:
            edge_map: Not used by this method
            labels_2d: Cluster labels from level set, shape (H, W) - REQUIRED

        Returns:
            result: SilhouetteResult with convex hull silhouette and mask
        """
        if labels_2d is None:
            raise ValueError("ConvexHullSilhouetteExtractor requires labels_2d")

        start_time = time.time()
        h, w = self.config.h, self.config.w

        # 1. Find background label
        border_pixels = np.concatenate([
            labels_2d[0, :], labels_2d[-1, :],
            labels_2d[:, 0], labels_2d[:, -1]
        ])
        background_label = np.bincount(border_pixels).argmax()

        # 2. Create foreground mask
        fg_mask = (labels_2d != background_label).astype(np.uint8) * 255

        # 3. Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 4. Draw convex hull
        silhouette = np.ones((h, w, 3), dtype=np.uint8) * 255
        valid_contours = 0

        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)

            # Compute convex hull
            hull = cv2.convexHull(largest)

            # Draw hull
            cv2.drawContours(silhouette, [hull], -1, (0, 0, 0), self.config.hull_line_thickness)
            valid_contours = 1

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
            method=SilhouetteMethod.CONVEX_HULL,
            contour_count=valid_contours,
            extraction_time=elapsed_time,
            metadata={
                'background_label': int(background_label),
                'line_thickness': self.config.hull_line_thickness
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
    Factory function to create appropriate silhouette extractor.

    Args:
        config: Silhouette extraction configuration

    Returns:
        extractor: Concrete silhouette extractor instance

    Example:
        >>> config = SilhouetteConfig(
        ...     method=SilhouetteMethod.LABELS,
        ...     h=481, w=321
        ... )
        >>> extractor = create_silhouette_extractor(config)
        >>> result = extractor.extract(labels_2d=refined_labels)
        >>> # result.binary_mask is ready for PRI!
        >>> pri = evaluator.evaluate(result.binary_mask, [ground_truth])
    """
    if config.method == SilhouetteMethod.CANNY_EDGES:
        return CannyEdgesSilhouetteExtractor(config)
    elif config.method == SilhouetteMethod.LABELS:
        return LabelsSilhouetteExtractor(config)
    elif config.method == SilhouetteMethod.CONVEX_HULL:
        return ConvexHullSilhouetteExtractor(config)
    else:
        raise ValueError(f"Unknown silhouette method: {config.method}")
