"""
Contour Extraction for Segmentation Boundaries

Implementation of Stage 4 of the customized k-means pipeline.

This module provides multiple methods for extracting clean contours from
edge maps or cluster labels. Three advanced methods are provided:

1. Morphological: Uses morphological operations + external contours
2. Sobel Threshold: Direct extraction from labels using Sobel (RECOMMENDED)
3. Convex Hull: Merges regions and extracts convex hulls

Based on: research/custom_k_means/paper_results_v2.py
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import numpy as np
import cv2
from scipy.ndimage import sobel, median_filter


# ============================================================================
# Enums
# ============================================================================

class ContourMethod(Enum):
    """Available contour extraction methods."""
    MORPHOLOGICAL = "morphological"
    SOBEL_THRESHOLD = "sobel_threshold"
    CONVEX_HULL = "convex_hull"
    BASIC = "basic"  # Legacy: simple Canny + filtering


class FilterStrategy(Enum):
    """Contour filtering strategies for basic method."""
    ALL = "all"
    LARGEST = "largest"
    TOP_N = "top_n"
    AREA_THRESHOLD = "area_threshold"


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ContourConfig:
    """
    Configuration for contour extraction.

    Different parameters are used depending on the selected method.
    """
    method: ContourMethod = ContourMethod.SOBEL_THRESHOLD
    """Contour extraction method."""

    # Common parameters
    min_contour_area: int = 500
    """Minimum contour area (in pixels) to keep."""

    epsilon_factor: float = 0.005
    """Douglas-Peucker simplification factor (0 = no simplification)."""

    # Morphological method parameters
    morph_smooth_iterations: int = 3
    """Number of morphological closing iterations (morphological method)."""

    morph_num_major_regions: int = 3
    """Number of major semantic regions to quantize to (morphological method)."""

    morph_kernel_size: int = 7
    """Morphological kernel size (morphological method)."""

    morph_median_filter_size: int = 7
    """Median filter size for label smoothing (morphological method)."""

    # Sobel threshold method parameters
    sobel_threshold_percentile: float = 80.0
    """Percentile threshold for boundary strength (sobel method).
    80 = keep top 20% strongest boundaries."""

    sobel_median_filter_size: int = 5
    """Median filter size for label smoothing (sobel method)."""

    sobel_morph_iterations: int = 2
    """Morphological closing iterations (sobel method)."""

    # Convex hull method parameters
    hull_min_region_size: int = 1000
    """Minimum region size before merging (convex hull method)."""

    hull_merge_iterations: int = 3
    """Number of merge-and-smooth iterations (convex hull method)."""

    hull_median_filter_size: int = 5
    """Median filter size during merging (convex hull method)."""

    # Basic/legacy method parameters
    basic_filter_strategy: FilterStrategy = FilterStrategy.TOP_N
    """Filtering strategy for basic method."""

    basic_n_contours: int = 5
    """Number of contours to keep for 'top_n' strategy (basic method)."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.min_contour_area < 0:
            raise ValueError(f"min_contour_area must be >= 0, got {self.min_contour_area}")
        if self.epsilon_factor < 0:
            raise ValueError(f"epsilon_factor must be >= 0, got {self.epsilon_factor}")
        if self.sobel_threshold_percentile < 0 or self.sobel_threshold_percentile > 100:
            raise ValueError(
                f"sobel_threshold_percentile must be in [0, 100], "
                f"got {self.sobel_threshold_percentile}"
            )


# ============================================================================
# Results
# ============================================================================

@dataclass
class ContourResult:
    """
    Results from contour extraction.

    Contains extracted contours and optional metadata.
    """
    contours: List[np.ndarray]
    """List of contours, each shape (N, 1, 2) in OpenCV format."""

    method: ContourMethod
    """Method used for extraction."""

    hierarchy: Optional[np.ndarray] = None
    """Contour hierarchy from cv2.findContours (for basic method)."""

    edge_map: Optional[np.ndarray] = None
    """Binary edge map used for extraction (if available)."""

    labels_processed: Optional[np.ndarray] = None
    """Processed cluster labels (if available)."""

    def get_contour_count(self) -> int:
        """Get number of extracted contours."""
        return len(self.contours)

    def get_total_contour_area(self) -> float:
        """Get total area of all contours."""
        return sum(cv2.contourArea(c) for c in self.contours)

    def get_contour_areas(self) -> List[float]:
        """Get areas of all contours."""
        return [cv2.contourArea(c) for c in self.contours]

    def get_largest_contour(self) -> Optional[np.ndarray]:
        """Get the largest contour by area."""
        if not self.contours:
            return None
        return max(self.contours, key=cv2.contourArea)


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BaseContourExtractor(ABC):
    """
    Abstract base class for contour extraction implementations.

    This interface allows for different contour extraction methods:
    - MorphologicalContourExtractor: Morphological ops + external contours
    - SobelThresholdContourExtractor: Direct extraction from labels (recommended)
    - ConvexHullContourExtractor: Convex hulls of merged regions
    - BasicContourExtractor: Simple Canny + filtering (legacy)
    """

    def __init__(self, config: ContourConfig):
        """
        Initialize contour extractor.

        Args:
            config: Configuration parameters
        """
        self.config = config

    @abstractmethod
    def extract_contours(
        self,
        labels: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        return_intermediates: bool = False
    ) -> ContourResult:
        """
        Extract contours from cluster labels or edge map.

        Args:
            labels: Cluster labels from level set, shape (H, W)
            edges: Binary edge map from Canny, shape (H, W)
            return_intermediates: Whether to return intermediate results

        Returns:
            result: ContourResult with extracted contours

        Notes:
            Different methods require different inputs:
            - Morphological: requires both labels and edges
            - Sobel Threshold: requires only labels
            - Convex Hull: requires only labels
            - Basic: requires only edges
        """
        pass


# ============================================================================
# Method 1: Morphological + External Contours
# ============================================================================

class MorphologicalContourExtractor(BaseContourExtractor):
    """
    Extract clean external contours using morphological operations.

    This method uses aggressive smoothing and morphological operations to merge
    fine-grained clusters into major semantic regions, then extracts only
    external boundaries.

    Based on: research/custom_k_means/paper_results_v2.py extract_clean_contours_from_canny
    """

    def extract_contours(
        self,
        labels: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        return_intermediates: bool = False
    ) -> ContourResult:
        """
        Extract contours using morphological operations.

        Args:
            labels: Cluster labels (H, W) - REQUIRED
            edges: Binary edge map (H, W) - not used but accepted for interface
            return_intermediates: If True, include processed labels

        Returns:
            result: ContourResult with simplified external contours
        """
        if labels is None:
            raise ValueError("MorphologicalContourExtractor requires labels")

        # 1. Aggressive smoothing to merge similar clusters
        labels_smooth = median_filter(labels, size=self.config.morph_median_filter_size)

        # 2. Quantize to fewer major regions (typically 3-5 major semantic objects)
        labels_max = labels_smooth.max()
        if labels_max > 0:
            labels_quantized = (
                labels_smooth * self.config.morph_num_major_regions / labels_max
            ).astype(np.uint8)
        else:
            labels_quantized = labels_smooth.astype(np.uint8)

        # 3. Morphological closing to fill holes and smooth boundaries
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        labels_closed = cv2.morphologyEx(
            labels_quantized,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.morph_smooth_iterations
        )

        # 4. Extract external contours for each major region
        all_contours = []

        for label_id in np.unique(labels_closed):
            if label_id == 0:
                continue

            # Create binary mask for this region
            mask = (labels_closed == label_id).astype(np.uint8)

            # Find only external contours
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                area = cv2.contourArea(contour)
                if area >= self.config.min_contour_area:
                    # Simplify contour using Douglas-Peucker
                    simplified = self._simplify_contour(contour)
                    all_contours.append(simplified)

        return ContourResult(
            contours=all_contours,
            method=ContourMethod.MORPHOLOGICAL,
            labels_processed=labels_closed if return_intermediates else None
        )

    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """Simplify contour using Douglas-Peucker algorithm."""
        if self.config.epsilon_factor > 0:
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.config.epsilon_factor * perimeter
            return cv2.approxPolyDP(contour, epsilon, True)
        return contour


# ============================================================================
# Method 2: Thresholded Sobel (RECOMMENDED)
# ============================================================================

class SobelThresholdContourExtractor(BaseContourExtractor):
    """
    Extract contours directly from labels using thresholded Sobel (RECOMMENDED).

    This method bypasses Canny edge detection entirely and extracts boundaries
    directly from cluster labels using Sobel operators, keeping only the
    strongest boundaries (top percentile).

    Based on: research/custom_k_means/paper_results_v2.py extract_contours_from_labels
    """

    def extract_contours(
        self,
        labels: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        return_intermediates: bool = False
    ) -> ContourResult:
        """
        Extract contours using thresholded Sobel gradients.

        Args:
            labels: Cluster labels (H, W) - REQUIRED
            edges: Binary edge map (H, W) - not used but accepted for interface
            return_intermediates: If True, include edge map and processed labels

        Returns:
            result: ContourResult with contours from strongest boundaries
        """
        if labels is None:
            raise ValueError("SobelThresholdContourExtractor requires labels")

        # 1. Smooth labels to reduce noise
        labels_smooth = median_filter(labels, size=self.config.sobel_median_filter_size)

        # 2. Compute gradients using Sobel operator
        grad_x = np.abs(sobel(labels_smooth.astype(float), axis=1))
        grad_y = np.abs(sobel(labels_smooth.astype(float), axis=0))
        boundaries = np.sqrt(grad_x**2 + grad_y**2)

        # 3. Threshold to keep only strong boundaries (top percentile)
        if boundaries.max() > 0:
            threshold = np.percentile(
                boundaries[boundaries > 0],
                self.config.sobel_threshold_percentile
            )
            edge_map = (boundaries > threshold).astype(np.uint8) * 255
        else:
            edge_map = boundaries.astype(np.uint8)

        # 4. Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges_closed = cv2.morphologyEx(
            edge_map,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=self.config.sobel_morph_iterations
        )

        # 5. Find external contours
        contours, _ = cv2.findContours(
            edges_closed,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 6. Filter and simplify
        clean_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= self.config.min_contour_area:
                simplified = self._simplify_contour(contour)
                clean_contours.append(simplified)

        return ContourResult(
            contours=clean_contours,
            method=ContourMethod.SOBEL_THRESHOLD,
            edge_map=edge_map if return_intermediates else None,
            labels_processed=labels_smooth if return_intermediates else None
        )

    def _simplify_contour(self, contour: np.ndarray) -> np.ndarray:
        """Simplify contour using Douglas-Peucker algorithm."""
        if self.config.epsilon_factor > 0:
            perimeter = cv2.arcLength(contour, True)
            epsilon = self.config.epsilon_factor * perimeter
            return cv2.approxPolyDP(contour, epsilon, True)
        return contour


# ============================================================================
# Method 3: Convex Hull
# ============================================================================

class ConvexHullContourExtractor(BaseContourExtractor):
    """
    Extract convex hull contours after merging small regions.

    This method merges small regions hierarchically, then extracts convex hulls
    for the largest remaining regions. Produces the simplest, cleanest boundaries.

    Based on: research/custom_k_means/paper_results_v2.py extract_convex_hull_contours
    """

    def extract_contours(
        self,
        labels: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        return_intermediates: bool = False
    ) -> ContourResult:
        """
        Extract convex hull contours from merged regions.

        Args:
            labels: Cluster labels (H, W) - REQUIRED
            edges: Binary edge map (H, W) - not used but accepted for interface
            return_intermediates: If True, include merged labels

        Returns:
            result: ContourResult with convex hull contours
        """
        if labels is None:
            raise ValueError("ConvexHullContourExtractor requires labels")

        labels_merged = labels.copy()

        # 1. Iteratively merge small regions
        for _ in range(self.config.hull_merge_iterations):
            labels_merged = median_filter(
                labels_merged,
                size=self.config.hull_median_filter_size
            )

            # Find region sizes
            unique_labels, counts = np.unique(labels_merged, return_counts=True)

            # Merge small regions into neighbors
            for label_id, count in zip(unique_labels, counts):
                if count < self.config.hull_min_region_size:
                    # Find largest neighbor
                    mask = (labels_merged == label_id)
                    dilated = cv2.dilate(
                        mask.astype(np.uint8),
                        np.ones((3, 3), np.uint8)
                    )
                    neighbors = labels_merged[dilated > mask.astype(np.uint8)]

                    if len(neighbors) > 0:
                        neighbor_labels, neighbor_counts = np.unique(
                            neighbors,
                            return_counts=True
                        )
                        # Exclude self
                        valid_idx = neighbor_labels != label_id
                        if np.any(valid_idx):
                            largest_neighbor = neighbor_labels[valid_idx][
                                np.argmax(neighbor_counts[valid_idx])
                            ]
                            labels_merged[mask] = largest_neighbor

        # 2. Extract convex hulls for remaining regions
        convex_contours = []
        unique_labels = np.unique(labels_merged)

        for label_id in unique_labels:
            if label_id == 0:
                continue

            mask = (labels_merged == label_id).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:
                if cv2.contourArea(contour) >= self.config.hull_min_region_size:
                    # Compute convex hull
                    hull = cv2.convexHull(contour)
                    convex_contours.append(hull)

        return ContourResult(
            contours=convex_contours,
            method=ContourMethod.CONVEX_HULL,
            labels_processed=labels_merged if return_intermediates else None
        )


# ============================================================================
# Legacy: Basic Canny + Filtering
# ============================================================================

class BasicContourExtractor(BaseContourExtractor):
    """
    Basic contour extraction from Canny edge map with filtering.

    Legacy method that applies simple filtering strategies to contours
    extracted from a Canny edge map.

    Based on: research/custom_k_means/paper_results_v2.py extract_contours_from_edges
    """

    def extract_contours(
        self,
        labels: Optional[np.ndarray] = None,
        edges: Optional[np.ndarray] = None,
        return_intermediates: bool = False
    ) -> ContourResult:
        """
        Extract contours from edge map with basic filtering.

        Args:
            labels: Cluster labels (H, W) - not used but accepted for interface
            edges: Binary edge map (H, W) - REQUIRED
            return_intermediates: If True, include hierarchy

        Returns:
            result: ContourResult with filtered contours
        """
        if edges is None:
            raise ValueError("BasicContourExtractor requires edges")

        # Find all contours
        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # First filter: remove very small contours
        filtered_contours = [
            c for c in contours
            if cv2.contourArea(c) >= self.config.min_contour_area
        ]

        # Apply filtering strategy
        if self.config.basic_filter_strategy == FilterStrategy.LARGEST:
            # Keep only the largest contour
            if filtered_contours:
                largest = max(filtered_contours, key=cv2.contourArea)
                filtered_contours = [largest]

        elif self.config.basic_filter_strategy == FilterStrategy.TOP_N:
            # Keep top N largest contours
            filtered_contours = sorted(
                filtered_contours,
                key=cv2.contourArea,
                reverse=True
            )[:self.config.basic_n_contours]

        elif self.config.basic_filter_strategy == FilterStrategy.AREA_THRESHOLD:
            # Keep contours above 10% of largest contour area
            if filtered_contours:
                max_area = cv2.contourArea(max(filtered_contours, key=cv2.contourArea))
                threshold = max_area * 0.1
                filtered_contours = [
                    c for c in filtered_contours
                    if cv2.contourArea(c) >= threshold
                ]

        # else: ALL - keep all filtered contours

        return ContourResult(
            contours=filtered_contours,
            method=ContourMethod.BASIC,
            hierarchy=hierarchy if return_intermediates else None,
            edge_map=edges
        )


# ============================================================================
# Factory Function
# ============================================================================

def create_contour_extractor(config: ContourConfig) -> BaseContourExtractor:
    """
    Factory function to create appropriate contour extractor.

    Args:
        config: Contour extraction configuration

    Returns:
        extractor: Concrete contour extractor instance

    Example:
        >>> config = ContourConfig(method=ContourMethod.SOBEL_THRESHOLD)
        >>> extractor = create_contour_extractor(config)
        >>> result = extractor.extract_contours(labels=refined_labels)
    """
    if config.method == ContourMethod.MORPHOLOGICAL:
        return MorphologicalContourExtractor(config)
    elif config.method == ContourMethod.SOBEL_THRESHOLD:
        return SobelThresholdContourExtractor(config)
    elif config.method == ContourMethod.CONVEX_HULL:
        return ConvexHullContourExtractor(config)
    elif config.method == ContourMethod.BASIC:
        return BasicContourExtractor(config)
    else:
        raise ValueError(f"Unknown contour method: {config.method}")
