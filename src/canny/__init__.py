"""
Canny Edge Detection and Silhouette Extraction Module.

This module provides:
1. Canny edge detection for segmentation boundaries
2. Silhouette extraction from Canny edge maps

Silhouette extraction:
- Uses Canny edge detection with morphological closing and flood fill
- Extracts clean silhouettes with filled binary masks
- Suitable for PRI evaluation against ground truth
"""

from .edge_detector import (
    CannyEdgeDetector,
    CannyConfig,
    CannyResult
)

from .silhouette_extractor import (
    SilhouetteMethod,
    SilhouetteConfig,
    SilhouetteResult,
    BaseSilhouetteExtractor,
    CannyEdgesSilhouetteExtractor,
    silhouette_to_mask,
    create_silhouette_extractor
)

__all__ = [
    # Edge detection
    'CannyEdgeDetector',
    'CannyConfig',
    'CannyResult',
    # Silhouette extraction
    'SilhouetteMethod',
    'SilhouetteConfig',
    'SilhouetteResult',
    'BaseSilhouetteExtractor',
    'CannyEdgesSilhouetteExtractor',
    'silhouette_to_mask',
    'create_silhouette_extractor'
]
