"""
Core algorithm components for customized k-means segmentation.

Modules:
- kmeans: K-Means clustering (Stage 1)
- level_set: Level set evolution (Stage 2 - Key Innovation)
- edge_detector: Canny edge detection (Stage 3)
- contour_extractor: Contour extraction for visualization
- silhouette_extractor: Silhouette extraction and mask filling (preprocessing for PRI)
- pri_evaluator: PRI evaluation (Stage 5)
- grid_search_results: Load and manage grid search experiment results
"""

# Stage 1: K-Means Clustering
from .kmeans import (
    KMeansConfig,
    KMeansResult,
    BaseKMeans,
    SklearnKMeans,
    extract_pixels,
    extract_pixels_from_image,
    create_segmented_image
)

# Stage 2: Level Set Evolution
from .level_set import (
    LevelSetConfig,
    LevelSetResult,
    BaseLevelSet,
    FastMarchingLevelSet,
    SimpleLevelSet
)

# Stage 3: Canny Edge Detection
from .edge_detector import (
    CannyConfig,
    EdgeDetectionResult,
    BaseEdgeDetector,
    CannyEdgeDetector,
    SobelEdgeDetector
)

# Stage 4: Contour Extraction
from .contour_extractor import (
    ContourMethod,
    FilterStrategy,
    ContourConfig,
    ContourResult,
    BaseContourExtractor,
    MorphologicalContourExtractor,
    SobelThresholdContourExtractor,
    ConvexHullContourExtractor,
    BasicContourExtractor,
    create_contour_extractor
)

# Silhouette Extraction (Preprocessing for PRI)
from .silhouette_extractor import (
    SilhouetteMethod,
    SilhouetteConfig,
    SilhouetteResult,
    BaseSilhouetteExtractor,
    CannyEdgesSilhouetteExtractor,
    LabelsSilhouetteExtractor,
    ConvexHullSilhouetteExtractor,
    silhouette_to_mask,
    create_silhouette_extractor
)

# Stage 5: PRI Evaluation
from .pri_evaluator import (
    PRIConfig,
    PRIResult,
    BasePRIEvaluator,
    SklearnPRIEvaluator,
    compute_pri_for_k_range,
    find_best_k,
    create_pri_report,
)

# Grid Search Results
from .grid_search_results import (
    GridSearchMetrics,
    GridSearchParameters,
    GridSearchResult,
    GridSearchMetadata,
    GridConfig,
    SilhouetteConfig,
    GridSearchExperiment,
    GridSearchLoader
)

__all__ = [
    # K-Means
    'KMeansConfig',
    'KMeansResult',
    'BaseKMeans',
    'SklearnKMeans',
    'extract_pixels',
    'extract_pixels_from_image',
    'create_segmented_image',
    # Level Set
    'LevelSetConfig',
    'LevelSetResult',
    'BaseLevelSet',
    'FastMarchingLevelSet',
    'SimpleLevelSet',
    # Edge Detection
    'CannyConfig',
    'EdgeDetectionResult',
    'BaseEdgeDetector',
    'CannyEdgeDetector',
    'SobelEdgeDetector',
    # Contour Extraction
    'ContourMethod',
    'FilterStrategy',
    'ContourConfig',
    'ContourResult',
    'BaseContourExtractor',
    'MorphologicalContourExtractor',
    'SobelThresholdContourExtractor',
    'ConvexHullContourExtractor',
    'BasicContourExtractor',
    'create_contour_extractor',
    # Silhouette Extraction
    'SilhouetteMethod',
    'SilhouetteConfig',
    'SilhouetteResult',
    'BaseSilhouetteExtractor',
    'CannyEdgesSilhouetteExtractor',
    'LabelsSilhouetteExtractor',
    'ConvexHullSilhouetteExtractor',
    'silhouette_to_mask',
    'create_silhouette_extractor',
    # PRI Evaluation
    'PRIConfig',
    'PRIResult',
    'BasePRIEvaluator',
    'SklearnPRIEvaluator',
    'compute_pri_for_k_range',
    'find_best_k',
    'create_pri_report',
    # Grid Search Results
    'GridSearchMetrics',
    'GridSearchParameters',
    'GridSearchResult',
    'GridSearchMetadata',
    'GridConfig',
    'SilhouetteConfig',
    'GridSearchExperiment',
    'GridSearchLoader',
]
