"""
K-Means Clustering for Image Segmentation

Implementation of Stage 1 of the customized k-means pipeline.

Paper Reference: Islam et al. (2021), Section III.A
Objective Function: J(V) = Σ Σ ||xn - vl||²
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeansAlgorithm


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class KMeansConfig:
    """
    Configuration for K-means clustering.

    Parameters match those in config/default_params.yaml.
    """
    n_clusters: int = 5
    """Number of clusters (k parameter). Paper: varies [3-7] per image."""

    max_iter: int = 300
    """Maximum number of iterations for convergence."""

    tol: float = 1e-4
    """Convergence tolerance for centroid movement."""

    n_init: int = 10
    """Number of times k-means runs with different centroid seeds.
    Best result (lowest inertia) is kept."""

    random_state: Optional[int] = 42
    """Random seed for reproducibility."""

    init_method: str = 'k-means++'
    """Centroid initialization method.
    Options: 'k-means++' (smart), 'random' (basic)."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.n_init < 1:
            raise ValueError(f"n_init must be >= 1, got {self.n_init}")
        if self.init_method not in ['k-means++', 'random']:
            raise ValueError(
                f"init_method must be 'k-means++' or 'random', got {self.init_method}"
            )


# ============================================================================
# Results
# ============================================================================

@dataclass
class KMeansResult:
    """
    Results from k-means clustering.

    Contains all relevant outputs for the segmentation pipeline.
    """
    labels: np.ndarray
    """Cluster labels for each pixel. Shape: (H*W,)"""

    centroids: np.ndarray
    """RGB values of cluster centroids. Shape: (n_clusters, 3)"""

    inertia: float
    """Objective function J(V) = Σ Σ ||xn - vl||².
    Lower values indicate tighter clusters."""

    n_iter: int
    """Number of iterations until convergence."""

    converged: bool
    """Whether the algorithm converged within max_iter."""

    def reshape_labels(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Reshape flat labels to 2D image shape.

        Args:
            shape: (H, W) image dimensions

        Returns:
            labels_2d: (H, W) cluster labels
        """
        return self.labels.reshape(shape)


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BaseKMeans(ABC):
    """
    Abstract base class for k-means clustering implementations.

    This interface allows for different k-means implementations:
    - SklearnKMeans: Wrapper around scikit-learn (production-ready)
    - CustomKMeans: From-scratch implementation (future)
    - MiniBatchKMeans: Fast approximation for large images (future)

    All implementations must:
    1. Accept pixel RGB values as input
    2. Return KMeansResult with labels, centroids, and metrics
    3. Support fit/predict/fit_predict interface
    """

    def __init__(self, config: KMeansConfig):
        """
        Initialize k-means clusterer.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self._fitted = False
        self._centroids: Optional[np.ndarray] = None

    @abstractmethod
    def fit(self, pixels: np.ndarray) -> 'BaseKMeans':
        """
        Fit k-means on pixel RGB values.

        Args:
            pixels: RGB values, shape (N, 3) where N = H*W

        Returns:
            self (for method chaining)
        """
        pass

    @abstractmethod
    def predict(self, pixels: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels for pixels using fitted centroids.

        Args:
            pixels: RGB values, shape (N, 3)

        Returns:
            labels: Cluster assignments, shape (N,)

        Raises:
            RuntimeError: If called before fit()
        """
        pass

    @abstractmethod
    def fit_predict(self, pixels: np.ndarray) -> KMeansResult:
        """
        Fit k-means and return complete results.

        Args:
            pixels: RGB values, shape (N, 3) where N = H*W

        Returns:
            result: KMeansResult with labels, centroids, inertia, etc.
        """
        pass

    def compute_inertia(
        self,
        pixels: np.ndarray,
        labels: np.ndarray,
        centroids: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute k-means objective function J(V).

        Paper equation (Section III.A):
        J(V) = Σ(n=1 to c) Σ(k=1 to cn) ||xn - vl||²

        Args:
            pixels: RGB values, shape (N, 3)
            labels: Cluster assignments, shape (N,)
            centroids: Cluster centers, shape (K, 3).
                      If None, uses fitted centroids.

        Returns:
            inertia: Sum of squared distances to nearest centroid
        """
        if centroids is None:
            if self._centroids is None:
                raise RuntimeError("Must fit() before computing inertia")
            centroids = self._centroids

        inertia = 0.0
        for k in range(len(centroids)):
            # Get pixels assigned to cluster k
            cluster_pixels = pixels[labels == k]

            if len(cluster_pixels) > 0:
                # Compute squared distances to centroid
                distances_sq = np.sum(
                    (cluster_pixels - centroids[k]) ** 2,
                    axis=1
                )
                inertia += np.sum(distances_sq)

        return inertia


# ============================================================================
# Sklearn Implementation
# ============================================================================

class SklearnKMeans(BaseKMeans):
    """
    K-means clustering using scikit-learn.

    This is a wrapper around sklearn.cluster.KMeans that:
    - Provides a consistent interface (BaseKMeans)
    - Returns rich results (KMeansResult)
    - Matches the existing research implementation

    Based on: research/intuition/custom_kmeans/customized_kmeans.py
    """

    def __init__(self, config: KMeansConfig):
        """
        Initialize sklearn-based k-means.

        Args:
            config: K-means configuration
        """
        super().__init__(config)
        self._sklearn_kmeans: Optional[SklearnKMeansAlgorithm] = None

    def fit(self, pixels: np.ndarray) -> 'SklearnKMeans':
        """
        Fit k-means using sklearn.

        Args:
            pixels: RGB pixel values, shape (N, 3)

        Returns:
            self
        """
        # Validate input
        if pixels.ndim != 2 or pixels.shape[1] != 3:
            raise ValueError(
                f"pixels must have shape (N, 3), got {pixels.shape}"
            )

        # Create sklearn k-means instance
        self._sklearn_kmeans = SklearnKMeansAlgorithm(
            n_clusters=self.config.n_clusters,
            max_iter=self.config.max_iter,
            tol=self.config.tol,
            n_init=self.config.n_init,
            init=self.config.init_method,
            random_state=self.config.random_state
        )

        # Fit on pixel data
        self._sklearn_kmeans.fit(pixels)

        # Store centroids
        self._centroids = self._sklearn_kmeans.cluster_centers_
        self._fitted = True

        return self

    def predict(self, pixels: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels using fitted k-means.

        Args:
            pixels: RGB pixel values, shape (N, 3)

        Returns:
            labels: Cluster assignments, shape (N,)
        """
        if not self._fitted or self._sklearn_kmeans is None:
            raise RuntimeError(
                "Must call fit() before predict(). "
                "Or use fit_predict() to do both."
            )

        return self._sklearn_kmeans.predict(pixels)

    def fit_predict(self, pixels: np.ndarray) -> KMeansResult:
        """
        Fit k-means and return complete results.

        Args:
            pixels: RGB pixel values, shape (N, 3) where N = H*W

        Returns:
            result: KMeansResult with all clustering information

        Example:
            >>> from src.core.kmeans import SklearnKMeans, KMeansConfig
            >>> import numpy as np
            >>>
            >>> # Load image and extract RGB features
            >>> image = plt.imread('image.jpg') / 255.0  # Normalize to [0, 1]
            >>> h, w = image.shape[:2]
            >>> pixels = image.reshape(-1, 3)  # (H*W, 3)
            >>>
            >>> # Configure and run k-means
            >>> config = KMeansConfig(n_clusters=5)
            >>> kmeans = SklearnKMeans(config)
            >>> result = kmeans.fit_predict(pixels)
            >>>
            >>> # Use results
            >>> labels_2d = result.reshape_labels((h, w))
            >>> print(f"Inertia: {result.inertia:.2f}")
            >>> print(f"Converged: {result.converged}")
        """
        # Fit the model
        self.fit(pixels)

        # Get predictions
        labels = self._sklearn_kmeans.predict(pixels)

        # Extract results
        centroids = self._sklearn_kmeans.cluster_centers_
        inertia = self._sklearn_kmeans.inertia_
        n_iter = self._sklearn_kmeans.n_iter_

        # Check convergence
        # sklearn sets n_iter_ = max_iter if it didn't converge
        converged = (n_iter < self.config.max_iter)

        return KMeansResult(
            labels=labels,
            centroids=centroids,
            inertia=inertia,
            n_iter=n_iter,
            converged=converged
        )


# ============================================================================
# Future Implementations (Placeholders)
# ============================================================================

# TODO: Implement CustomKMeans for from-scratch k-means
# class CustomKMeans(BaseKMeans):
#     """
#     Custom k-means implementation following paper's equation directly.
#
#     Algorithm:
#     1. Initialize k random centroids
#     2. Assign each pixel to nearest centroid
#     3. Update centroids as mean of assigned pixels
#     4. Repeat until convergence or max_iter
#
#     Advantages:
#     - Full control over algorithm
#     - Educational value
#     - Can add custom features
#
#     Disadvantages:
#     - Slower than sklearn (no C optimizations)
#     - Needs thorough testing
#     """
#     pass


# TODO: Implement MiniBatchKMeans for faster clustering on large images
# class MiniBatchKMeans(BaseKMeans):
#     """
#     Mini-batch k-means for fast clustering on large images.
#
#     Referenced in: research/intuition/optimization_summary.md
#
#     Instead of processing all pixels each iteration, uses random batches:
#     - 10-100x faster than standard k-means
#     - 95-99% similar results
#     - Ideal for high-resolution images
#
#     Algorithm:
#     1. Initialize k centroids
#     2. For each iteration:
#        a. Sample random batch of pixels
#        b. Assign batch to nearest centroids
#        c. Update centroids incrementally
#     3. Final assignment of all pixels
#
#     Based on: sklearn.cluster.MiniBatchKMeans
#     """
#     pass


# ============================================================================
# Helper Functions
# ============================================================================

def extract_pixels(image: np.ndarray) -> np.ndarray:
    """
    Extract RGB pixel features from image.

    Converts 3D image array to 2D feature matrix for k-means.

    Args:
        image: RGB image, shape (H, W, 3), values in [0, 1]

    Returns:
        pixels: RGB features, shape (H*W, 3)

    Example:
        >>> image = plt.imread('image.jpg') / 255.0
        >>> pixels = extract_pixels(image)
        >>> print(pixels.shape)  # (H*W, 3)
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"image must have shape (H, W, 3), got {image.shape}"
        )

    h, w, c = image.shape
    pixels = image.reshape(-1, c)  # (H*W, 3)

    return pixels


# Alias for backward compatibility
extract_pixels_from_image = extract_pixels


def create_segmented_image(
    result: KMeansResult,
    image_shape: Tuple[int, int]
) -> np.ndarray:
    """
    Create segmented image with mean color per cluster.

    Each pixel is replaced by its cluster's centroid color.

    Args:
        result: K-means clustering result
        image_shape: (H, W) dimensions of the original image

    Returns:
        segmented: Image with K colors, shape (H, W, 3)

    Example:
        >>> pixels = extract_pixels(image)
        >>> result = kmeans.fit_predict(pixels)
        >>> segmented = create_segmented_image(result, image.shape[:2])
        >>> plt.imshow(segmented)
    """
    h, w = image_shape

    # Create output image
    segmented_pixels = np.zeros((h * w, 3))

    # Replace each pixel with its centroid color
    for k in range(len(result.centroids)):
        mask = (result.labels == k)
        segmented_pixels[mask] = result.centroids[k]

    segmented = segmented_pixels.reshape(h, w, 3)

    return segmented
