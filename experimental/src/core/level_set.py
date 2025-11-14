"""
Level Set Evolution for Boundary Refinement

Implementation of Stage 2 of the customized k-means pipeline.

Paper Reference: Islam et al. (2021), Section III.B
Evolution Equation: dψ/dt = u⃗(y)

This module uses the Fast Marching Method for computing exact signed
distance functions, which is critical for accurate boundary refinement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt

# Try to import scikit-fmm for Fast Marching Method
try:
    import skfmm
    HAS_SKFMM = True
except ImportError:
    HAS_SKFMM = False
    print("Warning: scikit-fmm not installed. Fast Marching Method unavailable.")
    print("Install with: pip install scikit-fmm")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class LevelSetConfig:
    """
    Configuration for level set evolution.

    Parameters match those in config/default_params.yaml.
    """
    iterations: int = 10
    """Number of evolution iterations. Paper: 10 iterations."""

    dt: float = 0.5
    """Time step for evolution equation. Paper: dt = 0.5."""

    velocity_method: str = 'gradient'
    """Method for computing velocity field.
    Options: 'gradient', 'curvature', 'combined', 'research'
    - 'research': Exact implementation from research prototype (velocity = κ * (1 - |∇I|))"""

    reinit_interval: int = 5
    """Re-initialize as SDF every N iterations to maintain stability."""

    epsilon: float = 1.0
    """Smoothing parameter for regularization."""

    use_fast_marching: bool = True
    """Use Fast Marching Method (skfmm) for exact SDF computation.
    If False or skfmm unavailable, uses Euclidean distance transform."""

    edge_lambda: float = 1.0
    """Weight for edge-stopping function."""

    curvature_weight: float = 1.0
    """Weight for curvature flow term."""

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {self.iterations}")
        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.reinit_interval < 1:
            raise ValueError(f"reinit_interval must be >= 1, got {self.reinit_interval}")
        if self.velocity_method not in ['gradient', 'curvature', 'combined', 'research']:
            raise ValueError(
                f"velocity_method must be 'gradient', 'curvature', 'combined', or 'research', "
                f"got '{self.velocity_method}'"
            )


# ============================================================================
# Results
# ============================================================================

@dataclass
class LevelSetResult:
    """
    Results from level set evolution.

    Contains refined segmentation and intermediate results.
    """
    refined_labels: np.ndarray
    """Refined cluster labels after evolution. Shape: (H, W)"""

    phi: np.ndarray
    """Final level set functions. Shape: (n_clusters, H, W)"""

    n_iterations: int
    """Number of iterations performed."""

    converged: bool
    """Whether evolution converged before max iterations."""

    def get_boundaries(self) -> np.ndarray:
        """
        Extract zero-level set boundaries from phi.

        Returns:
            boundaries: Binary boundary map (H, W)
        """
        # Find where phi crosses zero (boundaries between clusters)
        boundaries = np.zeros(self.phi.shape[1:], dtype=np.uint8)

        for k in range(self.phi.shape[0]):
            # Zero-level set is the boundary
            zero_crossing = np.abs(self.phi[k]) < 0.5
            boundaries = np.logical_or(boundaries, zero_crossing)

        return boundaries.astype(np.uint8) * 255


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BaseLevelSet(ABC):
    """
    Abstract base class for level set evolution implementations.

    This interface allows for different level set methods:
    - FastMarchingLevelSet: Uses skfmm for exact SDF (production)
    - SimpleLevelSet: Uses scipy distance transform (fallback)
    - CustomLevelSet: From-scratch implementation (future)
    """

    def __init__(self, config: LevelSetConfig):
        """
        Initialize level set evolution.

        Args:
            config: Configuration parameters
        """
        self.config = config
        self._phi: Optional[np.ndarray] = None
        self._n_clusters: Optional[int] = None

    @abstractmethod
    def evolve(
        self,
        labels: np.ndarray,
        image_shape: Tuple[int, int],
        image: np.ndarray
    ) -> LevelSetResult:
        """
        Perform level set evolution on cluster labels.

        Args:
            labels: Initial cluster labels from K-means, shape (H*W,)
            image_shape: (H, W) dimensions of original image
            image: Original RGB image for edge-stopping, shape (H, W, 3)

        Returns:
            result: LevelSetResult with refined segmentation
        """
        pass

    @abstractmethod
    def compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute signed distance function from binary mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            sdf: Signed distance function (H, W)
        """
        pass

    @abstractmethod
    def reinitialize_sdf(self, phi: np.ndarray) -> np.ndarray:
        """
        Re-initialize level set function as signed distance function.

        Args:
            phi: Current level set functions (n_clusters, H, W)

        Returns:
            phi_reinit: Re-initialized SDF (n_clusters, H, W)
        """
        pass


# ============================================================================
# Fast Marching Implementation
# ============================================================================

class FastMarchingLevelSet(BaseLevelSet):
    """
    Level set evolution using Fast Marching Method.

    This is the primary implementation that uses scikit-fmm for computing
    exact signed distance functions by solving the Eikonal equation.

    Based on: research/custom_k_means/paper_results_v2.py
              research/intuition/custom_kmeans/customized_kmeans.py
    """

    def __init__(self, config: LevelSetConfig):
        """
        Initialize Fast Marching level set evolution.

        Args:
            config: Level set configuration

        Raises:
            ImportError: If skfmm not installed and use_fast_marching=True
        """
        super().__init__(config)

        if config.use_fast_marching and not HAS_SKFMM:
            raise ImportError(
                "scikit-fmm is required for Fast Marching Method. "
                "Install with: pip install scikit-fmm"
            )

    def evolve(
        self,
        labels: np.ndarray,
        image_shape: Tuple[int, int],
        image: np.ndarray
    ) -> LevelSetResult:
        """
        Perform level set evolution on cluster labels.

        Implements the paper's equation: dψ/dt = u⃗(y)

        Args:
            labels: Cluster labels from K-means, shape (H*W,)
            image_shape: (H, W) dimensions
            image: Original RGB image, shape (H, W, 3), values in [0, 1]

        Returns:
            result: LevelSetResult with refined labels
        """
        h, w = image_shape
        labels_2d = labels.reshape(h, w)

        # Get number of clusters
        self._n_clusters = labels_2d.max() + 1

        # Store image for research velocity method (needs raw gradients each iteration)
        self._image = image

        # Initialize level set functions as signed distance functions
        phi = self._initialize_phi(labels_2d)

        # Compute edge-stopping function once (doesn't change)
        edge_function = self._compute_edge_stopping_function(image)

        # Evolution loop
        converged = False
        for iteration in range(self.config.iterations):
            # Compute velocity field
            velocity = self._compute_velocity_field(phi, edge_function)

            # Update phi using forward Euler: ψ^(n+1) = ψ^n + dt * u⃗
            phi = phi + self.config.dt * velocity

            # Periodically re-initialize as signed distance function
            if (iteration + 1) % self.config.reinit_interval == 0:
                phi = self.reinitialize_sdf(phi)

        # Extract refined labels from final phi
        refined_labels = self._extract_labels(phi)

        return LevelSetResult(
            refined_labels=refined_labels,
            phi=phi,
            n_iterations=self.config.iterations,
            converged=converged
        )

    def compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """
        Compute exact signed distance function using Fast Marching Method.

        Solves the Eikonal equation: |∇ψ| = 1

        Args:
            mask: Binary mask (H, W), 1=inside, 0=outside

        Returns:
            sdf: Signed distance function (H, W)
                 Negative inside, positive outside
        """
        if self.config.use_fast_marching and HAS_SKFMM:
            # Convert to level set representation
            # Inside (mask=1) → -1, Outside (mask=0) → +1
            mask_bool = (mask > 0).astype(np.float64)
            phi_init = np.where(mask_bool > 0, -1.0, 1.0)

            # Apply Fast Marching Method to get exact signed distance
            sdf = skfmm.distance(phi_init)
            return sdf
        else:
            # Fallback to Euclidean distance transform approximation
            mask_bool = (mask > 0).astype(np.uint8)
            pos_dist = distance_transform_edt(mask_bool)
            neg_dist = distance_transform_edt(1 - mask_bool)
            sdf = pos_dist - neg_dist
            return sdf

    def reinitialize_sdf(self, phi: np.ndarray) -> np.ndarray:
        """
        Re-initialize level set functions as signed distance functions.

        This maintains numerical stability during evolution by ensuring
        that |∇ψ| ≈ 1 (property of signed distance functions).

        Args:
            phi: Current level set functions (n_clusters, H, W)

        Returns:
            phi_reinit: Re-initialized SDFs (n_clusters, H, W)
        """
        n_clusters, h, w = phi.shape
        phi_reinit = np.zeros_like(phi)

        for k in range(n_clusters):
            # Create binary mask from current phi
            mask = (phi[k] > 0).astype(np.uint8)

            # Re-compute as signed distance function
            phi_reinit[k] = self.compute_sdf(mask)

        return phi_reinit

    def _initialize_phi(self, labels_2d: np.ndarray) -> np.ndarray:
        """
        Initialize level set functions from cluster labels.

        For each cluster k, create a signed distance function where:
        - ψ_k < 0 inside cluster k
        - ψ_k > 0 outside cluster k
        - ψ_k = 0 at cluster boundary

        Args:
            labels_2d: Cluster labels (H, W)

        Returns:
            phi: Initial level set functions (n_clusters, H, W)
        """
        h, w = labels_2d.shape
        phi = np.zeros((self._n_clusters, h, w))

        for k in range(self._n_clusters):
            mask = (labels_2d == k).astype(np.uint8)
            phi[k] = self.compute_sdf(mask)

        return phi

    def _compute_edge_stopping_function(self, image: np.ndarray) -> np.ndarray:
        """
        Compute edge-stopping function g(|∇I|).

        This function is small at strong edges and large in smooth regions,
        preventing the level set from crossing object boundaries.

        Formula: g(|∇I|) = 1 / (1 + λ * |∇I|²)

        Args:
            image: RGB image (H, W, 3), values in [0, 1]

        Returns:
            g: Edge-stopping function (H, W)
        """
        # Convert to grayscale
        if image.max() <= 1.0:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        gray = gray.astype(np.float64) / 255.0

        # Compute image gradient magnitude
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize gradient
        grad_mag = grad_mag / (grad_mag.max() + 1e-10)

        # Edge-stopping function
        g = 1.0 / (1.0 + self.config.edge_lambda * grad_mag**2)

        return g

    def _compute_raw_edge_strength(self) -> np.ndarray:
        """
        Compute raw edge strength exactly as in research code.

        This matches: research/intuition/custom_kmeans/customized_kmeans.py lines 188-192

        Formula: edge_strength = |∇I| / max(|∇I|)
        (No geodesic active contours transformation, just normalized gradient magnitude)

        CRITICAL: Research code computes Sobel on uint8 values [0,255], NOT normalized [0,1]!

        Returns:
            edge_strength: Raw normalized gradient magnitude (H, W)
        """
        image = self._image

        # Convert to grayscale (EXACTLY like research - no normalization!)
        if image.max() <= 1.0:
            gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # DO NOT NORMALIZE - work on uint8 like research code!
        # Research line 189-190: Sobel directly on uint8 gray image

        # Compute gradient magnitude (EXACTLY like research code - on uint8)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_strength = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize by max (research formula)
        edge_strength = edge_strength / (edge_strength.max() + 1e-10)

        return edge_strength

    def _compute_velocity_field(
        self,
        phi: np.ndarray,
        edge_function: np.ndarray
    ) -> np.ndarray:
        """
        Compute velocity field u⃗(y) for level set evolution.

        Combines:
        - Curvature flow: κ (smooths boundaries)
        - Edge-stopping: g(|∇I|) (preserves edges)

        Velocity: u⃗(y) = κ · g(|∇I|)

        Args:
            phi: Level set functions (n_clusters, H, W)
            edge_function: Edge-stopping function (H, W)

        Returns:
            velocity: Velocity field (n_clusters, H, W)
        """
        n_clusters, h, w = phi.shape
        velocity = np.zeros_like(phi)

        for k in range(n_clusters):
            if self.config.velocity_method == 'curvature':
                # Pure curvature flow
                curvature = self._compute_curvature(phi[k])
                velocity[k] = self.config.curvature_weight * curvature * edge_function

            elif self.config.velocity_method == 'gradient':
                # Gradient-based flow (simpler, faster)
                curvature = self._compute_curvature(phi[k])
                velocity[k] = curvature * (1.0 - edge_function)

            elif self.config.velocity_method == 'combined':
                # Combined curvature and edge-driven flow
                curvature = self._compute_curvature(phi[k])
                velocity[k] = curvature * edge_function

            elif self.config.velocity_method == 'research':
                # EXACT RESEARCH IMPLEMENTATION (matches research/custom_kmeans)
                # Uses raw normalized gradient magnitude, not geodesic formula
                # This is the formula that achieved PRI=0.7674 in the paper
                curvature = self._compute_curvature(phi[k])
                # Compute edge strength directly from image (no geodesic transform)
                edge_strength = self._compute_raw_edge_strength()
                velocity[k] = curvature * (1.0 - edge_strength)

        return velocity

    def _compute_curvature(self, phi: np.ndarray) -> np.ndarray:
        """
        Compute mean curvature of level set function.

        Mean curvature κ measures how much the level set curves.
        Large positive κ → concave (curves inward)
        Large negative κ → convex (curves outward)

        Formula:
        κ = (φ_xx * φ_y² - 2φ_xy * φ_x * φ_y + φ_yy * φ_x²) / (φ_x² + φ_y² + ε)^(3/2)

        Args:
            phi: Level set function (H, W)

        Returns:
            curvature: Mean curvature (H, W)
        """
        # Compute first derivatives
        phi_y, phi_x = np.gradient(phi)

        # Compute second derivatives
        phi_xx = np.gradient(phi_x, axis=1)
        phi_yy = np.gradient(phi_y, axis=0)
        phi_xy = np.gradient(phi_x, axis=0)

        # Compute curvature
        eps = self.config.epsilon * 1e-10
        numerator = phi_xx * phi_y**2 - 2 * phi_xy * phi_x * phi_y + phi_yy * phi_x**2
        denominator = (phi_x**2 + phi_y**2 + eps)**(1.5)

        curvature = numerator / denominator

        return curvature

    def _extract_labels(self, phi: np.ndarray) -> np.ndarray:
        """
        Extract cluster labels from level set functions.

        Each pixel is assigned to the cluster with largest (most positive) phi value.

        Args:
            phi: Level set functions (n_clusters, H, W)

        Returns:
            labels: Cluster assignments (H, W)
        """
        labels = np.argmax(phi, axis=0)
        return labels


# ============================================================================
# Fallback Implementation (No skfmm)
# ============================================================================

class SimpleLevelSet(BaseLevelSet):
    """
    Simple level set evolution using scipy distance transforms.

    Fallback implementation when scikit-fmm is not available.
    Uses Euclidean distance transform as approximation to Fast Marching.

    Note: Less accurate than FastMarchingLevelSet but doesn't require skfmm.
    """

    def evolve(
        self,
        labels: np.ndarray,
        image_shape: Tuple[int, int],
        image: np.ndarray
    ) -> LevelSetResult:
        """Similar to FastMarchingLevelSet but uses scipy distance_transform_edt."""
        # Temporarily set use_fast_marching=False
        original_setting = self.config.use_fast_marching
        self.config.use_fast_marching = False

        # Use same logic as FastMarchingLevelSet
        fmm = FastMarchingLevelSet(self.config)
        result = fmm.evolve(labels, image_shape, image)

        # Restore setting
        self.config.use_fast_marching = original_setting

        return result

    def compute_sdf(self, mask: np.ndarray) -> np.ndarray:
        """Compute approximate SDF using Euclidean distance transform."""
        mask_bool = (mask > 0).astype(np.uint8)
        pos_dist = distance_transform_edt(mask_bool)
        neg_dist = distance_transform_edt(1 - mask_bool)
        sdf = pos_dist - neg_dist
        return sdf

    def reinitialize_sdf(self, phi: np.ndarray) -> np.ndarray:
        """Re-initialize using distance transforms."""
        n_clusters, h, w = phi.shape
        phi_reinit = np.zeros_like(phi)

        for k in range(n_clusters):
            mask = (phi[k] > 0).astype(np.uint8)
            phi_reinit[k] = self.compute_sdf(mask)

        return phi_reinit
