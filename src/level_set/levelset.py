"""
Level Set Implementation for K-Means Clustering Refinement.

Implements the customized k-means level set algorithm:
1. Initialize level sets as signed distance functions using Fast Marching Method
2. Compute velocity field using curvature with scikit-image edge-stopping
3. Evolve level sets to refine segmentation boundaries

This module provides the complete pipeline for refining k-means segmentations
using level set evolution with edge-aware boundary smoothing.
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict, TYPE_CHECKING
import numpy as np
import cv2
import skfmm
from scipy import ndimage

if TYPE_CHECKING:
    from ..data_loader import NotebookSnapshot


@dataclass
class LevelSetConfig:
    """
    Configuration for level set initialization and evolution.

    Attributes:
        normalize: Whether to normalize distance values to [-1, 1] range
        dt: Time step for level set evolution (default: 0.5)
        n_iterations: Number of evolution iterations (default: 10)
        smoothing_param: Smoothing parameter for curvature-based evolution (default: 0.5)
    """
    normalize: bool = True
    dt: float = 0.5
    n_iterations: int = 10
    smoothing_param: float = 0.5


class LevelSet:
    """
    Level Set representation initialized from k-means clustering.

    This class implements the customized k-means level set algorithm:
    1. Initialize level sets as signed distance functions using Fast Marching Method
    2. Compute velocity field using curvature with edge-stopping
    3. Evolve level sets to refine boundaries

    The signed distance function φ(x,y) represents (paper convention):
    - Negative values: inside a cluster region
    - Positive values: outside a cluster region
    - Zero: on the cluster boundary

    Velocity computation uses curvature-based evolution with scikit-image's
    inverse gaussian gradient for edge-stopping, which is the proven method
    from morphological geodesic active contours.

    Example:
        >>> from src.kmeans import KMeans, KMeansConfig
        >>> from src.level_set import LevelSet, LevelSetConfig
        >>>
        >>> # After running k-means
        >>> kmeans = KMeans(KMeansConfig(n_clusters=5))
        >>> kmeans.fit_image(image)
        >>>
        >>> # Initialize level set
        >>> config = LevelSetConfig(normalize=True, smoothing_param=0.1)
        >>> levelset = LevelSet.from_labels(kmeans.labels, image.shape[:2], config)
        >>>
        >>> # Compute velocity and evolve
        >>> velocity = levelset.compute_velocity(image, smoothing=0.1, alpha=100.0, sigma=5.0)
        >>> evolved_phi = levelset.evolve(velocity, n_iterations=10, dt=0.5)
        >>>
        >>> # Extract final segmentation
        >>> final_labels = levelset.get_evolved_labels()
    """

    def __init__(self, config: Optional[LevelSetConfig] = None):
        """
        Initialize level set container.

        Args:
            config: Level set configuration. If None, uses defaults.
        """
        self.config = config or LevelSetConfig()
        self._phi = None  # Combined signed distance function (H, W)
        self._phi_per_cluster = None  # Per-cluster distance functions (K, H, W)
        self._labels = None  # Original cluster labels (H, W)
        self._n_clusters = 0

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        image_shape: Tuple[int, int],
        config: Optional[LevelSetConfig] = None
    ) -> 'LevelSet':
        """
        Initialize signed distance function from k-means labels.

        Uses Fast Marching Method (scikit-fmm) to compute signed distance
        functions for each cluster.

        Args:
            labels: Cluster assignments (H*W,) or (H, W)
            image_shape: Image dimensions (H, W)
            config: Level set configuration

        Returns:
            LevelSet instance with initialized phi

        Raises:
            ValueError: If labels shape doesn't match image_shape
        """
        instance = cls(config)

        # Reshape labels if needed
        if labels.ndim == 1:
            h, w = image_shape
            if labels.size != h * w:
                raise ValueError(
                    f"Labels size {labels.size} doesn't match image shape {image_shape}"
                )
            labels_2d = labels.reshape(image_shape)
        else:
            labels_2d = labels
            if labels_2d.shape != image_shape:
                raise ValueError(
                    f"Labels shape {labels_2d.shape} doesn't match image_shape {image_shape}"
                )

        instance._labels = labels_2d
        instance._n_clusters = int(labels_2d.max()) + 1

        # Compute signed distance functions
        instance._compute_distance_functions()

        return instance

    def _compute_distance_functions(self):
        """
        Compute signed distance using Fast Marching Method.

        For each cluster k, creates a binary mask and computes the signed
        distance function using skfmm.distance().
        """
        h, w = self._labels.shape
        k = self._n_clusters

        # Allocate storage for per-cluster distance functions
        self._phi_per_cluster = np.zeros((k, h, w), dtype=np.float32)

        for cluster_id in range(k):
            # Create binary mask: -1 inside cluster, +1 outside (paper convention)
            mask = np.where(self._labels == cluster_id, -1, 1)

            # Compute signed distance using Fast Marching Method
            # Returns negative distances inside, positive outside (paper convention)
            phi = skfmm.distance(mask)

            # Normalize if requested
            if self.config.normalize:
                max_abs_dist = np.abs(phi).max()
                if max_abs_dist > 0:
                    phi = phi / max_abs_dist

            self._phi_per_cluster[cluster_id] = phi

        # Compute combined distance map
        self._compute_combined_distance()

    def _compute_combined_distance(self):
        """
        Compute combined distance map (distance to nearest cluster boundary).

        For each pixel, finds the minimum absolute distance across all clusters.
        This represents the distance to the nearest cluster boundary.
        """
        # Take absolute distances to get distance magnitudes
        abs_distances = np.abs(self._phi_per_cluster)

        # Find minimum distance to any boundary
        self._phi = np.min(abs_distances, axis=0)

    @property
    def phi(self) -> np.ndarray:
        """
        Combined signed distance function.

        Returns:
            Distance map (H, W) showing distance to nearest cluster boundary
        """
        if self._phi is None:
            raise RuntimeError("Level set not initialized. Call from_labels() first.")
        return self._phi

    @property
    def phi_per_cluster(self) -> np.ndarray:
        """
        Per-cluster signed distance functions.

        Returns:
            Distance maps (K, H, W) where K is the number of clusters.
            Each map shows signed distance for that cluster (paper convention):
            - Negative: inside the cluster
            - Positive: outside the cluster
            - Zero: on the cluster boundary
        """
        if self._phi_per_cluster is None:
            raise RuntimeError("Level set not initialized. Call from_labels() first.")
        return self._phi_per_cluster

    @property
    def labels(self) -> np.ndarray:
        """
        Original cluster labels.

        Returns:
            Label array (H, W)
        """
        return self._labels

    @property
    def n_clusters(self) -> int:
        """Number of clusters."""
        return self._n_clusters

    def compute_velocity_curvature_skimage_edges(
        self,
        image: np.ndarray,
        smoothing: Optional[float] = None,
        alpha: float = 100.0,
        sigma: float = 5.0
    ) -> np.ndarray:
        """
        Compute curvature velocity with scikit-image edge-stopping.

        This method uses scikit-image's inverse_gaussian_gradient for edge detection,
        which is the proven preprocessing function used in morphological geodesic
        active contours (MorphGAC).

        The inverse gaussian gradient creates an edge-stopping function where:
        - Flat regions → values close to 1 (high velocity, boundary smooths)
        - Strong edges → values close to 0 (low velocity, boundary stops)

        Velocity formula: v = smoothing × κ × edge_function
        where κ is mean curvature and edge_function = inverse_gaussian_gradient(image)

        Args:
            image: Original RGB image (H, W, 3) for edge detection
            smoothing: Smoothing strength parameter. If None, uses config.smoothing_param
            alpha: Controls steepness of edge inversion. Larger values create sharper
                   transitions between flat and edge regions (default: 100.0)
            sigma: Standard deviation of Gaussian filter applied during preprocessing
                   (default: 5.0)

        Returns:
            velocity: Velocity field (K, H, W) for each cluster

        References:
            - scikit-image morphological_geodesic_active_contour
            - inverse_gaussian_gradient function
        """
        if smoothing is None:
            smoothing = self.config.smoothing_param

        # Import scikit-image function
        try:
            from skimage.segmentation import inverse_gaussian_gradient
        except ImportError:
            raise ImportError(
                "scikit-image is required for curvature_skimage_edges method. "
                "Install with: pip install scikit-image"
            )

        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute edge-stopping function using scikit-image
        # Returns values in [0, 1] where edges→0, flat regions→1
        edge_function = inverse_gaussian_gradient(gray, alpha=alpha, sigma=sigma)

        # Allocate velocity array
        velocity = np.zeros_like(self._phi_per_cluster, dtype=np.float32)

        # Compute curvature for each cluster
        for cluster_idx in range(self._n_clusters):
            phi = self._phi_per_cluster[cluster_idx]

            # Compute gradient using Sobel filter
            grad_y = ndimage.sobel(phi, axis=0)
            grad_x = ndimage.sobel(phi, axis=1)

            # Gradient magnitude (add epsilon to avoid division by zero)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-10

            # Normalized gradient (normal vector to level set)
            nx = grad_x / grad_mag
            ny = grad_y / grad_mag

            # Compute divergence of normalized gradient = mean curvature
            # div(n) = ∂nx/∂x + ∂ny/∂y
            nxx = ndimage.sobel(nx, axis=1)
            nyy = ndimage.sobel(ny, axis=0)
            curvature = nxx + nyy

            # Apply scikit-image edge-stopping function
            # High edge_function (flat regions) → high velocity (smoothing)
            # Low edge_function (edges) → low velocity (boundary stops)
            velocity[cluster_idx] = smoothing * curvature * edge_function

        return velocity

    def compute_velocity(
        self,
        image: np.ndarray,
        smoothing: Optional[float] = None,
        alpha: float = 100.0,
        sigma: float = 5.0,
        method: str = 'curvature_skimage',
        **kwargs
    ) -> np.ndarray:
        """
        Compute velocity field for level set evolution.

        Uses curvature-based velocity with scikit-image edge-stopping function
        (inverse gaussian gradient). This is the proven method from morphological
        geodesic active contours that smooths boundaries while stopping at edges.

        Args:
            image: Original RGB image (H, W, 3) for edge detection
            smoothing: Smoothing strength parameter. If None, uses config.smoothing_param
            alpha: Controls steepness of edge inversion (default: 100.0)
            sigma: Standard deviation of Gaussian filter (default: 5.0)
            method: Velocity method (kept for API compatibility, always uses 'curvature_skimage')
            **kwargs: Additional parameters (for future extensibility)

        Returns:
            velocity: Velocity field (K, H, W) for each cluster

        Raises:
            ValueError: If image is not provided or has wrong format

        Note:
            The 'method' parameter is kept for backward compatibility but is ignored.
            Only the curvature_skimage method is available.
        """
        if image is None:
            raise ValueError("image parameter is required")

        return self.compute_velocity_curvature_skimage_edges(
            image=image,
            smoothing=smoothing,
            alpha=alpha,
            sigma=sigma
        )

    def evolve(
        self,
        velocity: np.ndarray,
        n_iterations: Optional[int] = None,
        dt: Optional[float] = None,
        reinit_freq: int = 5
    ) -> np.ndarray:
        """
        Evolve level set using computed velocity field.

        Implements the evolution equation: dφ/dt = u(y)
        Discretized as: φ_new = φ_old + dt × velocity

        Periodically re-initializes φ as a signed distance function using
        Fast Marching Method to prevent numerical instabilities.

        Args:
            velocity: Velocity field (K, H, W) for each cluster
            n_iterations: Number of evolution steps. If None, uses config.n_iterations
            dt: Time step size. If None, uses config.dt
            reinit_freq: Re-initialize every N iterations (default: 5)

        Returns:
            evolved_phi: Evolved signed distance functions (K, H, W)

        Example:
            >>> # Compute velocity
            >>> velocity = levelset.compute_velocity(method='curvature')
            >>>
            >>> # Evolve for 20 iterations
            >>> evolved_phi = levelset.evolve(velocity, n_iterations=20, dt=0.1)
            >>>
            >>> # Extract evolved boundaries
            >>> boundaries = np.abs(evolved_phi) < 0.05
        """
        if n_iterations is None:
            n_iterations = self.config.n_iterations
        if dt is None:
            dt = self.config.dt

        # Start with current phi
        phi_evolved = self._phi_per_cluster.copy()

        # Evolution loop
        for iteration in range(n_iterations):
            # Evolution step: φ += dt × velocity
            phi_evolved = phi_evolved + dt * velocity

            # Periodic re-initialization using Fast Marching Method
            if reinit_freq > 0 and (iteration + 1) % reinit_freq == 0:
                for cluster_idx in range(self._n_clusters):
                    # Convert to binary mask (negative inside, positive outside - paper convention)
                    mask = np.where(phi_evolved[cluster_idx] < 0, -1, 1)

                    # Recompute signed distance function
                    phi_evolved[cluster_idx] = skfmm.distance(mask)

                    # Normalize if configured
                    if self.config.normalize:
                        max_abs_dist = np.abs(phi_evolved[cluster_idx]).max()
                        if max_abs_dist > 0:
                            phi_evolved[cluster_idx] = phi_evolved[cluster_idx] / max_abs_dist

        # Cache evolved phi
        self._evolved_phi = phi_evolved

        return phi_evolved

    def get_evolved_labels(self) -> np.ndarray:
        """
        Extract segmentation labels from evolved level sets.

        Each pixel is assigned to the cluster with the most negative φ value
        (i.e., the cluster it's most "inside of"). This implements a winner-takes-all
        strategy where pixels belong to the cluster whose boundary they are furthest inside.

        This label map is what gets compared to ground truth for PRI calculation in the paper.

        Returns:
            segmentation_labels: Label array (H, W) where each pixel value is the cluster ID (0 to K-1)

        Raises:
            ValueError: If evolve() has not been called yet

        Note:
            Paper convention: φ < 0 means inside a region, φ > 0 means outside.
            argmin finds the cluster with the most negative (most inside) φ value.
        """
        if self._evolved_phi is None:
            raise ValueError(
                "Must run evolve() before extracting segmentation labels. "
                "Call levelset.evolve() first to compute evolved level sets."
            )

        # Shape: (K, H, W) → argmin along axis 0 → (H, W)
        # argmin finds the index (cluster ID) with the MOST NEGATIVE phi value
        segmentation_labels = np.argmin(self._evolved_phi, axis=0)

        return segmentation_labels


def process_levelsets_batch(
    kmeans_results: 'Dict[str, KMeans]',
    original_images: 'Dict[str, np.ndarray]',
    config: 'Optional[LevelSetConfig]' = None
) -> 'Dict[str, LevelSet]':
    """
    Transforma resultados de K-Means a Signed Distance Functions para múltiples imágenes.

    Para cada resultado de K-Means, extrae las etiquetas de clusters y crea
    una representación Level Set usando Fast Marching Method. El signed distance
    function (SDF) representa la distancia de cada píxel al borde del cluster más cercano.

    Args:
        kmeans_results: Diccionario {image_id: KMeans} con objetos KMeans fitted.
        original_images: Diccionario {image_id: numpy_array} con imágenes originales.
                        Necesario para obtener las dimensiones de la imagen.
        config: Configuración de Level Set. Si None, usa valores por defecto.
                LevelSetConfig(normalize=True)

    Returns:
        Diccionario {image_id: LevelSet} con objetos LevelSet inicializados.
        Cada objeto contiene:
        - phi: Combined signed distance function (H, W)
        - phi_per_cluster: Per-cluster distance functions (K, H, W)
        - labels: Etiquetas de clusters originales (H, W)

    Raises:
        ValueError: Si kmeans_results y original_images tienen IDs diferentes.
        ValueError: Si alguna imagen no tiene el formato correcto.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, LevelSetConfig
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # Aplicar K-Means
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>>
        >>> # Transformar a Level Sets
        >>> levelset_config = LevelSetConfig(normalize=True)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>>
        >>> # Acceder a resultados
        >>> levelset_12074 = levelset_results['12074']
        >>> phi = levelset_12074.phi  # Combined distance map (H, W)
        >>> print(f"Shape: {phi.shape}, Range: [{phi.min():.3f}, {phi.max():.3f}]")
    """
    # Validar que ambos diccionarios tienen los mismos IDs
    kmeans_ids = set(kmeans_results.keys())
    image_ids = set(original_images.keys())

    if kmeans_ids != image_ids:
        missing_in_images = kmeans_ids - image_ids
        missing_in_kmeans = image_ids - kmeans_ids
        error_msg = "Los diccionarios kmeans_results y original_images deben tener los mismos IDs.\n"
        if missing_in_images:
            error_msg += f"Faltan en original_images: {missing_in_images}\n"
        if missing_in_kmeans:
            error_msg += f"Faltan en kmeans_results: {missing_in_kmeans}\n"
        raise ValueError(error_msg)

    # Usar config por defecto si no se proporciona
    if config is None:
        config = LevelSetConfig(normalize=True)

    # Procesar cada imagen
    results = {}

    for image_id in kmeans_results:
        # Obtener KMeans object e imagen original
        kmeans = kmeans_results[image_id]
        img = original_images[image_id]

        # Validar formato de imagen
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Imagen '{image_id}' debe tener shape (H, W, 3), "
                f"tiene shape {img.shape}"
            )

        # Obtener labels y shape de imagen
        labels = kmeans.labels  # Shape: (H*W,)
        image_shape = img.shape[:2]  # (H, W)

        # Crear Level Set desde labels
        levelset = LevelSet.from_labels(
            labels,
            image_shape=image_shape,
            config=config
        )

        # Guardar resultado
        results[image_id] = levelset

    return results


def process_evolution_batch(
    levelset_results: 'Dict[str, LevelSet]',
    original_images: 'Dict[str, np.ndarray]',
    snapshots: Optional['Dict[str, NotebookSnapshot]'] = None,
    alpha: float = 100.0,
    sigma: float = 5.0,
    smoothing: float = 0.1,
    dt: float = 0.5,
    n_iterations: int = 10,
    reinit_freq: int = 5
) -> 'Dict[str, np.ndarray]':
    """
    Aplica velocity field computation y level set evolution a múltiples imágenes.

    Usa el método de curvature con edge-stopping de scikit-image (inverse gaussian gradient),
    que es el método probado de morphological geodesic active contours. Este método
    suaviza las fronteras mientras las detiene en bordes fuertes de la imagen.

    El proceso para cada imagen:
    1. Compute velocity field usando 'curvature_skimage' method
    2. Evolve level sets usando la ecuación dφ/dt = u(y)
    3. Re-inicializar periódicamente usando Fast Marching Method

    **NUEVO: Uso de snapshots**
    Si se proporciona `snapshots`, los parámetros de evolución se extraen automáticamente
    del JSON guardado para cada imagen. Esto asegura que se usen los mejores parámetros
    encontrados en experimentos previos. Los parámetros extraídos son:
    - alpha → snapshot.parameters.alpha
    - sigma → snapshot.parameters.sigma
    - smoothing → snapshot.parameters.smoothing_param
    - dt → snapshot.parameters.dt
    - n_iterations → snapshot.parameters.n_iterations
    - reinit_freq → snapshot.parameters.reinit_frequency

    Args:
        levelset_results: Diccionario {image_id: LevelSet} con level sets inicializados
        original_images: Diccionario {image_id: numpy_array} con imágenes originales RGB.
                        Necesarias para edge detection en velocity computation.
        snapshots: Diccionario {image_id: NotebookSnapshot} con parámetros guardados.
                  Si se proporciona, los parámetros de evolución se extraen de aquí
                  (alpha, sigma, smoothing, dt, n_iterations, reinit_freq).
                  Si es None, usa los parámetros proporcionados como argumentos.
        alpha: Steepness of edge inversion (default: 100.0)
               Solo usado si snapshots=None.
        sigma: Gaussian blur standard deviation (default: 5.0)
               Solo usado si snapshots=None.
        smoothing: Smoothing strength parameter (default: 0.1)
                   Solo usado si snapshots=None.
        dt: Time step size (default: 0.5)
            Solo usado si snapshots=None.
        n_iterations: Number of evolution iterations (default: 10)
                     Solo usado si snapshots=None.
        reinit_freq: Re-initialization frequency (default: 5)
                    Solo usado si snapshots=None.

    Returns:
        Diccionario {image_id: evolved_phi} donde cada evolved_phi es un array (K, H, W)
        conteniendo los signed distance functions evolucionados para cada cluster.

    Raises:
        ValueError: Si levelset_results y original_images tienen IDs diferentes
        ValueError: Si snapshots se proporciona pero falta algún image_id
        ImportError: Si scikit-image no está instalado (requerido para inverse_gaussian_gradient)

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, process_evolution_batch, LevelSetConfig
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # K-Means
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>>
        >>> # Level Sets (SDF initialization)
        >>> levelset_config = LevelSetConfig(normalize=True)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>>
        >>> # Evolution CON snapshots (RECOMENDADO - usa mejores parámetros del JSON)
        >>> evolved_results = process_evolution_batch(
        ...     levelset_results,
        ...     images,
        ...     snapshots=snapshots  # ← Parámetros extraídos del JSON automáticamente
        ... )
        >>>
        >>> # Evolution SIN snapshots (usa parámetros hardcoded)
        >>> evolved_results_manual = process_evolution_batch(
        ...     levelset_results,
        ...     images,
        ...     alpha=100.0,
        ...     sigma=5.0,
        ...     smoothing=0.1,
        ...     dt=0.5,
        ...     n_iterations=10,
        ...     reinit_freq=5
        ... )
        >>>
        >>> # Access results
        >>> evolved_phi_12074 = evolved_results['12074']  # Shape: (K, H, W)
        >>> print(f"Evolved phi shape: {evolved_phi_12074.shape}")
    """
    # Validar que ambos diccionarios tienen los mismos IDs
    levelset_ids = set(levelset_results.keys())
    image_ids = set(original_images.keys())

    if levelset_ids != image_ids:
        missing_in_images = levelset_ids - image_ids
        missing_in_levelsets = image_ids - levelset_ids
        error_msg = "Los diccionarios levelset_results y original_images deben tener los mismos IDs.\n"
        if missing_in_images:
            error_msg += f"Faltan en original_images: {missing_in_images}\n"
        if missing_in_levelsets:
            error_msg += f"Faltan en levelset_results: {missing_in_levelsets}\n"
        raise ValueError(error_msg)

    # Validar snapshots si se proporciona
    if snapshots is not None:
        snapshot_ids = set(snapshots.keys())
        if levelset_ids != snapshot_ids:
            missing_in_snapshots = levelset_ids - snapshot_ids
            extra_in_snapshots = snapshot_ids - levelset_ids
            error_msg = "Los diccionarios levelset_results y snapshots deben tener los mismos IDs.\n"
            if missing_in_snapshots:
                error_msg += f"Faltan en snapshots: {missing_in_snapshots}\n"
            if extra_in_snapshots:
                error_msg += f"Extra en snapshots: {extra_in_snapshots}\n"
            raise ValueError(error_msg)

    # Procesar cada imagen
    results = {}

    for image_id in levelset_results:
        # Obtener LevelSet object e imagen original
        levelset = levelset_results[image_id]
        img = original_images[image_id]

        # Extraer parámetros del snapshot si se proporciona
        if snapshots is not None:
            snapshot = snapshots[image_id]
            params = snapshot.parameters

            # Extraer parámetros de evolución del snapshot
            alpha_img = params.alpha if params.alpha is not None else alpha
            sigma_img = params.sigma if params.sigma is not None else sigma
            smoothing_img = params.smoothing_param
            dt_img = params.dt
            n_iterations_img = params.n_iterations
            reinit_freq_img = params.reinit_frequency
        else:
            # Usar parámetros proporcionados como argumentos
            alpha_img = alpha
            sigma_img = sigma
            smoothing_img = smoothing
            dt_img = dt
            n_iterations_img = n_iterations
            reinit_freq_img = reinit_freq

        # Normalizar imagen a [0, 1] si es necesario
        if img.max() > 1.0:
            img_normalized = img.astype(np.float32) / 255.0
        else:
            img_normalized = img.astype(np.float32)

        # Paso 1: Compute velocity field usando curvature_skimage method
        velocity = levelset.compute_velocity(
            image=img_normalized,
            method='curvature_skimage',
            smoothing=smoothing_img,
            alpha=alpha_img,
            sigma=sigma_img
        )

        # Paso 2: Evolve level sets
        evolved_phi = levelset.evolve(
            velocity=velocity,
            n_iterations=n_iterations_img,
            dt=dt_img,
            reinit_freq=reinit_freq_img
        )

        # Guardar resultado
        results[image_id] = evolved_phi

    return results


def extract_evolved_labels_batch(
    levelset_results: 'Dict[str, LevelSet]'
) -> 'Dict[str, np.ndarray]':
    """
    Extrae segmentation labels de evolved level sets para múltiples imágenes.

    Después de ejecutar level set evolution, cada píxel debe asignarse a exactamente
    un cluster. Esta función extrae esas asignaciones finales usando la estrategia
    winner-takes-all: cada píxel se asigna al cluster con el φ más negativo
    (el cluster dentro del cual está más profundamente).

    Este label map es lo que se compara con ground truth para PRI calculation en el paper.

    Args:
        levelset_results: Diccionario {image_id: LevelSet} con level sets que ya
                         ejecutaron evolve(). Cada LevelSet debe tener evolved_phi
                         computado.

    Returns:
        Diccionario {image_id: evolved_labels} donde cada evolved_labels es un
        array (H, W) con valores de 0 a K-1 indicando el cluster asignado.

    Raises:
        ValueError: Si algún LevelSet no ha ejecutado evolve() aún.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, process_evolution_batch
        >>> from src.level_set import extract_evolved_labels_batch, LevelSetConfig
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # Pipeline completo
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>> levelset_config = LevelSetConfig(normalize=True)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>> evolved_results = process_evolution_batch(levelset_results, images, snapshots=snapshots)
        >>>
        >>> # Extraer labels evolucionados
        >>> evolved_labels_dict = extract_evolved_labels_batch(levelset_results)
        >>>
        >>> # Acceder a resultados
        >>> labels_12074 = evolved_labels_dict['12074']  # Shape: (H, W)
        >>> print(f"Labels shape: {labels_12074.shape}")
        >>> print(f"Clusters: {labels_12074.min()} to {labels_12074.max()}")
        >>> print(f"Unique labels: {np.unique(labels_12074)}")
    """
    # Procesar cada imagen
    results = {}

    for image_id, levelset in levelset_results.items():
        try:
            # Extraer labels usando get_evolved_labels()
            # Esta función internamente verifica que evolved_phi existe
            evolved_labels = levelset.get_evolved_labels()

            # Guardar resultado
            results[image_id] = evolved_labels

        except ValueError as e:
            # get_evolved_labels() lanza ValueError si evolve() no se ha ejecutado
            raise ValueError(
                f"Error extrayendo labels para imagen '{image_id}': {e}\n"
                f"Asegúrate de ejecutar process_evolution_batch() antes de "
                f"extract_evolved_labels_batch()."
            )

    return results
