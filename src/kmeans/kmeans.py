"""
K-Means Clustering for Image Segmentation

A concise implementation that wraps sklearn's K-Means for use in the
segmentation pipeline. Handles both 2D data arrays and RGB images.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import numpy as np
from sklearn.cluster import KMeans as SklearnKMeans


@dataclass
class KMeansConfig:
    """
    Configuration for K-Means clustering.

    Attributes:
        n_clusters: Number of clusters to form
        max_iter: Maximum iterations for convergence
        random_state: Random seed for reproducibility
        init: Centroid initialization method ('k-means++' or 'random')
    """
    n_clusters: int = 5
    max_iter: int = 300
    random_state: int = 42
    init: str = 'k-means++'

    def __post_init__(self):
        """Validate configuration parameters."""
        if self.n_clusters < 2:
            raise ValueError(f"n_clusters must be >= 2, got {self.n_clusters}")
        if self.max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {self.max_iter}")
        if self.init not in ['k-means++', 'random']:
            raise ValueError(f"init must be 'k-means++' or 'random', got '{self.init}'")


class KMeans:
    """
    K-Means clustering wrapper for image segmentation.

    Provides a clean interface for clustering both 2D data arrays and RGB images.
    Uses sklearn's KMeans internally for reliability and performance.

    Example:
        >>> # For images
        >>> kmeans = KMeans(KMeansConfig(n_clusters=5))
        >>> kmeans.fit_image(image)  # image shape: (H, W, 3)
        >>> segmented = kmeans.get_segmented_image(image.shape[:2])

        >>> # For 2D data
        >>> data = image.reshape(-1, 3)  # (N, 3)
        >>> kmeans.fit(data)
        >>> labels = kmeans.labels
    """

    def __init__(self, config: Optional[KMeansConfig] = None):
        """
        Initialize K-Means clusterer.

        Args:
            config: Configuration parameters. If None, uses defaults.
        """
        self.config = config or KMeansConfig()
        self._model: Optional[SklearnKMeans] = None
        self._fitted = False

    def fit(self, data: np.ndarray) -> 'KMeans':
        """
        Fit K-Means on 2D data array.

        Args:
            data: Feature array of shape (N, features), typically (N, 3) for RGB

        Returns:
            self: For method chaining

        Raises:
            ValueError: If data is not 2D
        """
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D array (N, features), got shape {data.shape}")

        # Create and fit sklearn model
        self._model = SklearnKMeans(
            n_clusters=self.config.n_clusters,
            max_iter=self.config.max_iter,
            init=self.config.init,
            random_state=self.config.random_state,
            n_init=10  # Run multiple times with different seeds
        )

        self._model.fit(data)
        self._fitted = True

        return self

    def fit_image(self, image: np.ndarray) -> 'KMeans':
        """
        Fit K-Means on RGB image.

        Convenience method that automatically reshapes the image to 2D data.

        Args:
            image: RGB image of shape (H, W, 3), values in [0, 1] or [0, 255]

        Returns:
            self: For method chaining

        Raises:
            ValueError: If image is not 3D with 3 channels
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Image must be (H, W, 3), got shape {image.shape}")

        # Reshape to 2D: (H*W, 3)
        h, w, c = image.shape
        data = image.reshape(-1, c)

        return self.fit(data)

    @property
    def labels(self) -> np.ndarray:
        """
        Get cluster labels for fitted data.

        Returns:
            labels: Cluster assignments, shape (N,)

        Raises:
            RuntimeError: If not fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() or fit_image() before accessing labels")
        return self._model.labels_

    @property
    def centroids(self) -> np.ndarray:
        """
        Get cluster centroids (centers).

        Returns:
            centroids: Cluster centers, shape (n_clusters, features)

        Raises:
            RuntimeError: If not fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() or fit_image() before accessing centroids")
        return self._model.cluster_centers_

    @property
    def inertia(self) -> float:
        """
        Get inertia (sum of squared distances to nearest cluster center).

        Lower values indicate tighter, more compact clusters.

        Returns:
            inertia: Objective function value

        Raises:
            RuntimeError: If not fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() or fit_image() before accessing inertia")
        return self._model.inertia_

    def get_segmented_image(self, image_shape: Tuple[int, int]) -> np.ndarray:
        """
        Create segmented image with mean cluster colors.

        Each pixel is replaced by its cluster's centroid color, creating
        a posterized effect with n_clusters distinct colors.

        Args:
            image_shape: Original image dimensions (H, W)

        Returns:
            segmented: Image with K colors, shape (H, W, 3)

        Raises:
            RuntimeError: If not fitted yet
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() or fit_image() before creating segmented image")

        h, w = image_shape
        n_pixels = h * w

        if len(self.labels) != n_pixels:
            raise ValueError(
                f"Image shape {image_shape} doesn't match fitted data "
                f"({len(self.labels)} pixels)"
            )

        # Create segmented pixels by replacing each pixel with its centroid
        segmented_flat = np.zeros((n_pixels, 3))
        for k in range(self.config.n_clusters):
            mask = (self.labels == k)
            segmented_flat[mask] = self.centroids[k]

        # Reshape to image
        segmented = segmented_flat.reshape(h, w, 3)

        return segmented


def process_images_batch(
    images: Dict[str, np.ndarray],
    snapshots: Dict[str, 'NotebookSnapshot']
) -> Dict[str, 'KMeans']:
    """
    Aplica K-Means a un batch de imágenes usando parámetros guardados.

    Para cada imagen, crea una instancia de K-Means con el número de clusters
    especificado en el snapshot correspondiente, normaliza la imagen al rango
    [0, 1] y aplica el algoritmo de clustering.

    Args:
        images: Diccionario {image_id: numpy_array} donde cada array tiene
                shape (H, W, 3) con valores uint8 en rango [0, 255].
        snapshots: Diccionario {image_id: NotebookSnapshot} con parámetros
                   de configuración. Se utiliza snapshot.parameters.k_clusters
                   para determinar el número de clusters.

    Returns:
        Diccionario {image_id: KMeans} con objetos KMeans fitted.
        Cada objeto contiene:
        - centroids: Los K colores dominantes
        - labels: Asignación de cluster por píxel
        - Método get_segmented_image() para obtener imagen clusterizada

    Raises:
        ValueError: Si images y snapshots tienen IDs diferentes.
        ValueError: Si alguna imagen no tiene el formato correcto.

    Example:
        >>> from src.data_loader import DataLoader
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>> results = process_images_batch(images, snapshots)
        >>> # Acceder a resultados
        >>> kmeans_12074 = results['12074']
        >>> print(f"K = {kmeans_12074.config.n_clusters}")
        >>> segmented = kmeans_12074.get_segmented_image(images['12074'].shape[:2])
    """
    # Validar que ambos diccionarios tienen los mismos IDs
    image_ids = set(images.keys())
    snapshot_ids = set(snapshots.keys())

    if image_ids != snapshot_ids:
        missing_in_snapshots = image_ids - snapshot_ids
        missing_in_images = snapshot_ids - image_ids
        error_msg = "Los diccionarios images y snapshots deben tener los mismos IDs.\n"
        if missing_in_snapshots:
            error_msg += f"Faltan en snapshots: {missing_in_snapshots}\n"
        if missing_in_images:
            error_msg += f"Faltan en images: {missing_in_images}\n"
        raise ValueError(error_msg)

    # Procesar cada imagen
    results = {}

    for image_id in images:
        # Obtener imagen y snapshot
        img = images[image_id]
        snapshot = snapshots[image_id]

        # Validar formato de imagen
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(
                f"Imagen '{image_id}' debe tener shape (H, W, 3), "
                f"tiene shape {img.shape}"
            )

        # Normalizar imagen de [0, 255] a [0, 1]
        # K-Means funciona mejor con valores normalizados
        if img.dtype == np.uint8:
            img_normalized = img.astype(np.float64) / 255.0
        else:
            # Si ya está en float, asumir que está en [0, 1]
            img_normalized = img.astype(np.float64)

        # Obtener número de clusters del snapshot
        k_clusters = snapshot.parameters.k_clusters

        # Crear configuración de K-Means
        config = KMeansConfig(n_clusters=k_clusters)

        # Crear instancia de K-Means y aplicar a la imagen
        kmeans = KMeans(config)
        kmeans.fit_image(img_normalized)

        # Guardar resultado
        results[image_id] = kmeans

    return results
