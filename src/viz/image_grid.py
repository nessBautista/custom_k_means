"""
Utilidades de visualización para grids de imágenes.

Proporciona funciones para mostrar múltiples imágenes en formato de cuadrícula
y cargar imágenes del dataset BSD500 para análisis de K-Means.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Dict, Optional


def plot_image_grid(
    images: List[np.ndarray],
    titles: List[str],
    figsize: Tuple[int, int] = (18, 6)
) -> plt.Figure:
    """
    Crea un grid de imágenes en una sola fila.

    Muestra múltiples imágenes lado a lado en una cuadrícula horizontal.
    Útil para comparar imágenes del mismo dataset o diferentes procesamiento.

    Args:
        images: Lista de arrays de imágenes RGB con shape (H, W, 3).
                Todas las imágenes pueden tener dimensiones diferentes.
        titles: Lista de títulos para cada imagen. Debe tener la misma
                longitud que images.
        figsize: Tamaño de la figura (ancho, alto) en pulgadas.
                 Por defecto (18, 6) para 3 imágenes.

    Returns:
        fig: Figura de matplotlib con el grid de imágenes configurado.
             La figura incluye todas las imágenes sin ejes visibles.

    Raises:
        ValueError: Si la longitud de images y titles no coincide.

    Example:
        >>> import numpy as np
        >>> img1 = np.random.rand(100, 100, 3)
        >>> img2 = np.random.rand(150, 150, 3)
        >>> img3 = np.random.rand(120, 120, 3)
        >>> images = [img1, img2, img3]
        >>> titles = ['Image 1', 'Image 2', 'Image 3']
        >>> fig = plot_image_grid(images, titles)
        >>> plt.show()
    """
    # Validación de entrada
    if len(images) != len(titles):
        raise ValueError(
            f"El número de imágenes ({len(images)}) debe coincidir "
            f"con el número de títulos ({len(titles)})"
        )

    n_images = len(images)

    # Crear figura con subplots en una fila
    fig, axes = plt.subplots(1, n_images, figsize=figsize)

    # Si solo hay una imagen, axes no es un array
    if n_images == 1:
        axes = [axes]

    # Iterar sobre cada imagen y subplot
    for idx, (img, title) in enumerate(zip(images, titles)):
        # Mostrar imagen
        axes[idx].imshow(img)

        # Configurar subplot
        axes[idx].axis('off')  # Ocultar ejes
        axes[idx].set_title(title, fontsize=14, pad=10)

    # Ajustar layout para evitar solapamiento
    plt.tight_layout()

    return fig


def load_and_display_bsd_images(
    images: Optional[Dict[str, np.ndarray]] = None,
    image_ids: Optional[List[str]] = None,
    data_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (18, 6)
) -> Tuple[plt.Figure, Dict[str, np.ndarray]]:
    """
    Visualiza imágenes BSD500 en un grid horizontal.

    **NUEVO**: Ahora acepta imágenes PRE-CARGADAS vía el parámetro `images`.
    Esto permite usar DataLoader para cargar las imágenes separadamente.

    Esta función puede:
    1. Mostrar imágenes PRE-CARGADAS (nuevo flujo recomendado con DataLoader)
    2. Cargar Y mostrar imágenes desde disco (comportamiento legacy)

    **Flujo recomendado** (usando DataLoader):
    ```python
    from src.data_loader import DataLoader
    loader = DataLoader()
    images = loader.load_all_images()
    fig, _ = load_and_display_bsd_images(images=images)
    ```

    **Flujo legacy** (carga automática desde disco):
    ```python
    fig, images = load_and_display_bsd_images()  # Carga automáticamente
    ```

    Args:
        images: Diccionario {image_id: numpy_array} con imágenes PRE-CARGADAS.
                Si se proporciona, las imágenes se usan directamente SIN cargar desde disco.
                Si es None, carga automáticamente desde disco (comportamiento legacy).
        image_ids: Lista de IDs de imágenes a mostrar.
                   Por defecto: ['12074', '42044', '100075']
                   Si `images` se proporciona, estos IDs deben existir en ese dict.
                   Si `images` es None, estas imágenes se cargarán desde disco.
        data_path: Ruta al directorio de datos (solo usado si images=None).
                   Si None, se infiere automáticamente buscando KmeansV3/src/data.
        figsize: Tamaño de la figura matplotlib (ancho, alto) en pulgadas.

    Returns:
        fig: Figura de matplotlib con las imágenes mostradas en grid horizontal.
        images: Diccionario {image_id: numpy_array} con las imágenes.
                Si se pasó `images`, retorna el mismo dict.
                Si se cargó desde disco, retorna las imágenes cargadas.

    Raises:
        FileNotFoundError: Si images=None y no se encuentra el directorio o alguna imagen.
        KeyError: Si images se proporciona pero falta algún image_id especificado.

    Example:
        >>> # NUEVO: Usar con DataLoader (recomendado)
        >>> from src.data_loader import DataLoader
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> fig, _ = load_and_display_bsd_images(images=images)
        >>> fig  # Mostrar en Marimo

        >>> # LEGACY: Carga automática desde disco
        >>> fig, images = load_and_display_bsd_images()
        >>> print(images.keys())  # dict_keys(['12074', '42044', '100075'])

        >>> # Mostrar solo ciertas imágenes de un dict pre-cargado
        >>> loader = DataLoader()
        >>> all_images = loader.load_all_images()
        >>> fig, _ = load_and_display_bsd_images(
        ...     images=all_images,
        ...     image_ids=['12074', '42044']  # Solo mostrar 2
        ... )
    """
    # Valores por defecto para image_ids
    if image_ids is None:
        image_ids = ['12074', '42044', '100075']

    # ========== NUEVO: Si images se proporciona, usar directamente ==========
    if images is not None:
        # Validar que todos los image_ids existen en el dict
        missing_ids = [img_id for img_id in image_ids if img_id not in images]
        if missing_ids:
            raise KeyError(
                f"Las siguientes imágenes no están en el diccionario provisto: {missing_ids}\n"
                f"IDs solicitados: {image_ids}\n"
                f"IDs disponibles en dict: {list(images.keys())}"
            )

        # Crear visualización directamente con las imágenes provistas
        image_list = [images[img_id] for img_id in image_ids]
        titles = [f'BSD500 #{img_id}' for img_id in image_ids]

        fig = plot_image_grid(image_list, titles, figsize=figsize)

        return fig, images

    # ========== LEGACY: Cargar desde disco (comportamiento original) ==========

    # ========== Paso 1: Determinar ruta al directorio de datos ==========
    if data_path is None:
        # Buscar KmeansV3/src/data desde el directorio actual
        current = Path.cwd()

        # Caso 1: Estamos en KmeansV3/ (directorio raíz del proyecto)
        if current.name == 'KmeansV3':
            data_path = current / 'src' / 'data'

        # Caso 2: KmeansV3/ es subdirectorio del directorio actual
        elif (current / 'KmeansV3').exists():
            data_path = current / 'KmeansV3' / 'src' / 'data'

        # Caso 3: Buscar hacia arriba en el árbol de directorios
        else:
            data_path = None
            for parent in current.parents:
                candidate = parent / 'KmeansV3' / 'src' / 'data'
                if candidate.exists():
                    data_path = candidate
                    break

        # Validar que se encontró el directorio
        if data_path is None or not data_path.exists():
            raise FileNotFoundError(
                f"No se pudo encontrar el directorio de datos KmeansV3/src/data.\n"
                f"Búsqueda iniciada desde: {Path.cwd()}\n"
                f"Casos explorados:\n"
                f"  1. {Path.cwd()}/src/data/ (si cwd es KmeansV3)\n"
                f"  2. {Path.cwd()}/KmeansV3/src/data/\n"
                f"  3. Directorios parent: {list(Path.cwd().parents)}\n"
                f"Solución: Ejecuta desde KmeansV3/ o especifica data_path manualmente."
            )

    # ========== Paso 2: Cargar imágenes desde disco ==========
    images = {}

    for img_id in image_ids:
        img_path = data_path / f'{img_id}.jpg'

        # Validar que la imagen existe
        if not img_path.exists():
            raise FileNotFoundError(
                f"Imagen '{img_id}.jpg' no encontrada en {data_path}\n"
                f"Ruta completa buscada: {img_path}\n"
                f"Imágenes disponibles en {data_path}:\n"
                f"  {list(data_path.glob('*.jpg'))}"
            )

        # Cargar imagen usando PIL
        # PIL.Image.open() retorna un objeto PIL Image
        img = Image.open(img_path)

        # Convertir a numpy array (H, W, 3) con valores uint8 [0-255]
        # Si la imagen es RGB, se mantiene como (H, W, 3)
        # Si es grayscale, PIL la convierte a RGB automáticamente
        images[img_id] = np.array(img)

    # ========== Paso 3: Crear visualización en grid ==========
    # Preparar listas para plot_image_grid()
    image_list = [images[img_id] for img_id in image_ids]
    titles = [f'BSD500 #{img_id}' for img_id in image_ids]

    # Crear figura usando la función de grid
    fig = plot_image_grid(image_list, titles, figsize=figsize)

    return fig, images


def plot_kmeans_results(
    original_images: Dict[str, np.ndarray],
    kmeans_results: Dict[str, 'KMeans'],
    image_ids: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 18)
) -> plt.Figure:
    """
    Visualiza resultados de K-Means clustering en un grid 3x3.

    Muestra para cada imagen procesada:
    - Columna 1: Imagen original
    - Columna 2: Imagen clusterizada (posterizada con K colores dominantes)
    - Columna 3: Gráfico de barras mostrando colores dominantes y sus frecuencias

    Args:
        original_images: Diccionario {image_id: numpy_array} con imágenes originales.
                        Arrays con shape (H, W, 3) y valores uint8 [0-255].
        kmeans_results: Diccionario {image_id: KMeans} con objetos KMeans fitted.
        image_ids: Lista de IDs de imágenes a mostrar en orden.
                   Por defecto: ['12074', '42044', '100075']
        figsize: Tamaño de la figura (ancho, alto) en pulgadas.

    Returns:
        fig: Figura de matplotlib con el grid de resultados.

    Raises:
        ValueError: Si image_ids no coinciden entre los diccionarios.
        ValueError: Si alguna imagen no tiene el formato correcto.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.viz import plot_kmeans_results
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>>
        >>> fig = plot_kmeans_results(images, kmeans_results)
        >>> plt.show()
    """
    # Valores por defecto para image_ids
    if image_ids is None:
        image_ids = ['12074', '42044', '100075']

    # Validar que todos los image_ids existen en ambos diccionarios
    missing_in_original = [img_id for img_id in image_ids if img_id not in original_images]
    missing_in_results = [img_id for img_id in image_ids if img_id not in kmeans_results]

    if missing_in_original or missing_in_results:
        error_msg = "Faltan image_ids en los diccionarios:\n"
        if missing_in_original:
            error_msg += f"  - Faltan en original_images: {missing_in_original}\n"
        if missing_in_results:
            error_msg += f"  - Faltan en kmeans_results: {missing_in_results}\n"
        error_msg += f"IDs solicitados: {image_ids}\n"
        error_msg += f"IDs en original_images: {list(original_images.keys())}\n"
        error_msg += f"IDs en kmeans_results: {list(kmeans_results.keys())}"
        raise ValueError(error_msg)

    n_images = len(image_ids)

    # Crear figura con grid de n_images filas x 3 columnas
    fig, axes = plt.subplots(n_images, 3, figsize=figsize)

    # Si solo hay una imagen, axes no es 2D
    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Iterar sobre cada imagen
    for row_idx, img_id in enumerate(image_ids):
        # Obtener datos
        original_img = original_images[img_id]
        kmeans = kmeans_results[img_id]

        # === COLUMNA 1: Imagen Original ===
        ax_original = axes[row_idx, 0]
        ax_original.imshow(original_img)
        ax_original.axis('off')
        ax_original.set_title(f'BSD500 #{img_id} - Original', fontsize=12, pad=10)

        # === COLUMNA 2: Imagen Clusterizada ===
        ax_clustered = axes[row_idx, 1]

        # Obtener imagen segmentada (en rango [0, 1] porque fit_image fue con normalización)
        segmented_img = kmeans.get_segmented_image(original_img.shape[:2])

        # Asegurar que está en rango [0, 1] para imshow
        segmented_img = np.clip(segmented_img, 0, 1)

        ax_clustered.imshow(segmented_img)
        ax_clustered.axis('off')
        ax_clustered.set_title(
            f'Clusterizada (K={kmeans.config.n_clusters})',
            fontsize=12,
            pad=10
        )

        # === COLUMNA 3: Gráfico de Barras con Colores Dominantes ===
        ax_colors = axes[row_idx, 2]

        # Calcular frecuencias de cada cluster
        labels = kmeans.labels
        n_pixels = len(labels)
        n_clusters = kmeans.config.n_clusters

        # Contar píxeles por cluster
        cluster_counts = np.bincount(labels, minlength=n_clusters)
        cluster_percentages = (cluster_counts / n_pixels) * 100

        # Obtener centroides (colores dominantes) en rango [0, 1]
        centroids = kmeans.centroids

        # Asegurar que centroides están en [0, 1]
        centroids = np.clip(centroids, 0, 1)

        # Crear gráfico de barras
        x_pos = np.arange(n_clusters)
        bars = ax_colors.bar(
            x_pos,
            cluster_percentages,
            color=centroids,  # Usar los colores RGB de los centroides
            edgecolor='black',
            linewidth=1.5
        )

        # Configurar ejes
        ax_colors.set_xlabel('Cluster', fontsize=10)
        ax_colors.set_ylabel('Frecuencia (%)', fontsize=10)
        ax_colors.set_title('Colores Dominantes', fontsize=12, pad=10)
        ax_colors.set_xticks(x_pos)
        ax_colors.set_xticklabels([f'{i}' for i in range(n_clusters)])
        ax_colors.set_ylim(0, max(cluster_percentages) * 1.1)  # 10% padding arriba

        # Agregar valores de porcentaje sobre cada barra
        for bar, percentage in zip(bars, cluster_percentages):
            height = bar.get_height()
            ax_colors.text(
                bar.get_x() + bar.get_width() / 2.,
                height,
                f'{percentage:.1f}%',
                ha='center',
                va='bottom',
                fontsize=8
            )

    # Ajustar layout
    plt.tight_layout()

    return fig


def plot_levelset_results(
    original_images: Dict[str, np.ndarray],
    kmeans_results: Dict[str, 'KMeans'],
    levelset_results: Dict[str, 'LevelSet'],
    image_ids: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 18)
) -> plt.Figure:
    """
    Visualiza transformación de K-Means a Signed Distance Function en un grid 3x2.

    Muestra para cada imagen procesada:
    - Columna 1: Imagen segmentada K-Means (posterizada con K colores dominantes)
    - Columna 2: Signed Distance Function (mapa de distancia con colorbar)

    El SDF representa la distancia de cada píxel al borde del cluster más cercano,
    mostrando la transformación desde K-Means labels hacia una representación continua
    de nivel de conjunto (level set) usando Fast Marching Method.

    Args:
        original_images: Diccionario {image_id: numpy_array} con imágenes originales.
                        Arrays con shape (H, W, 3) y valores uint8 [0-255].
                        Necesario para obtener dimensiones de imagen.
        kmeans_results: Diccionario {image_id: KMeans} con objetos KMeans fitted.
        levelset_results: Diccionario {image_id: LevelSet} con objetos LevelSet inicializados.
        image_ids: Lista de IDs de imágenes a mostrar en orden.
                   Por defecto: ['12074', '42044', '100075']
        figsize: Tamaño de la figura (ancho, alto) en pulgadas.

    Returns:
        fig: Figura de matplotlib con el grid de resultados.

    Raises:
        ValueError: Si image_ids no coinciden entre los diccionarios.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, LevelSetConfig
        >>> from src.viz import plot_levelset_results
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # K-Means
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>>
        >>> # Level Sets
        >>> levelset_config = LevelSetConfig(normalize=True, boundary_threshold=0.05)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>>
        >>> # Visualizar
        >>> fig = plot_levelset_results(images, kmeans_results, levelset_results)
        >>> plt.show()
    """
    # Valores por defecto para image_ids
    if image_ids is None:
        image_ids = ['12074', '42044', '100075']

    # Validar que todos los image_ids existen en los tres diccionarios
    missing_in_original = [img_id for img_id in image_ids if img_id not in original_images]
    missing_in_kmeans = [img_id for img_id in image_ids if img_id not in kmeans_results]
    missing_in_levelset = [img_id for img_id in image_ids if img_id not in levelset_results]

    if missing_in_original or missing_in_kmeans or missing_in_levelset:
        error_msg = "Faltan image_ids en los diccionarios:\n"
        if missing_in_original:
            error_msg += f"  - Faltan en original_images: {missing_in_original}\n"
        if missing_in_kmeans:
            error_msg += f"  - Faltan en kmeans_results: {missing_in_kmeans}\n"
        if missing_in_levelset:
            error_msg += f"  - Faltan en levelset_results: {missing_in_levelset}\n"
        error_msg += f"IDs solicitados: {image_ids}"
        raise ValueError(error_msg)

    n_images = len(image_ids)

    # Crear figura con grid de n_images filas x 2 columnas
    fig, axes = plt.subplots(n_images, 2, figsize=figsize)

    # Si solo hay una imagen, axes no es 2D
    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Iterar sobre cada imagen
    for row_idx, img_id in enumerate(image_ids):
        # Obtener datos
        original_img = original_images[img_id]
        kmeans = kmeans_results[img_id]
        levelset = levelset_results[img_id]

        # === COLUMNA 1: Imagen Segmentada K-Means ===
        ax_segmented = axes[row_idx, 0]

        # Obtener imagen segmentada (en rango [0, 1] porque fit_image fue con normalización)
        segmented_img = kmeans.get_segmented_image(original_img.shape[:2])

        # Asegurar que está en rango [0, 1] para imshow
        segmented_img = np.clip(segmented_img, 0, 1)

        ax_segmented.imshow(segmented_img)
        ax_segmented.axis('off')
        ax_segmented.set_title(
            f'BSD500 #{img_id} - K-Means (K={kmeans.config.n_clusters})',
            fontsize=12,
            pad=10
        )

        # === COLUMNA 2: Signed Distance Function ===
        ax_sdf = axes[row_idx, 1]

        # Obtener phi (combined signed distance function)
        phi = levelset.phi  # Shape: (H, W)

        # Mostrar SDF como mapa de calor
        im = ax_sdf.imshow(phi, cmap='RdBu_r', interpolation='nearest')
        ax_sdf.axis('off')
        ax_sdf.set_title('Signed Distance Function', fontsize=12, pad=10)

        # Agregar colorbar
        cbar = plt.colorbar(im, ax=ax_sdf, fraction=0.046, pad=0.04)
        cbar.set_label('Distance to Boundary', fontsize=9)

    # Título general
    fig.suptitle(
        'Transformación: K-Means → Signed Distance Function',
        fontsize=14,
        y=0.995
    )

    # Ajustar layout
    plt.tight_layout()

    return fig


def plot_cluster_distances_grid(
    image_id: str,
    kmeans: 'KMeans',
    levelset: 'LevelSet',
    original_img: np.ndarray,
    figsize: Optional[Tuple[int, int]] = None
) -> plt.Figure:
    """
    Visualiza Signed Distance Functions por cluster individual en un grid.

    Crea un grid con un subplot por cada cluster, mostrando el SDF de ese cluster.
    Cada subplot muestra:
    - SDF como heatmap con colormap divergente (RdBu_r)
    - Contorno amarillo en φ=0 indicando el borde exacto del cluster
    - Valores negativos (azul): píxeles dentro del cluster
    - Valores positivos (rojo): píxeles fuera del cluster
    - Zero (blanco): frontera del cluster

    Args:
        image_id: ID de la imagen BSD500 (e.g., '12074', '42044', '100075')
        kmeans: Objeto KMeans fitted con información de clustering
        levelset: Objeto LevelSet con phi_per_cluster calculados
        original_img: Imagen original (H, W, 3) para obtener dimensiones
        figsize: Tamaño de la figura. Si None, se calcula automáticamente
                 basado en el número de clusters.

    Returns:
        fig: Figura de matplotlib con el grid de SDFs por cluster.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, LevelSetConfig
        >>> from src.viz import plot_cluster_distances_grid
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>> levelset_config = LevelSetConfig(normalize=True, boundary_threshold=0.05)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>>
        >>> # Mostrar SDFs por cluster para imagen 12074
        >>> fig = plot_cluster_distances_grid(
        ...     '12074',
        ...     kmeans_results['12074'],
        ...     levelset_results['12074'],
        ...     images['12074']
        ... )
        >>> plt.show()
    """
    # Obtener número de clusters
    n_clusters = kmeans.config.n_clusters

    # Obtener phi_per_cluster del levelset
    phi_per_cluster = levelset.phi_per_cluster  # Shape: (K, H, W)

    # Obtener labels para overlays (opcional)
    labels = levelset.labels  # Shape: (H, W)

    # Calcular layout óptimo del grid
    # Usar una disposición aproximadamente cuadrada
    ncols = int(np.ceil(np.sqrt(n_clusters)))
    nrows = int(np.ceil(n_clusters / ncols))

    # Calcular figsize si no se proporciona
    if figsize is None:
        # 5 pulgadas por subplot en cada dirección
        figsize = (ncols * 5, nrows * 5)

    # Crear figura con subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Si solo hay un subplot, axes no es un array
    if n_clusters == 1:
        axes = np.array([axes])
    else:
        axes = axes.flatten()

    # Iterar sobre cada cluster
    for cluster_idx in range(n_clusters):
        ax = axes[cluster_idx]

        # Obtener SDF para este cluster
        phi_cluster = phi_per_cluster[cluster_idx]  # Shape: (H, W)

        # Calcular límite simétrico para colormap centrado en cero
        vmax = np.abs(phi_cluster).max()

        # Mostrar SDF como heatmap
        im = ax.imshow(
            phi_cluster,
            cmap='RdBu_r',  # Rojo=dentro (negativo), Azul=fuera (positivo)
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest'
        )

        # Agregar contorno en φ=0 (frontera del cluster)
        ax.contour(
            phi_cluster,
            levels=[0],
            colors='yellow',
            linewidths=2,
            alpha=0.8
        )

        # Configurar subplot
        ax.axis('off')
        ax.set_title(f'Cluster {cluster_idx}', fontsize=11, pad=8)

        # Agregar colorbar pequeño
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('SDF', fontsize=9)

    # Ocultar subplots vacíos si los hay
    for idx in range(n_clusters, len(axes)):
        axes[idx].axis('off')
        axes[idx].set_visible(False)

    # Título general de la figura
    fig.suptitle(
        f'BSD500 #{image_id} - Per-Cluster Signed Distance Functions (K={n_clusters})',
        fontsize=14,
        y=0.98
    )

    # Ajustar layout
    plt.tight_layout()

    return fig


def plot_evolution_results(
    original_images: Dict[str, np.ndarray],
    evolved_results: Dict[str, np.ndarray],
    kmeans_results: Dict[str, 'KMeans'],
    image_ids: Optional[List[str]] = None
) -> Dict[str, plt.Figure]:
    """
    Visualiza resultados de level set evolution para múltiples imágenes.

    Para cada imagen, crea un grid de subplots mostrando la imagen original
    con las fronteras evolucionadas (zero-level contours de φ) overlaid para
    cada cluster. Similar a la visualización "Step 4: Evolution Results" del
    notebook de referencia.

    Args:
        original_images: Diccionario {image_id: numpy_array} con imágenes originales.
                        Arrays con shape (H, W, 3) y valores uint8 [0-255].
        evolved_results: Diccionario {image_id: evolved_phi} donde evolved_phi
                        es un array (K, H, W) con signed distance functions evolucionados.
        kmeans_results: Diccionario {image_id: KMeans} con objetos KMeans fitted.
                       Necesario para obtener el número de clusters K.
        image_ids: Lista de IDs de imágenes a mostrar.
                   Por defecto: ['12074', '42044', '100075']

    Returns:
        Diccionario {image_id: fig} con una figura matplotlib por cada imagen.
        Cada figura muestra un grid de K subplots (uno por cluster) con:
        - Imagen original como background
        - Contorno amarillo en φ=0 (frontera evolucionada del cluster)

    Raises:
        ValueError: Si image_ids no coinciden entre los diccionarios.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, process_evolution_batch, LevelSetConfig
        >>> from src.viz import plot_evolution_results
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # K-Means
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>>
        >>> # Level Sets
        >>> levelset_config = LevelSetConfig(normalize=True, boundary_threshold=0.05)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, levelset_config)
        >>>
        >>> # Evolution
        >>> evolved_results = process_evolution_batch(levelset_results, images)
        >>>
        >>> # Visualizar
        >>> figs = plot_evolution_results(images, evolved_results, kmeans_results)
        >>> figs['12074']  # Mostrar figura de imagen 12074
    """
    # Valores por defecto para image_ids
    if image_ids is None:
        image_ids = ['12074', '42044', '100075']

    # Validar que todos los image_ids existen en los tres diccionarios
    missing_in_original = [img_id for img_id in image_ids if img_id not in original_images]
    missing_in_evolved = [img_id for img_id in image_ids if img_id not in evolved_results]
    missing_in_kmeans = [img_id for img_id in image_ids if img_id not in kmeans_results]

    if missing_in_original or missing_in_evolved or missing_in_kmeans:
        error_msg = "Faltan image_ids en los diccionarios:\n"
        if missing_in_original:
            error_msg += f"  - Faltan en original_images: {missing_in_original}\n"
        if missing_in_evolved:
            error_msg += f"  - Faltan en evolved_results: {missing_in_evolved}\n"
        if missing_in_kmeans:
            error_msg += f"  - Faltan en kmeans_results: {missing_in_kmeans}\n"
        error_msg += f"IDs solicitados: {image_ids}"
        raise ValueError(error_msg)

    # Crear una figura por cada imagen
    figures = {}

    for img_id in image_ids:
        # Obtener datos
        original_img = original_images[img_id]
        evolved_phi = evolved_results[img_id]  # Shape: (K, H, W)
        kmeans = kmeans_results[img_id]
        n_clusters = kmeans.config.n_clusters

        # Calcular layout óptimo del grid
        ncols = int(np.ceil(np.sqrt(n_clusters)))
        nrows = int(np.ceil(n_clusters / ncols))

        # Calcular figsize automáticamente
        figsize = (ncols * 5, nrows * 4)

        # Crear figura con subplots
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        # Si solo hay un subplot, axes no es un array
        if n_clusters == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()

        # Plot evolved boundaries para cada cluster
        for cluster_idx in range(n_clusters):
            ax = axes[cluster_idx]

            # Mostrar imagen original como background
            ax.imshow(original_img)

            # Overlay evolved boundary (zero-level contour de φ)
            phi_cluster = evolved_phi[cluster_idx]  # Shape: (H, W)
            ax.contour(
                phi_cluster,
                levels=[0],
                colors='yellow',
                linewidths=2,
                alpha=0.9
            )

            # Configurar subplot
            ax.axis('off')
            ax.set_title(f'Cluster {cluster_idx} Evolved Boundary', fontsize=10, pad=5)

        # Ocultar subplots vacíos
        for idx in range(n_clusters, len(axes)):
            axes[idx].axis('off')
            axes[idx].set_visible(False)

        # Título general
        fig.suptitle(
            f'BSD500 #{img_id} - Evolved Boundaries on Original Image (K={n_clusters} clusters)',
            fontsize=14,
            y=0.98
        )

        # Ajustar layout
        plt.tight_layout()

        # Guardar figura en diccionario
        figures[img_id] = fig

    return figures


def _colorize_labels(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    """
    Convierte array de labels a imagen RGB con colores distintos por cluster.

    Asigna a cada cluster un color distinto de un colormap para visualización.

    Args:
        labels: Array de labels (H, W) con valores de 0 a n_clusters-1
        n_clusters: Número de clusters

    Returns:
        colored_image: Imagen RGB (H, W, 3) con valores uint8 [0, 255]

    Raises:
        ValueError: Si labels no es un array 2D

    Example:
        >>> labels = np.array([[0, 0, 1], [1, 2, 2]])  # Shape: (2, 3)
        >>> colored = _colorize_labels(labels, n_clusters=3)
        >>> colored.shape
        (2, 3, 3)
    """
    # Asegurar que labels es 2D
    if labels.ndim == 1:
        raise ValueError(f"Labels debe ser un array 2D (H, W), se recibió shape {labels.shape}")

    h, w = labels.shape

    # Usar 'tab10' para hasta 10 clusters, 'tab20' para más
    if n_clusters <= 10:
        cmap = plt.cm.get_cmap('tab10')
    else:
        cmap = plt.cm.get_cmap('tab20')

    # Crear imagen RGB mapeando cada cluster a un color
    colored_image = np.zeros((h, w, 3), dtype=np.uint8)

    for cluster_id in range(n_clusters):
        # Obtener color para este cluster del colormap
        # Normalizar cluster_id al rango [0, 1]
        normalized_id = cluster_id / max(n_clusters - 1, 1)
        color_rgba = cmap(normalized_id)  # Retorna (R, G, B, A) en [0, 1]
        color_rgb = (np.array(color_rgba[:3]) * 255).astype(np.uint8)

        # Asignar color a todos los píxeles de este cluster
        mask = (labels == cluster_id)
        colored_image[mask] = color_rgb

    return colored_image


def plot_segmentation_comparison_batch(
    original_images: Dict[str, np.ndarray],
    kmeans_results: Dict[str, 'KMeans'],
    evolved_labels_dict: Dict[str, np.ndarray],
    image_ids: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (18, 18)
) -> plt.Figure:
    """
    Visualiza comparación de segmentación K-Means vs Evolved en grid 3x3.

    Muestra para cada imagen procesada:
    - Columna 1: Imagen original
    - Columna 2: Segmentación K-Means (labels colorizados)
    - Columna 3: Segmentación Evolved (labels evolucionados colorizados)

    Este grid permite comparar visualmente cómo la evolución de level sets
    refina las fronteras de los clusters obtenidos con K-Means.

    Args:
        original_images: Diccionario {image_id: numpy_array} con imágenes originales.
                        Arrays con shape (H, W, 3) y valores uint8 [0-255].
        kmeans_results: Diccionario {image_id: KMeans} con objetos KMeans fitted.
        evolved_labels_dict: Diccionario {image_id: evolved_labels} donde evolved_labels
                            es un array (H, W) con cluster assignments después de evolución.
        image_ids: Lista de IDs de imágenes a mostrar en orden.
                   Por defecto: ['12074', '42044', '100075']
        figsize: Tamaño de la figura (ancho, alto) en pulgadas.

    Returns:
        fig: Figura de matplotlib con el grid 3x3 de comparación.

    Raises:
        ValueError: Si image_ids no coinciden entre los diccionarios.

    Example:
        >>> from src.data_loader import DataLoader
        >>> from src.kmeans import process_images_batch
        >>> from src.level_set import process_levelsets_batch, process_evolution_batch
        >>> from src.level_set import extract_evolved_labels_batch
        >>> from src.viz import plot_segmentation_comparison_batch
        >>>
        >>> loader = DataLoader()
        >>> images = loader.load_all_images()
        >>> snapshots = {img_id: loader.load_result(img_id) for img_id in images}
        >>>
        >>> # Pipeline completo
        >>> kmeans_results = process_images_batch(images, snapshots)
        >>> levelset_results = process_levelsets_batch(kmeans_results, images, config)
        >>> evolved_results = process_evolution_batch(levelset_results, images, snapshots)
        >>> evolved_labels = extract_evolved_labels_batch(levelset_results)
        >>>
        >>> # Comparación visual
        >>> fig = plot_segmentation_comparison_batch(images, kmeans_results, evolved_labels)
        >>> plt.show()
    """
    # Valores por defecto para image_ids
    if image_ids is None:
        image_ids = ['12074', '42044', '100075']

    # Validar que todos los image_ids existen en los tres diccionarios
    missing_in_original = [img_id for img_id in image_ids if img_id not in original_images]
    missing_in_kmeans = [img_id for img_id in image_ids if img_id not in kmeans_results]
    missing_in_evolved = [img_id for img_id in image_ids if img_id not in evolved_labels_dict]

    if missing_in_original or missing_in_kmeans or missing_in_evolved:
        error_msg = "Faltan image_ids en los diccionarios:\n"
        if missing_in_original:
            error_msg += f"  - Faltan en original_images: {missing_in_original}\n"
        if missing_in_kmeans:
            error_msg += f"  - Faltan en kmeans_results: {missing_in_kmeans}\n"
        if missing_in_evolved:
            error_msg += f"  - Faltan en evolved_labels_dict: {missing_in_evolved}\n"
        error_msg += f"IDs solicitados: {image_ids}"
        raise ValueError(error_msg)

    n_images = len(image_ids)

    # Crear figura con grid de n_images filas x 3 columnas
    fig, axes = plt.subplots(n_images, 3, figsize=figsize)

    # Si solo hay una imagen, axes no es 2D
    if n_images == 1:
        axes = axes.reshape(1, -1)

    # Iterar sobre cada imagen
    for row_idx, img_id in enumerate(image_ids):
        # Obtener datos
        original_img = original_images[img_id]
        kmeans = kmeans_results[img_id]
        evolved_labels = evolved_labels_dict[img_id]

        # Obtener dimensiones y número de clusters
        h, w = original_img.shape[:2]
        n_clusters = kmeans.config.n_clusters

        # === COLUMNA 1: Imagen Original ===
        ax_original = axes[row_idx, 0]
        ax_original.imshow(original_img)
        ax_original.axis('off')
        ax_original.set_title(f'BSD500 #{img_id} - Original', fontsize=12, pad=10)

        # === COLUMNA 2: K-Means Segmentation ===
        ax_kmeans = axes[row_idx, 1]

        # Obtener labels de K-Means y reshape si es necesario
        kmeans_labels = kmeans.labels
        if kmeans_labels.ndim == 1:
            kmeans_labels = kmeans_labels.reshape(h, w)

        # Colorizar labels de K-Means
        kmeans_colored = _colorize_labels(kmeans_labels, n_clusters)

        ax_kmeans.imshow(kmeans_colored)
        ax_kmeans.axis('off')
        ax_kmeans.set_title(
            f'K-Means Segmentation (k={n_clusters})',
            fontsize=12,
            pad=10
        )

        # === COLUMNA 3: Evolved Segmentation ===
        ax_evolved = axes[row_idx, 2]

        # Reshape evolved labels si es necesario
        if evolved_labels.ndim == 1:
            evolved_labels = evolved_labels.reshape(h, w)

        # Colorizar labels evolucionados
        evolved_colored = _colorize_labels(evolved_labels, n_clusters)

        ax_evolved.imshow(evolved_colored)
        ax_evolved.axis('off')
        ax_evolved.set_title(
            f'Evolved Segmentation (after level set)',
            fontsize=12,
            pad=10
        )

    # Título general
    fig.suptitle(
        'Segmentation Comparison: K-Means → Level Set Evolution',
        fontsize=14,
        y=0.995
    )

    # Ajustar layout
    plt.tight_layout()

    return fig


def plot_silhouette_comparison(
    original_image: np.ndarray,
    silhouette_image: np.ndarray,
    ground_truth_labels: Optional[np.ndarray] = None,
    method_name: str = "Canny Edges",
    figsize: Tuple[int, int] = (18, 6)
) -> plt.Figure:
    """
    Display paper-style 3-column silhouette comparison.

    Creates a 3-column visualization matching the paper's presentation style:
    - Column 1: Original RGB image
    - Column 2: Silhouette extraction result (black contours on white background)
    - Column 3: Ground truth contours (black contours on white background)

    This matches the presentation in Figure 3 of the paper.

    Args:
        original_image: Original RGB image (H, W, 3)
        silhouette_image: Silhouette result (H, W, 3) - black contours on white
        ground_truth_labels: Ground truth segmentation labels (H, W), optional
        method_name: Name of silhouette extraction method used
        figsize: Figure size (width, height)

    Returns:
        fig: Matplotlib figure object
    """
    # Get image dimensions
    h, w = original_image.shape[:2]

    # Create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # Column 1: Original Image
    axes[0].imshow(original_image)
    axes[0].axis('off')
    axes[0].set_title('Original Image', fontsize=12, pad=10)

    # Column 2: Silhouette Extraction Result
    axes[1].imshow(silhouette_image)
    axes[1].axis('off')
    axes[1].set_title(f'Our Result - {method_name}', fontsize=12, pad=10)

    # Column 3: Ground Truth Contours
    if ground_truth_labels is not None:
        # Reshape if needed
        if ground_truth_labels.ndim == 1:
            ground_truth_labels = ground_truth_labels.reshape(h, w)

        # Extract ground truth contours using Sobel edge detection
        from scipy.ndimage import sobel
        gt_labels = ground_truth_labels.astype(np.float64)
        grad_x = np.abs(sobel(gt_labels, axis=1))
        grad_y = np.abs(sobel(gt_labels, axis=0))
        gt_boundaries = ((grad_x + grad_y) > 0).astype(np.uint8) * 255

        # Extract contours from GT boundaries
        import cv2
        gt_contours, _ = cv2.findContours(
            gt_boundaries,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Draw GT contours on white background
        gt_contours_img = np.ones((h, w, 3), dtype=np.uint8) * 255  # White background
        cv2.drawContours(gt_contours_img, gt_contours, -1, (0, 0, 0), 1)  # Black contours

        axes[2].imshow(gt_contours_img)
        axes[2].axis('off')
        axes[2].set_title('Ground Truth\n(Human Annotation)', fontsize=12, pad=10)
    else:
        # No ground truth available - show placeholder
        placeholder = np.ones((h, w, 3), dtype=np.uint8) * 240
        axes[2].imshow(placeholder)
        axes[2].axis('off')
        axes[2].set_title('Ground Truth\n(Not Available)', fontsize=12, pad=10)
        axes[2].text(
            w/2, h/2, 'No Ground Truth',
            ha='center', va='center',
            fontsize=16, color='gray'
        )

    fig.suptitle(
        'Silhouette Extraction Results (Paper-Style Presentation)',
        fontsize=14,
        y=0.98
    )
    plt.tight_layout()

    return fig
