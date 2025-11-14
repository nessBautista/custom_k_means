"""
Cargador de archivos JSON con parámetros y resultados guardados.

Proporciona la clase DataLoader para cargar y parsear archivos JSON
que contienen parámetros de configuración y resultados de ejecuciones previas,
así como las imágenes JPG correspondientes.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Optional, Dict, Tuple
import scipy.io as sio
from .notebook_params import NotebookSnapshot


class DataLoader:
    """
    Cargador de archivos JSON con parámetros y resultados guardados.

    Busca automáticamente archivos JSON en el directorio src/data/
    del proyecto KmeansV3 y los parsea a objetos NotebookSnapshot.

    Los archivos JSON siguen el formato:
        params_{image_id}_{timestamp}.json

    Example:
        >>> # Uso básico
        >>> loader = DataLoader()
        >>> snapshot = loader.load_result('12074')
        >>> print(snapshot.parameters.k_clusters)  # 5
        >>> print(snapshot.results.pri_evolved)    # 0.6594

        >>> # Listar imágenes disponibles
        >>> images = loader.get_available_images()
        >>> print(images)  # ['12074', '42044', '100075']

        >>> # Usar ruta personalizada
        >>> loader = DataLoader(data_dir=Path('/custom/path/'))
        >>> snapshot = loader.load_result('42044')
    """

    def __init__(self, data_dir: Optional[Path] = None):
        """
        Inicializa el cargador de datos.

        La búsqueda del directorio de datos es automática y robusta:
        - Si data_dir es None, busca automáticamente KmeansV3/src/data/
        - Primero busca desde el directorio actual
        - Luego busca en directorios padre
        - Si no encuentra, lanza FileNotFoundError

        Args:
            data_dir: Directorio donde buscar los archivos JSON.
                     Si None, busca automáticamente en KmeansV3/src/data/.
                     Útil para testing o rutas no estándar.

        Raises:
            FileNotFoundError: Si no se puede localizar el directorio de datos
                              y data_dir es None.

        Example:
            >>> # Búsqueda automática
            >>> loader = DataLoader()

            >>> # Ruta personalizada
            >>> from pathlib import Path
            >>> loader = DataLoader(data_dir=Path('/path/to/data/'))
        """
        if data_dir is None:
            # Búsqueda automática del directorio de datos
            data_dir = self._find_data_directory()

        self.data_dir = Path(data_dir)

        # Validar que el directorio existe
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"El directorio de datos no existe: {self.data_dir}"
            )

    def _find_data_directory(self) -> Path:
        """
        Busca automáticamente el directorio KmeansV3/src/data/.

        Estrategia de búsqueda:
        1. Si cwd es KmeansV3/, busca en ./src/data/
        2. Si cwd contiene KmeansV3/, busca en ./KmeansV3/src/data/
        3. Busca en directorios padre hasta encontrar KmeansV3/

        Returns:
            Path: Ruta al directorio de datos

        Raises:
            FileNotFoundError: Si no se encuentra el directorio de datos
        """
        current = Path.cwd()

        # Caso 1: Estamos en KmeansV3/ (directorio raíz del proyecto)
        if current.name == 'KmeansV3':
            data_path = current / 'src' / 'data'
            if data_path.exists():
                return data_path

        # Caso 2: KmeansV3/ es subdirectorio del directorio actual
        if (current / 'KmeansV3').exists():
            data_path = current / 'KmeansV3' / 'src' / 'data'
            if data_path.exists():
                return data_path

        # Caso 3: Buscar hacia arriba en el árbol de directorios
        for parent in current.parents:
            candidate = parent / 'KmeansV3' / 'src' / 'data'
            if candidate.exists():
                return candidate

        # No se encontró el directorio
        raise FileNotFoundError(
            f"No se pudo encontrar el directorio KmeansV3/src/data/.\n"
            f"Búsqueda iniciada desde: {Path.cwd()}\n"
            f"Casos explorados:\n"
            f"  1. {Path.cwd()}/src/data/ (si cwd es KmeansV3)\n"
            f"  2. {Path.cwd()}/KmeansV3/src/data/\n"
            f"  3. Directorios parent hasta encontrar KmeansV3\n"
            f"Solución: Ejecuta desde KmeansV3/ o especifica data_dir manualmente."
        )

    def _find_json_for_image(self, image_id: str) -> Optional[Path]:
        """
        Busca el archivo JSON más reciente para una imagen específica.

        Si hay múltiples archivos para la misma imagen (diferentes timestamps),
        retorna el más reciente según la fecha de modificación del archivo.

        Args:
            image_id: ID de imagen BSD500 (e.g., '12074')

        Returns:
            Path al archivo JSON más reciente, o None si no se encuentra ninguno.

        Example:
            >>> loader = DataLoader()
            >>> path = loader._find_json_for_image('12074')
            >>> print(path)  # KmeansV3/src/data/params_12074_20251113_104531.json
        """
        # Patrón de búsqueda: params_{image_id}_*.json
        pattern = f"params_{image_id}_*.json"
        json_files = list(self.data_dir.glob(pattern))

        if not json_files:
            return None

        # Si hay múltiples archivos, ordenar por fecha de modificación
        # y retornar el más reciente
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        return json_files[0]

    def load_result(self, image_id: str) -> NotebookSnapshot:
        """
        Carga el archivo JSON de resultados para una imagen específica.

        Busca archivos con patrón: params_{image_id}_*.json
        Si hay múltiples archivos, carga el más reciente según fecha de modificación.

        El JSON se parsea a un objeto NotebookSnapshot que contiene:
        - metadata: dict con image_id, timestamp, description
        - parameters: NotebookParameters con k_clusters, velocity_method, etc.
        - results: NotebookResults con PRI scores, contour_count, etc.

        Args:
            image_id: ID de imagen BSD500 (e.g., '12074', '42044', '100075').
                     Debe ser string, no int.

        Returns:
            NotebookSnapshot: Objeto con metadata, parameters y results.

        Raises:
            FileNotFoundError: Si no se encuentra JSON para el image_id especificado.
            json.JSONDecodeError: Si el archivo JSON está malformado.
            KeyError: Si el JSON no tiene la estructura esperada.

        Example:
            >>> loader = DataLoader()
            >>>
            >>> # Cargar resultados de imagen 12074
            >>> snapshot = loader.load_result('12074')
            >>>
            >>> # Acceder a metadata
            >>> print(snapshot.metadata['timestamp'])  # '2025-11-13T10:45:31...'
            >>>
            >>> # Acceder a parámetros
            >>> params = snapshot.parameters
            >>> print(f"K={params.k_clusters}")              # K=5
            >>> print(f"Método={params.velocity_method}")    # Método=curvature_skimage
            >>> print(f"Alpha={params.alpha}")               # Alpha=100.0
            >>>
            >>> # Acceder a resultados
            >>> results = snapshot.results
            >>> print(f"PRI K-means: {results.pri_kmeans:.4f}")  # 0.6583
            >>> print(f"PRI Evolved: {results.pri_evolved:.4f}") # 0.6594
            >>> improvement = results.pri_improvement_pct
            >>> print(f"Mejora: {improvement:.2f}%")             # 0.17%
        """
        # Buscar archivo JSON para la imagen
        json_path = self._find_json_for_image(image_id)

        if json_path is None:
            # No se encontró ningún archivo
            available = self.get_available_images()
            raise FileNotFoundError(
                f"No se encontró archivo JSON para image_id '{image_id}'.\n"
                f"Buscado en: {self.data_dir}\n"
                f"Patrón: params_{image_id}_*.json\n"
                f"Imágenes disponibles: {available}\n"
                f"Archivos JSON encontrados: {list(self.data_dir.glob('*.json'))}"
            )

        # Cargar y parsear el archivo JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                f"Error al parsear JSON de {json_path}: {e.msg}",
                e.doc,
                e.pos
            )

        # Convertir dict a NotebookSnapshot usando from_dict()
        try:
            snapshot = NotebookSnapshot.from_dict(data)
        except KeyError as e:
            raise KeyError(
                f"El JSON no tiene la estructura esperada.\n"
                f"Archivo: {json_path}\n"
                f"Campo faltante: {e}\n"
                f"Estructura esperada: {{'metadata': {...}, 'parameters': {...}, 'results': {...}}}"
            )

        return snapshot

    def get_available_images(self) -> List[str]:
        """
        Lista los image_ids que tienen archivos JSON disponibles.

        Busca todos los archivos con patrón params_*.json y extrae
        los image_ids únicos.

        Returns:
            Lista de IDs de imágenes (strings) ordenados alfabéticamente.
            Ejemplo: ['12074', '42044', '100075']

        Example:
            >>> loader = DataLoader()
            >>> images = loader.get_available_images()
            >>> print(images)  # ['12074', '42044', '100075']
            >>>
            >>> # Iterar sobre imágenes disponibles
            >>> for img_id in images:
            ...     snapshot = loader.load_result(img_id)
            ...     print(f"{img_id}: PRI={snapshot.results.pri_evolved:.4f}")
        """
        # Buscar todos los archivos JSON con patrón params_*.json
        json_files = list(self.data_dir.glob("params_*.json"))

        # Extraer image_ids únicos
        image_ids = set()
        for filepath in json_files:
            # Filename format: params_{image_id}_{timestamp}.json
            # Ejemplo: params_12074_20251113_104531.json
            filename = filepath.stem  # Quita la extensión .json

            # Split por '_' y extraer la segunda parte (image_id)
            parts = filename.split('_')
            if len(parts) >= 2:
                image_id = parts[1]  # Segunda parte es el image_id
                image_ids.add(image_id)

        # Convertir a lista ordenada
        return sorted(list(image_ids))

    def load_image(self, image_id: str) -> np.ndarray:
        """
        Carga una imagen JPG por su ID.

        Busca el archivo {image_id}.jpg en el directorio de datos
        y lo carga como un numpy array RGB.

        Args:
            image_id: ID de imagen BSD500 (e.g., '12074', '42044', '100075').
                     Debe ser string, no int.

        Returns:
            numpy array con shape (H, W, 3) y valores uint8 [0-255].
            La imagen está en formato RGB.

        Raises:
            FileNotFoundError: Si la imagen no existe en el directorio de datos.

        Example:
            >>> loader = DataLoader()
            >>> img = loader.load_image('12074')
            >>> print(img.shape)  # e.g., (321, 481, 3)
            >>> print(img.dtype)  # dtype('uint8')
            >>> print(img.min(), img.max())  # 0 255

            >>> # Usar con matplotlib
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(img)
            >>> plt.show()
        """
        # Construir ruta al archivo de imagen
        image_path = self.data_dir / f"{image_id}.jpg"

        # Validar que la imagen existe
        if not image_path.exists():
            available = self.get_available_images()
            available_images = list(self.data_dir.glob("*.jpg"))
            raise FileNotFoundError(
                f"Imagen '{image_id}.jpg' no encontrada en {self.data_dir}\n"
                f"Ruta completa buscada: {image_path}\n"
                f"IDs disponibles con JSON: {available}\n"
                f"Imágenes JPG encontradas: {[p.name for p in available_images]}"
            )

        # Cargar imagen usando PIL
        img = Image.open(image_path)

        # Convertir a numpy array (H, W, 3) con valores uint8 [0-255]
        # PIL.Image.open() retorna un objeto PIL Image
        # np.array() lo convierte a numpy array manteniendo el formato RGB
        img_array = np.array(img)

        return img_array

    def load_all_images(
        self,
        image_ids: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Carga múltiples imágenes.

        Por defecto carga las 3 imágenes estándar del proyecto: 12074, 42044, 100075.
        Puedes especificar una lista personalizada de IDs para cargar solo ciertas imágenes.

        Args:
            image_ids: Lista de IDs de imágenes a cargar.
                      Si None, carga ['12074', '42044', '100075'] por defecto.
                      Ejemplo: ['12074', '42044'] para cargar solo 2 imágenes.

        Returns:
            Diccionario {image_id: numpy_array} con las imágenes cargadas.
            Las claves son los image_ids (strings).
            Los valores son numpy arrays con shape (H, W, 3) y dtype uint8.

        Raises:
            FileNotFoundError: Si alguna de las imágenes especificadas no existe.

        Example:
            >>> loader = DataLoader()
            >>>
            >>> # Cargar las 3 imágenes estándar
            >>> images = loader.load_all_images()
            >>> print(images.keys())  # dict_keys(['12074', '42044', '100075'])
            >>> print(images['12074'].shape)  # (321, 481, 3)
            >>>
            >>> # Cargar solo 2 imágenes específicas
            >>> images = loader.load_all_images(image_ids=['12074', '42044'])
            >>> print(len(images))  # 2
            >>>
            >>> # Iterar sobre las imágenes cargadas
            >>> for img_id, img_array in images.items():
            ...     print(f"{img_id}: {img_array.shape}")
        """
        # Valores por defecto: las 3 imágenes estándar del proyecto
        if image_ids is None:
            image_ids = ['12074', '42044', '100075']

        # Cargar cada imagen y almacenar en diccionario
        images = {}
        for img_id in image_ids:
            # Usar load_image() para cada ID
            # Si alguna imagen no existe, load_image() lanzará FileNotFoundError
            images[img_id] = self.load_image(img_id)

        return images

    def load_image_and_result(
        self,
        image_id: str
    ) -> Tuple[np.ndarray, NotebookSnapshot]:
        """
        Carga imagen y resultados juntos para una imagen.

        Método conveniente que combina load_image() y load_result()
        en una sola llamada. Útil cuando necesitas tanto la imagen
        como los parámetros/resultados guardados.

        Args:
            image_id: ID de imagen BSD500 (e.g., '12074', '42044', '100075')

        Returns:
            Tupla (image_array, snapshot):
            - image_array: numpy array (H, W, 3) con la imagen
            - snapshot: NotebookSnapshot con metadata, parameters y results

        Raises:
            FileNotFoundError: Si la imagen o el JSON no existen.
            json.JSONDecodeError: Si el JSON está malformado.

        Example:
            >>> loader = DataLoader()
            >>> img, snapshot = loader.load_image_and_result('12074')
            >>>
            >>> # Acceder a la imagen
            >>> print(f"Imagen shape: {img.shape}")  # (321, 481, 3)
            >>>
            >>> # Acceder a parámetros
            >>> params = snapshot.parameters
            >>> print(f"K-clusters: {params.k_clusters}")  # 5
            >>> print(f"Método: {params.velocity_method}")  # curvature_skimage
            >>>
            >>> # Acceder a resultados
            >>> results = snapshot.results
            >>> print(f"PRI: {results.pri_evolved:.4f}")  # 0.6594
            >>>
            >>> # Usar ambos juntos
            >>> import matplotlib.pyplot as plt
            >>> plt.imshow(img)
            >>> plt.title(f"K={params.k_clusters}, PRI={results.pri_evolved:.4f}")
            >>> plt.show()
        """
        # Cargar imagen
        image = self.load_image(image_id)

        # Cargar resultados
        snapshot = self.load_result(image_id)

        return image, snapshot

    def load_ground_truths(self, image_id: str) -> List[np.ndarray]:
        """
        Carga todas las anotaciones de ground truth para una imagen desde archivo .mat.

        BSD500 proporciona múltiples anotaciones humanas por imagen (típicamente 5).
        Este método carga TODAS las anotaciones para cálculo de PRI.

        Los archivos .mat se buscan en: src/data/ground_truth/train/{image_id}.mat

        Args:
            image_id: ID de imagen BSD500 (e.g., '12074', '42044', '100075')

        Returns:
            Lista de segmentaciones ground truth (H, W) con labels enteros.
            Típicamente 5 anotaciones. Retorna lista vacía si no se encuentra.

        Raises:
            FileNotFoundError: Si el archivo .mat no existe.

        Example:
            >>> loader = DataLoader()
            >>> gts = loader.load_ground_truths('12074')
            >>> print(f"Encontradas {len(gts)} anotaciones")
            >>> # Encontradas 5 anotaciones
            >>> for i, gt in enumerate(gts):
            ...     print(f"  GT {i+1}: {gt.shape}, {len(np.unique(gt))} segmentos")
        """
        # Construir ruta al archivo .mat de ground truth
        gt_path = self.data_dir / 'ground_truth' / 'train' / f'{image_id}.mat'

        # Validar que el archivo existe
        if not gt_path.exists():
            raise FileNotFoundError(
                f"Ground truth para imagen '{image_id}' no encontrado.\n"
                f"Ruta buscada: {gt_path}\n"
                f"Asegúrate de que el archivo .mat exista en:\n"
                f"  {self.data_dir / 'ground_truth' / 'train' / f'{image_id}.mat'}"
            )

        # Cargar archivo .mat usando scipy
        try:
            mat_data = sio.loadmat(str(gt_path))
        except Exception as e:
            raise IOError(
                f"Error al cargar archivo .mat para imagen '{image_id}'.\n"
                f"Archivo: {gt_path}\n"
                f"Error: {e}"
            )

        # Extraer ground truths del formato BSD500
        # Formato: mat_data['groundTruth'][0, idx][0, 0]['Segmentation']
        ground_truths = []

        try:
            gt_array = mat_data['groundTruth']
            n_annotations = gt_array.shape[1]  # Típicamente 5

            for idx in range(n_annotations):
                # Extraer segmentation de esta anotación
                segmentation = gt_array[0, idx][0, 0]['Segmentation']
                ground_truths.append(segmentation)

        except (KeyError, IndexError) as e:
            raise ValueError(
                f"Formato inesperado en archivo .mat para imagen '{image_id}'.\n"
                f"Archivo: {gt_path}\n"
                f"Error al extraer groundTruth: {e}\n"
                f"Estructura esperada: mat['groundTruth'][0, idx][0, 0]['Segmentation']"
            )

        return ground_truths

    def get_pri_cache_path(self) -> Path:
        """
        Retorna la ruta al archivo de caché PRI.

        El archivo de caché almacena pares de píxeles pre-muestreados y valores p_im
        para evitar costosos recálculos durante la evaluación PRI.

        Returns:
            Path: Ruta al archivo pri_cache.json en src/data/

        Example:
            >>> loader = DataLoader()
            >>> cache_path = loader.get_pri_cache_path()
            >>> print(cache_path)
            >>> # KmeansV3/src/data/pri_cache.json
            >>>
            >>> # Usar con PRI evaluator
            >>> from src.pri import PRIConfig, PRICacheManager
            >>> config = PRIConfig(cache_path=cache_path)
            >>> cache_mgr = PRICacheManager(cache_path)
        """
        return self.data_dir / 'pri_cache.json'
