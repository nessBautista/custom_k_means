"""
Módulo de carga de datos para KmeansV3.

Proporciona dataclasses para representar parámetros y resultados guardados,
y un DataLoader para cargar archivos JSON de ejecuciones previas.

Componentes principales:
- NotebookParameters: Parámetros de entrada (k_clusters, velocity_method, etc.)
- NotebookResults: Resultados computados (PRI scores, contour_count, etc.)
- NotebookSnapshot: Contenedor completo con metadata, parameters y results
- DataLoader: Clase para cargar JSONs desde src/data/

Example:
    >>> from src.data_loader import DataLoader
    >>>
    >>> # Cargar resultados guardados
    >>> loader = DataLoader()
    >>> snapshot = loader.load_result('12074')
    >>>
    >>> # Acceder a parámetros
    >>> print(snapshot.parameters.k_clusters)      # 5
    >>> print(snapshot.parameters.velocity_method) # 'curvature_skimage'
    >>>
    >>> # Acceder a resultados
    >>> print(snapshot.results.pri_evolved)        # 0.6594
"""

from .notebook_params import (
    NotebookParameters,
    NotebookResults,
    NotebookSnapshot
)
from .json_loader import DataLoader

__all__ = [
    'NotebookParameters',
    'NotebookResults',
    'NotebookSnapshot',
    'DataLoader'
]
