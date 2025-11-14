"""
Módulo de visualización para KmeansV3.

Proporciona utilidades para mostrar imágenes en grids y cargar
imágenes del dataset BSD500 para análisis de segmentación con K-Means.
"""

from .image_grid import plot_image_grid, load_and_display_bsd_images, plot_kmeans_results, plot_levelset_results, plot_cluster_distances_grid, plot_evolution_results, plot_segmentation_comparison_batch, plot_silhouette_comparison

__all__ = ['plot_image_grid', 'load_and_display_bsd_images', 'plot_kmeans_results', 'plot_levelset_results', 'plot_cluster_distances_grid', 'plot_evolution_results', 'plot_segmentation_comparison_batch', 'plot_silhouette_comparison']
