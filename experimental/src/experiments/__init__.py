"""
Experiment runners and parameter tuning utilities.

Modules:
- runner: ExperimentRunner for reproducing paper results and grid search
- image_experiment_loader: Unified loader for BSD300 images and experiment results
"""

from .image_experiment_loader import ImageExperimentLoader, ImageExperiment, PAPER_TARGET_PRI_MAP

__all__ = [
    'ImageExperimentLoader',
    'ImageExperiment',
    'PAPER_TARGET_PRI_MAP'
]

# Imports will be added as modules are implemented
# from .runner import ExperimentRunner, ExperimentConfig
