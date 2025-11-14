"""
Dataset handling and loading utilities.

Modules:
- bsd_dataset: Berkeley Segmentation Dataset (BSD300) handler
"""

from .bsd_dataset import (
    BSD300Dataset,
    BSD300Config,
    PAPER_IMAGE_IDS,
    check_dataset_exists,
    get_expected_dataset_structure
)

__all__ = [
    'BSD300Dataset',
    'BSD300Config',
    'PAPER_IMAGE_IDS',
    'check_dataset_exists',
    'get_expected_dataset_structure',
]
