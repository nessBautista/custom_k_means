"""
Image Experiment Loader

Unified loader for loading BSD300 images and their best experiment results.
Combines BSD300Dataset and GridSearchLoader for easy use in marimo notebooks.

Usage:
    >>> loader = ImageExperimentLoader()
    >>> experiment = loader.load_experiment('42044')
    >>> print(f"Best PRI: {experiment.best_pri}")
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import numpy as np

# Import from our modules
import sys
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir.parent))

from data.bsd_dataset import BSD300Dataset, BSD300Config
from core.grid_search_results import GridSearchLoader


# Paper target PRI values from Table I (Islam et al., 2021)
PAPER_TARGET_PRI_MAP = {
    '12074': 0.7674,   # I1
    '42044': 0.7970,   # I2
    '86016': 0.7868,   # I3
    '147091': 0.7772,  # I4
    '160068': 0.7854,  # I5
    '176035': 0.8064,  # I6
    '176039': 0.7828,  # I7
    '178054': 0.8556,  # I8 (best in paper)
    '216066': 0.7948,  # I9
    '353013': 0.7932   # I10
}


@dataclass
class ImageExperiment:
    """
    Complete experiment data for a single image.

    Contains image data, ground truths, and best parameters from checkpoint.
    """
    # Image identification
    image_id: str

    # Image data
    image: np.ndarray
    ground_truths: List[np.ndarray]
    h: int
    w: int
    pixels: np.ndarray

    # Best parameters (from checkpoint or defaults)
    best_random_seed: int = 42
    best_k: int = 9
    best_iterations: int = 26
    best_dt: float = 0.1
    best_velocity_method: str = 'curvature'
    best_edge_lambda: float = 1.8
    best_curvature_weight: float = 2.5
    best_canny_low: int = 200
    best_canny_high: int = 300
    best_gt_index: int = 0

    # Metrics
    best_pri: float = 0.0
    paper_target_pri: float = 0.797
    gap_to_target: float = 0.0

    # Silhouette config
    silhouette_method: str = "canny_edges"  # canny_edges, labels, or convex_hull
    silhouette_kernel: int = 5
    silhouette_iterations: int = 2
    silhouette_min_area: int = 1000
    silhouette_line_thickness: int = 1

    # Source info
    checkpoint_loaded: bool = False
    checkpoint_path: Optional[str] = None


class ImageExperimentLoader:
    """
    Load image experiments with automatic parameter configuration.

    Loads:
    1. Image and ground truths from BSD300Dataset
    2. Best parameters from checkpoint_{image_id}.json (if exists)
    3. Falls back to reasonable defaults if checkpoint not found
    """

    def __init__(self,
                 results_dir: Optional[Path] = None,
                 dataset_path: Optional[Path] = None):
        """
        Initialize loader.

        Args:
            results_dir: Path to results directory (default: project_root/results)
            dataset_path: Path to BSD500 dataset (default: project_root/src/data/bsd500)
        """
        # Determine project root
        if results_dir is None:
            project_root = Path(__file__).parent.parent.parent
            results_dir = project_root / 'results'

        if dataset_path is None:
            project_root = Path(__file__).parent.parent.parent
            dataset_path = project_root / 'src' / 'data' / 'bsd500'

        self.results_dir = Path(results_dir)
        self.dataset_path = Path(dataset_path)
        self.grid_loader = GridSearchLoader(self.results_dir)

    def load_experiment(self, image_id: str) -> ImageExperiment:
        """
        Load complete experiment data for an image.

        Args:
            image_id: Image ID (e.g., '42044', '12074')

        Returns:
            ImageExperiment with image data and best parameters
        """
        # 1. Load image and ground truths
        image, ground_truths = self._load_image_data(image_id)

        # 2. Get image dimensions and pixels
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)

        # 3. Get paper target PRI
        paper_target_pri = PAPER_TARGET_PRI_MAP.get(image_id, 0.797)

        # 4. Try to load checkpoint
        checkpoint_data = self._load_checkpoint(image_id)

        # 5. Create experiment object
        if checkpoint_data:
            experiment = ImageExperiment(
                image_id=image_id,
                image=image,
                ground_truths=ground_truths,
                h=h,
                w=w,
                pixels=pixels,
                best_random_seed=checkpoint_data['random_seed'],
                best_k=checkpoint_data['k'],
                best_iterations=checkpoint_data['iterations'],
                best_dt=checkpoint_data['dt'],
                best_velocity_method=checkpoint_data['velocity_method'],
                best_edge_lambda=checkpoint_data['edge_lambda'],
                best_curvature_weight=checkpoint_data['curvature_weight'],
                best_canny_low=checkpoint_data['canny_low'],
                best_canny_high=checkpoint_data['canny_high'],
                best_gt_index=checkpoint_data['best_gt_index'],
                best_pri=checkpoint_data['best_pri'],
                paper_target_pri=paper_target_pri,
                gap_to_target=paper_target_pri - checkpoint_data['best_pri'],
                silhouette_method=checkpoint_data.get('silhouette_method', 'canny_edges'),
                silhouette_kernel=checkpoint_data.get('silhouette_kernel', 5),
                silhouette_iterations=checkpoint_data.get('silhouette_iterations', 2),
                silhouette_min_area=checkpoint_data.get('silhouette_min_area', 1000),
                checkpoint_loaded=True,
                checkpoint_path=str(checkpoint_data['path'])
            )
        else:
            # Use defaults
            experiment = ImageExperiment(
                image_id=image_id,
                image=image,
                ground_truths=ground_truths,
                h=h,
                w=w,
                pixels=pixels,
                paper_target_pri=paper_target_pri,
                gap_to_target=paper_target_pri,
                checkpoint_loaded=False,
                checkpoint_path=None
            )

        return experiment

    def _load_image_data(self, image_id: str):
        """Load image and ground truths from BSD300 dataset."""
        config = BSD300Config(
            image_ids=[image_id],
            split='train',  # Will auto-search across splits
            dataset_path=str(self.dataset_path)
        )
        dataset = BSD300Dataset(config)

        if len(dataset) == 0:
            raise ValueError(f"Image {image_id} not found in dataset")

        _, image, ground_truths = dataset[0]
        return image, ground_truths

    def _load_checkpoint(self, image_id: str):
        """
        Try to load latest checkpoint for image.

        Tries in order:
        1. results/{image_id}/checkpoint_{image_id}.json (exact match)
        2. results/{image_id}/checkpoint_{image_id}_*.json (timestamped, uses latest)
        3. Returns None if not found
        """
        # Try exact name first (no timestamp)
        checkpoint_path = self.results_dir / image_id / f'checkpoint_{image_id}.json'

        if not checkpoint_path.exists():
            # Look for timestamped versions: checkpoint_86016_*.json
            image_result_dir = self.results_dir / image_id
            if image_result_dir.exists():
                # Find all matching checkpoint files
                checkpoint_files = list(image_result_dir.glob(f'checkpoint_{image_id}_*.json'))
                if checkpoint_files:
                    # Get the most recent one by modification time
                    checkpoint_path = max(checkpoint_files, key=lambda p: p.stat().st_mtime)
                    print(f"Found timestamped checkpoint: {checkpoint_path.name}")
                else:
                    return None
            else:
                return None

        try:
            experiment = self.grid_loader.load_by_path(checkpoint_path)

            if not experiment.best_result:
                return None

            params = experiment.best_result.parameters
            metrics = experiment.best_result.metrics

            # Extract silhouette config if available
            sil_method = "canny_edges"
            sil_kernel = 5
            sil_iterations = 2
            sil_min_area = 1000

            if experiment.silhouette_config:
                sil_method = experiment.silhouette_config.method
                sil_kernel = experiment.silhouette_config.closing_kernel
                sil_iterations = experiment.silhouette_config.closing_iterations
                sil_min_area = experiment.silhouette_config.min_area

            return {
                'random_seed': params.random_seed,
                'k': params.k,
                'iterations': params.iterations,
                'dt': params.dt,
                'velocity_method': params.velocity_method,
                'edge_lambda': params.edge_lambda,
                'curvature_weight': params.curvature_weight,
                'canny_low': params.canny_low,
                'canny_high': params.canny_high,
                'best_gt_index': metrics.best_gt_index,
                'best_pri': metrics.best_gt_pri,
                'silhouette_method': sil_method,
                'silhouette_kernel': sil_kernel,
                'silhouette_iterations': sil_iterations,
                'silhouette_min_area': sil_min_area,
                'path': checkpoint_path
            }

        except Exception as e:
            print(f"Warning: Could not load checkpoint for {image_id}: {e}")
            return None

    def list_available_images(self) -> List[str]:
        """List all images with checkpoint files."""
        available = []
        for image_id in PAPER_TARGET_PRI_MAP.keys():
            checkpoint_path = self.results_dir / image_id / f'checkpoint_{image_id}.json'
            if checkpoint_path.exists():
                available.append(image_id)
        return available
