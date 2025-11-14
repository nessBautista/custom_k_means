"""
Grid Search Results Loader

Load and manage grid search experiment results for K-means + Level Set segmentation.

Classes:
    - GridSearchResult: Individual parameter configuration and its metrics
    - GridSearchMetadata: Experiment metadata
    - GridSearchConfig: Grid parameter ranges
    - SilhouetteConfig: Silhouette extraction configuration
    - GridSearchExperiment: Complete experiment data
    - GridSearchLoader: Load and query results by image ID

Example:
    >>> from core.grid_search_results import GridSearchLoader
    >>> loader = GridSearchLoader()
    >>> experiment = loader.load_by_image_id('42044')
    >>> print(f"Best PRI: {experiment.best_result.metrics.best_gt_pri}")
    >>> params = loader.get_best_parameters('42044')
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import json


@dataclass
class GridSearchMetrics:
    """Performance metrics for a grid search result."""
    best_gt_pri: float
    best_gt_index: int
    avg_pri: float
    pri_std: float = 0.0
    gap_to_target: Optional[float] = None
    improvement: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSearchMetrics':
        """Create from dictionary."""
        return cls(
            best_gt_pri=data['best_gt_pri'],
            best_gt_index=data['best_gt_index'],
            avg_pri=data['avg_pri'],
            pri_std=data.get('pri_std', 0.0),
            gap_to_target=data.get('gap_to_target'),
            improvement=data.get('improvement')
        )


@dataclass
class GridSearchParameters:
    """Parameter configuration for segmentation pipeline."""
    random_seed: int = 42
    k: int = 9
    iterations: int = 26
    dt: float = 0.1
    velocity_method: str = 'curvature'
    edge_lambda: float = 1.8
    curvature_weight: float = 2.5
    canny_low: int = 200
    canny_high: int = 300

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSearchParameters':
        """Create from dictionary."""
        return cls(
            random_seed=data.get('random_seed', 42),
            k=data['k'],
            iterations=data['iterations'],
            dt=data['dt'],
            velocity_method=data['velocity_method'],
            edge_lambda=data['edge_lambda'],
            curvature_weight=data['curvature_weight'],
            canny_low=data['canny_low'],
            canny_high=data['canny_high']
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'random_seed': self.random_seed,
            'k': self.k,
            'iterations': self.iterations,
            'dt': self.dt,
            'velocity_method': self.velocity_method,
            'edge_lambda': self.edge_lambda,
            'curvature_weight': self.curvature_weight,
            'canny_low': self.canny_low,
            'canny_high': self.canny_high
        }


@dataclass
class GridSearchResult:
    """Single grid search result with parameters and metrics."""
    parameters: GridSearchParameters
    metrics: GridSearchMetrics
    silhouette_method: Optional[str] = None
    silhouette_kernel: Optional[int] = None
    silhouette_iterations: Optional[int] = None
    silhouette_min_area: Optional[int] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSearchResult':
        """Create from dictionary (legacy format support)."""
        # Extract parameters
        params = GridSearchParameters.from_dict(data)

        # Extract metrics
        metrics = GridSearchMetrics(
            best_gt_pri=data['best_gt_pri'],
            best_gt_index=data['best_gt_index'],
            avg_pri=data['avg_pri'],
            pri_std=data.get('pri_std', 0.0)
        )

        return cls(
            parameters=params,
            metrics=metrics,
            silhouette_method=data.get('silhouette_method'),
            silhouette_kernel=data.get('silhouette_kernel'),
            silhouette_iterations=data.get('silhouette_iterations'),
            silhouette_min_area=data.get('silhouette_min_area')
        )


@dataclass
class GridSearchMetadata:
    """Experiment metadata."""
    image_id: str
    iteration: int = 1
    iteration_name: str = ""
    timestamp: Optional[datetime] = None
    paper_target_pri: float = 0.797
    previous_best_pri: Optional[float] = None
    total_combinations: int = 0
    evaluation_strategy: str = "best_gt"
    description: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSearchMetadata':
        """Create from dictionary."""
        timestamp = None
        if 'timestamp' in data and data['timestamp']:
            try:
                timestamp = datetime.fromisoformat(data['timestamp'])
            except (ValueError, TypeError):
                timestamp = None

        return cls(
            image_id=data['image_id'],
            iteration=data.get('iteration', 1),
            iteration_name=data.get('iteration_name', ''),
            timestamp=timestamp,
            paper_target_pri=data.get('paper_target_pri', 0.797),
            previous_best_pri=data.get('previous_best_pri'),
            total_combinations=data.get('total_combinations', 0),
            evaluation_strategy=data.get('evaluation_strategy', 'best_gt'),
            description=data.get('description', '')
        )


@dataclass
class GridConfig:
    """Grid search parameter ranges."""
    random_seed_values: List[int] = field(default_factory=lambda: [42])
    k_values: List[int] = field(default_factory=lambda: [9])
    iterations_values: List[int] = field(default_factory=lambda: [26])
    dt_values: List[float] = field(default_factory=lambda: [0.1])
    velocity_methods: List[str] = field(default_factory=lambda: ['curvature'])
    edge_lambda_values: List[float] = field(default_factory=lambda: [1.8])
    curvature_weight_values: List[float] = field(default_factory=lambda: [2.5])
    canny_low_values: List[int] = field(default_factory=lambda: [200])
    canny_high_values: List[int] = field(default_factory=lambda: [300])

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridConfig':
        """Create from dictionary."""
        return cls(
            random_seed_values=data.get('random_seed_values', [42]),
            k_values=data.get('k_values', [9]),
            iterations_values=data.get('iterations_values', [26]),
            dt_values=data.get('dt_values', [0.1]),
            velocity_methods=data.get('velocity_methods', ['curvature']),
            edge_lambda_values=data.get('edge_lambda_values', [1.8]),
            curvature_weight_values=data.get('curvature_weight_values', [2.5]),
            canny_low_values=data.get('canny_low_values', [200]),
            canny_high_values=data.get('canny_high_values', [300])
        )


@dataclass
class SilhouetteConfig:
    """Silhouette extraction configuration."""
    method: str = "canny_edges"
    closing_kernel: int = 5
    closing_iterations: int = 2
    min_area: int = 1000
    line_thickness: int = 1
    label_kernel: Optional[int] = None
    smoothing_epsilon: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SilhouetteConfig':
        """Create from dictionary."""
        return cls(
            method=data.get('method', 'canny_edges'),
            closing_kernel=data.get('closing_kernel', 5),
            closing_iterations=data.get('closing_iterations', 2),
            min_area=data.get('min_area', 1000),
            line_thickness=data.get('line_thickness', 1),
            label_kernel=data.get('label_kernel'),
            smoothing_epsilon=data.get('smoothing_epsilon')
        )


@dataclass
class GridSearchExperiment:
    """Complete grid search experiment data."""
    results: List[GridSearchResult]
    metadata: Optional[GridSearchMetadata] = None
    grid_config: Optional[GridConfig] = None
    silhouette_config: Optional[SilhouetteConfig] = None
    best_result: Optional[GridSearchResult] = None

    def __post_init__(self):
        """Calculate best result if not provided."""
        if self.best_result is None and self.results:
            self.best_result = max(self.results,
                                   key=lambda r: r.metrics.best_gt_pri)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GridSearchExperiment':
        """Create from dictionary (supports both legacy and new format)."""
        # Parse results array
        results = [GridSearchResult.from_dict(r) for r in data.get('results', [])]

        # Parse metadata if present
        metadata = None
        if 'metadata' in data:
            metadata = GridSearchMetadata.from_dict(data['metadata'])

        # Parse grid config if present
        grid_config = None
        if 'grid_config' in data:
            grid_config = GridConfig.from_dict(data['grid_config'])

        # Parse silhouette config if present
        silhouette_config = None
        if 'silhouette_config' in data:
            silhouette_config = SilhouetteConfig.from_dict(data['silhouette_config'])

        # Parse best result if present
        best_result = None
        if 'best_result' in data:
            params = GridSearchParameters.from_dict(data['best_result']['parameters'])
            metrics = GridSearchMetrics.from_dict(data['best_result']['metrics'])
            best_result = GridSearchResult(parameters=params, metrics=metrics)

        return cls(
            results=results,
            metadata=metadata,
            grid_config=grid_config,
            silhouette_config=silhouette_config,
            best_result=best_result
        )

    def get_top_n(self, n: int = 10) -> List[GridSearchResult]:
        """Get top N results by best_gt_pri."""
        return sorted(self.results,
                     key=lambda r: r.metrics.best_gt_pri,
                     reverse=True)[:n]

    def filter_by_params(self, **kwargs) -> List[GridSearchResult]:
        """Filter results by parameter values.

        Example:
            results = experiment.filter_by_params(k=9, dt=0.1)
        """
        filtered = []
        for result in self.results:
            match = True
            for key, value in kwargs.items():
                if hasattr(result.parameters, key):
                    if getattr(result.parameters, key) != value:
                        match = False
                        break
            if match:
                filtered.append(result)
        return filtered


class GridSearchLoader:
    """Load and query grid search experiment results."""

    def __init__(self, results_dir: Optional[Path] = None):
        """Initialize loader with results directory.

        Args:
            results_dir: Path to results directory.
                        Defaults to project_root/results/
        """
        if results_dir is None:
            # Default to project root / results
            current = Path(__file__).parent
            project_root = current.parent.parent
            results_dir = project_root / 'results'

        self.results_dir = Path(results_dir)

    def load_by_image_id(self, image_id: str,
                        experiment_type: str = 'best_gt') -> GridSearchExperiment:
        """Load latest results for a given image ID.

        Args:
            image_id: Image identifier (e.g., '42044', '12074')
            experiment_type: Experiment subdirectory (default: 'best_gt')

        Returns:
            GridSearchExperiment with all results

        Raises:
            FileNotFoundError: If checkpoint file doesn't exist
        """
        checkpoint_path = (self.results_dir / image_id /
                          f"{image_id}_{experiment_type}" / 'checkpoint.json')

        return self.load_by_path(checkpoint_path)

    def load_by_path(self, path: Path) -> GridSearchExperiment:
        """Load results from a specific checkpoint file.

        Args:
            path: Path to checkpoint.json file

        Returns:
            GridSearchExperiment with all results

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        with open(path, 'r') as f:
            data = json.load(f)

        return GridSearchExperiment.from_dict(data)

    def load_iteration(self, image_id: str, iteration: int,
                      experiment_type: str = 'best_gt') -> GridSearchExperiment:
        """Load specific iteration results.

        Args:
            image_id: Image identifier
            iteration: Iteration number
            experiment_type: Experiment subdirectory

        Returns:
            GridSearchExperiment for that iteration
        """
        checkpoint_path = (self.results_dir / image_id /
                          f"{image_id}_{experiment_type}" /
                          f'checkpoint_iteration{iteration}.json')

        return self.load_by_path(checkpoint_path)

    def get_best_parameters(self, image_id: str,
                           experiment_type: str = 'best_gt') -> GridSearchParameters:
        """Extract best parameters for an image.

        Args:
            image_id: Image identifier
            experiment_type: Experiment subdirectory

        Returns:
            GridSearchParameters with best configuration
        """
        experiment = self.load_by_image_id(image_id, experiment_type)
        if experiment.best_result:
            return experiment.best_result.parameters
        return experiment.get_top_n(1)[0].parameters

    def compare_iterations(self, image_id: str,
                          max_iterations: int = 10) -> Dict[int, GridSearchResult]:
        """Compare best results across iterations.

        Args:
            image_id: Image identifier
            max_iterations: Maximum iteration to check

        Returns:
            Dictionary mapping iteration number to best result
        """
        results = {}
        for i in range(1, max_iterations + 1):
            try:
                experiment = self.load_iteration(image_id, i)
                results[i] = experiment.best_result or experiment.get_top_n(1)[0]
            except FileNotFoundError:
                # No more iterations
                break
        return results

    def list_available_images(self) -> List[str]:
        """List all image IDs with results.

        Returns:
            List of image ID strings
        """
        if not self.results_dir.exists():
            return []

        images = []
        for path in self.results_dir.iterdir():
            if path.is_dir() and path.name.isdigit():
                images.append(path.name)
        return sorted(images)

    def export_best_to_template(self, image_id: str,
                               experiment_type: str = 'best_gt') -> Dict[str, Any]:
        """Export best parameters in template-ready format.

        Args:
            image_id: Image identifier
            experiment_type: Experiment subdirectory

        Returns:
            Dictionary with template constants
        """
        experiment = self.load_by_image_id(image_id, experiment_type)
        params = experiment.best_result.parameters if experiment.best_result else experiment.get_top_n(1)[0].parameters
        metrics = experiment.best_result.metrics if experiment.best_result else experiment.get_top_n(1)[0].metrics

        return {
            'IMAGE_ID': image_id,
            'PAPER_TARGET_PRI': experiment.metadata.paper_target_pri if experiment.metadata else 0.797,
            'BEST_GT_INDEX': metrics.best_gt_index,
            'K': params.k,
            'ITERATIONS': params.iterations,
            'DT': params.dt,
            'VELOCITY_METHOD': params.velocity_method,
            'EDGE_LAMBDA': params.edge_lambda,
            'CURVATURE_WEIGHT': params.curvature_weight,
            'CANNY_LOW': params.canny_low,
            'CANNY_HIGH': params.canny_high,
            'BEST_PRI': metrics.best_gt_pri,
            'AVG_PRI': metrics.avg_pri,
            'GAP_TO_TARGET': metrics.gap_to_target if metrics.gap_to_target else (experiment.metadata.paper_target_pri - metrics.best_gt_pri) if experiment.metadata else None
        }
