"""
Probabilistic Rand Index (PRI) Evaluation

Implementation of Stage 5 of the customized k-means pipeline.

Paper Reference: Islam et al. (2021), Section IV - Experimental Results
PRI is used to quantitatively evaluate segmentation quality against ground truth.

This module implements PRI evaluation for comparing segmentation results
against multiple human annotations (BSD500 provides ~5 annotations per image).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import adjusted_rand_score


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class PRIConfig:
    """
    Configuration for PRI evaluation.

    The Probabilistic Rand Index measures segmentation quality by comparing
    pixel pair agreements between segmentation and ground truth annotations.
    """
    use_adjusted_score: bool = True
    """Use adjusted rand score (accounts for chance agreements).
    True = adjusted_rand_score, False = rand_score."""

    normalize_to_unit: bool = True
    """Normalize scores to [0, 1] range.
    Adjusted rand score ranges from [-1, 1], normalization maps to [0, 1]."""

    require_shape_match: bool = True
    """Require ground truth to match segmentation shape.
    If True, skip ground truths with different dimensions."""

    def __post_init__(self):
        """Validate configuration parameters."""
        pass  # All boolean parameters, no validation needed


# ============================================================================
# Results
# ============================================================================

@dataclass
class PRIResult:
    """
    Results from PRI evaluation.

    Contains overall PRI score and individual scores for each ground truth.
    """
    pri_score: float
    """Overall PRI score (mean across all ground truths). Range: [0, 1]."""

    individual_scores: List[float]
    """PRI score for each ground truth annotation."""

    n_ground_truths: int
    """Number of ground truth annotations used in evaluation."""

    best_score: float
    """Best individual PRI score among all ground truths."""

    worst_score: float
    """Worst individual PRI score among all ground truths."""

    std_dev: float
    """Standard deviation of scores across ground truths."""

    def __str__(self) -> str:
        """String representation of results."""
        return (
            f"PRI Score: {self.pri_score:.4f} "
            f"(mean of {self.n_ground_truths} GTs, "
            f"std={self.std_dev:.4f}, "
            f"range=[{self.worst_score:.4f}, {self.best_score:.4f}])"
        )

    def summary(self) -> Dict[str, float]:
        """Get summary statistics as dictionary."""
        return {
            'pri_score': self.pri_score,
            'n_ground_truths': self.n_ground_truths,
            'best_score': self.best_score,
            'worst_score': self.worst_score,
            'std_dev': self.std_dev
        }


# ============================================================================
# Abstract Base Class (Interface)
# ============================================================================

class BasePRIEvaluator(ABC):
    """
    Abstract base class for PRI evaluation implementations.

    This interface allows for different PRI computation methods:
    - SklearnPRIEvaluator: Uses sklearn's adjusted_rand_score (production)
    - CustomPRIEvaluator: From-scratch implementation (future)
    """

    def __init__(self, config: PRIConfig):
        """
        Initialize PRI evaluator.

        Args:
            config: Evaluation configuration
        """
        self.config = config

    @abstractmethod
    def evaluate(
        self,
        segmentation: np.ndarray,
        ground_truths: List[np.ndarray]
    ) -> PRIResult:
        """
        Compute PRI score against multiple ground truth annotations.

        Args:
            segmentation: Segmentation labels, shape (H, W)
            ground_truths: List of ground truth segmentations, each (H, W)

        Returns:
            result: PRIResult with overall score and statistics
        """
        pass

    @abstractmethod
    def evaluate_single(
        self,
        segmentation: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Compute PRI score against single ground truth annotation.

        Args:
            segmentation: Segmentation labels, shape (H, W)
            ground_truth: Single ground truth segmentation, shape (H, W)

        Returns:
            score: PRI score in [0, 1] range
        """
        pass


# ============================================================================
# Sklearn Implementation (Production)
# ============================================================================

class SklearnPRIEvaluator(BasePRIEvaluator):
    """
    PRI evaluation using sklearn's adjusted_rand_score.

    This is the primary implementation that uses sklearn's efficient
    adjusted Rand index computation and averages across multiple ground truths.

    Based on: research/intuition/custom_kmeans/evaluation.py
              research/custom_k_means/paper_results_v2.py

    Formula from paper:
    PRI(S, {Gk}) = (1/T) Σ [c_im × p_im + (1 - c_im) × (1 - p_im)]

    Where:
    - S = segmented image
    - {Gk} = set of ground truth segmentations
    - c_im = 1 if pixels i and m have same label in S, else 0
    - p_im = probability that pixels i and m have same label across {Gk}
    - T = total number of pixel pairs

    Note: We use adjusted_rand_score as an approximation to true PRI,
    which accounts for chance agreements and provides similar performance.
    """

    def __init__(self, config: PRIConfig):
        """
        Initialize sklearn PRI evaluator.

        Args:
            config: PRI evaluation configuration
        """
        super().__init__(config)

    def evaluate(
        self,
        segmentation: np.ndarray,
        ground_truths: List[np.ndarray]
    ) -> PRIResult:
        """
        Compute PRI score against multiple ground truth annotations.

        Implements the paper's evaluation procedure:
        1. Compare segmentation against each ground truth
        2. Compute adjusted Rand score for each comparison
        3. Average scores across all ground truths

        Args:
            segmentation: Cluster labels from pipeline, shape (H, W)
            ground_truths: List of GT segmentations, each shape (H, W)

        Returns:
            result: PRIResult with mean PRI and statistics

        Example:
            >>> evaluator = SklearnPRIEvaluator(PRIConfig())
            >>> result = evaluator.evaluate(labels, ground_truths)
            >>> print(f"PRI: {result.pri_score:.4f}")
        """
        if not ground_truths:
            return PRIResult(
                pri_score=0.0,
                individual_scores=[],
                n_ground_truths=0,
                best_score=0.0,
                worst_score=0.0,
                std_dev=0.0
            )

        # Validate segmentation shape
        if segmentation.ndim != 2:
            raise ValueError(
                f"Segmentation must be 2D (H, W), got shape {segmentation.shape}"
            )

        # Compute score for each ground truth
        individual_scores = []

        for gt in ground_truths:
            # Check shape compatibility
            if self.config.require_shape_match and gt.shape != segmentation.shape:
                continue

            score = self.evaluate_single(segmentation, gt)
            individual_scores.append(score)

        if not individual_scores:
            return PRIResult(
                pri_score=0.0,
                individual_scores=[],
                n_ground_truths=0,
                best_score=0.0,
                worst_score=0.0,
                std_dev=0.0
            )

        # Compute statistics
        pri_score = np.mean(individual_scores)
        best_score = np.max(individual_scores)
        worst_score = np.min(individual_scores)
        std_dev = np.std(individual_scores)

        return PRIResult(
            pri_score=pri_score,
            individual_scores=individual_scores,
            n_ground_truths=len(individual_scores),
            best_score=best_score,
            worst_score=worst_score,
            std_dev=std_dev
        )

    def evaluate_single(
        self,
        segmentation: np.ndarray,
        ground_truth: np.ndarray
    ) -> float:
        """
        Compute PRI score against single ground truth.

        Uses sklearn's adjusted_rand_score and normalizes to [0, 1] range.

        Args:
            segmentation: Segmentation labels, shape (H, W)
            ground_truth: Ground truth labels, shape (H, W)

        Returns:
            score: Normalized PRI score in [0, 1]

        Notes:
            - Adjusted rand score ranges from -1 (worst) to 1 (perfect)
            - We normalize: (score + 1) / 2 to get [0, 1] range
            - Score = 0.5 indicates random labeling
            - Score > 0.5 indicates better than random
        """
        # Flatten to 1D for sklearn
        seg_flat = segmentation.flatten()
        gt_flat = ground_truth.flatten()

        # Compute adjusted Rand score
        if self.config.use_adjusted_score:
            score = adjusted_rand_score(seg_flat, gt_flat)
        else:
            # If using non-adjusted (not recommended for PRI)
            from sklearn.metrics import rand_score
            score = rand_score(seg_flat, gt_flat)

        # Normalize to [0, 1] range
        if self.config.normalize_to_unit:
            # Adjusted rand score is in [-1, 1]
            # Map to [0, 1]: (score + 1) / 2
            score = (score + 1.0) / 2.0

        return float(score)


# ============================================================================
# Helper Functions
# ============================================================================

def compute_pri_for_k_range(
    pipeline_class,
    image: np.ndarray,
    ground_truths: List[np.ndarray],
    k_range: Tuple[int, int] = (2, 10),
    evaluator: Optional[BasePRIEvaluator] = None,
    verbose: bool = True
) -> Dict[int, Dict]:
    """
    Compute PRI scores for multiple k values to find optimal clustering.

    Useful for reproducing paper's results which test multiple k values
    to find the best segmentation.

    Args:
        pipeline_class: CustomizedKMeansSegmentation class
        image: Input RGB image (H, W, 3)
        ground_truths: List of ground truth segmentations
        k_range: (k_min, k_max) range of k values to test
        evaluator: PRI evaluator instance (creates default if None)
        verbose: Whether to print progress

    Returns:
        results: Dict mapping k -> {pri_score, pri_result, pipeline_result}

    Example:
        >>> from src import CustomizedKMeansSegmentation
        >>> results = compute_pri_for_k_range(
        ...     CustomizedKMeansSegmentation,
        ...     image, ground_truths,
        ...     k_range=(3, 7)
        ... )
        >>> best_k = max(results, key=lambda k: results[k]['pri_score'])
    """
    if evaluator is None:
        evaluator = SklearnPRIEvaluator(PRIConfig())

    k_min, k_max = k_range
    results = {}

    if verbose:
        print(f"Testing k values from {k_min} to {k_max}...")

    for k in range(k_min, k_max + 1):
        if verbose:
            print(f"  k={k}...", end=" ")

        # Run pipeline with this k value
        # (Would need to modify pipeline config - simplified here)
        from .kmeans import KMeansConfig
        from .level_set import LevelSetConfig
        from src.pipeline import PipelineConfig

        config = PipelineConfig(
            kmeans_config=KMeansConfig(n_clusters=k)
        )

        pipeline = pipeline_class(config)
        pipeline_result = pipeline.run(image, verbose=False)

        # Evaluate
        labels = pipeline_result.get_final_labels()
        pri_result = evaluator.evaluate(labels, ground_truths)

        results[k] = {
            'pri_score': pri_result.pri_score,
            'pri_result': pri_result,
            'pipeline_result': pipeline_result
        }

        if verbose:
            print(f"PRI = {pri_result.pri_score:.4f}")

    return results


def find_best_k(
    results: Dict[int, Dict],
    criterion: str = 'pri_score'
) -> int:
    """
    Find best k value from evaluation results.

    Args:
        results: Results from compute_pri_for_k_range
        criterion: Evaluation criterion (default: 'pri_score')

    Returns:
        best_k: k value with highest criterion score

    Example:
        >>> results = compute_pri_for_k_range(...)
        >>> best_k = find_best_k(results)
        >>> print(f"Best k: {best_k}")
    """
    if not results:
        raise ValueError("No results provided")

    best_k = max(results.keys(), key=lambda k: results[k][criterion])
    return best_k


def create_pri_report(
    image_id: str,
    results: Dict[int, Dict],
    verbose: bool = True
) -> str:
    """
    Create formatted PRI evaluation report.

    Args:
        image_id: Image identifier
        results: Results from compute_pri_for_k_range
        verbose: Include detailed statistics

    Returns:
        report: Formatted string report

    Example:
        >>> results = compute_pri_for_k_range(...)
        >>> report = create_pri_report('12074', results)
        >>> print(report)
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"PRI Evaluation Report: Image {image_id}")
    lines.append("=" * 70)
    lines.append("")

    # Table header
    lines.append(f"{'k':<10}{'PRI Score':<15}{'Std Dev':<15}{'Best':<15}{'Worst':<15}")
    lines.append("-" * 70)

    # Table rows
    for k in sorted(results.keys()):
        pri_res = results[k]['pri_result']
        lines.append(
            f"{k:<10}"
            f"{pri_res.pri_score:<15.4f}"
            f"{pri_res.std_dev:<15.4f}"
            f"{pri_res.best_score:<15.4f}"
            f"{pri_res.worst_score:<15.4f}"
        )

    # Best result
    best_k = find_best_k(results)
    best_pri = results[best_k]['pri_score']

    lines.append("")
    lines.append("=" * 70)
    lines.append(f"Best k: {best_k} with PRI Score: {best_pri:.4f}")
    lines.append("=" * 70)

    return "\n".join(lines)
