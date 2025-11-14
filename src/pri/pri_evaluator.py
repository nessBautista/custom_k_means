"""
True Probabilistic Rand Index (PRI) Evaluator

Implements the TRUE PRI formula from the paper using Strategy A (Sampled Pairs):
    PRI(S, {Gk}) = (1/T) Σ [c_im × p_im + (1 - c_im) × (1 - p_im)]

Where:
- S = segmentation
- {Gk} = set of ground truths
- c_im = 1 if pixels i,m have same label in S, else 0
- p_im = probability that i,m have same label across {Gk}
- T = total number of sampled pairs

Reference: Islam et al. (2021), Section III.C - Probabilistic Rand Index
"""

from typing import List, Tuple, Optional
import numpy as np

from .config import PRIConfig
from .pri_cache import PRICacheManager


class TruePRIEvaluator:
    """
    True PRI evaluator using sampled pixel pairs (Strategy A).

    This implementation:
    1. Samples N random pixel pairs (e.g., 10,000)
    2. Computes p_im for each pair across all ground truths
    3. Caches p_im values to avoid recomputation
    4. Applies TRUE PRI formula: c_im × p_im + (1-c_im) × (1-p_im)

    Example:
        >>> from src.pri import TruePRIEvaluator, PRIConfig, PRICacheManager
        >>>
        >>> config = PRIConfig(n_samples=10000)
        >>> cache_mgr = PRICacheManager(Path("cache/pri_cache.json"))
        >>> evaluator = TruePRIEvaluator(config, cache_mgr)
        >>>
        >>> # Evaluate segmentation against 5 ground truths
        >>> pri_score = evaluator.evaluate("12074", segmentation, ground_truths)
        >>> print(f"PRI: {pri_score:.4f}")
    """

    def __init__(self, config: PRIConfig, cache_manager: Optional[PRICacheManager] = None):
        """
        Initialize True PRI evaluator.

        Args:
            config: PRI configuration
            cache_manager: Cache manager (creates default if None)
        """
        self.config = config

        if cache_manager is None and config.use_cache:
            cache_manager = PRICacheManager(config.cache_path)

        self.cache_manager = cache_manager

    def _sample_pixel_pairs(self, h: int, w: int) -> np.ndarray:
        """
        Sample N random pixel pairs from image.

        Uses fixed random seed for reproducibility - same image
        always gets same sampled pairs.

        Args:
            h: Image height
            w: Image width

        Returns:
            pairs: Array of shape (N, 2) with pixel indices (i, m)
        """
        n_pixels = h * w
        rng = np.random.RandomState(seed=self.config.random_seed)

        # Sample N random pairs with replacement
        pairs = rng.randint(0, n_pixels, size=(self.config.n_samples, 2))

        return pairs

    def _compute_p_im_matrix(
        self,
        ground_truths: List[np.ndarray],
        pairs: np.ndarray
    ) -> np.ndarray:
        """
        Compute p_im values for sampled pixel pairs.

        For each pair (i, m):
            p_im = (number of GTs where label[i] == label[m]) / total GTs

        Args:
            ground_truths: List of ground truth segmentations, each (H, W)
            pairs: Sampled pixel pairs, shape (N, 2)

        Returns:
            p_values: Array of p_im values, shape (N,)
        """
        n_gts = len(ground_truths)
        n_pairs = len(pairs)

        # Flatten ground truths for indexing
        gts_flat = [gt.flatten() for gt in ground_truths]

        p_values = np.zeros(n_pairs, dtype=np.float32)

        for pair_idx, (i, m) in enumerate(pairs):
            # Count how many GTs have same label for this pair
            agreement_count = 0

            for gt_flat in gts_flat:
                if gt_flat[i] == gt_flat[m]:
                    agreement_count += 1

            # p_im = probability of agreement
            p_values[pair_idx] = agreement_count / n_gts

        return p_values

    def _compute_c_im_values(
        self,
        segmentation: np.ndarray,
        pairs: np.ndarray
    ) -> np.ndarray:
        """
        Compute c_im values for segmentation.

        c_im = 1 if pixels i and m have same label in segmentation, else 0

        Args:
            segmentation: Segmentation labels, shape (H, W)
            pairs: Pixel pairs, shape (N, 2)

        Returns:
            c_values: Array of c_im values, shape (N,)
        """
        seg_flat = segmentation.flatten()

        c_values = np.zeros(len(pairs), dtype=np.float32)

        for pair_idx, (i, m) in enumerate(pairs):
            c_values[pair_idx] = 1.0 if seg_flat[i] == seg_flat[m] else 0.0

        return c_values

    def evaluate(
        self,
        image_id: str,
        segmentation: np.ndarray,
        ground_truths: List[np.ndarray]
    ) -> float:
        """
        Compute True PRI score for segmentation.

        Formula: PRI = (1/T) Σ [c_im × p_im + (1 - c_im) × (1 - p_im)]

        Args:
            image_id: Image identifier for caching (e.g., '12074')
            segmentation: Segmentation labels, shape (H, W)
            ground_truths: List of ground truth segmentations, each (H, W)

        Returns:
            pri_score: PRI score in [0, 1] range

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if segmentation.ndim != 2:
            raise ValueError(f"Segmentation must be 2D, got shape {segmentation.shape}")

        if not ground_truths:
            raise ValueError("At least one ground truth required")

        h, w = segmentation.shape

        # Check if ground truths have same shape
        for gt in ground_truths:
            if gt.shape != (h, w):
                raise ValueError(
                    f"Ground truth shape {gt.shape} doesn't match segmentation {(h, w)}"
                )

        # Try to load cached p_im data
        pairs = None
        p_values = None

        if self.config.use_cache and self.cache_manager is not None:
            cached_data = self.cache_manager.get_p_im_data(image_id)

            if cached_data is not None:
                # Verify cache is valid
                if cached_data["n_ground_truths"] == len(ground_truths):
                    pairs = np.array(cached_data["pairs"])
                    p_values = np.array(cached_data["p_values"])

        # If not cached, compute p_im matrix
        if pairs is None or p_values is None:
            # Sample pixel pairs
            pairs = self._sample_pixel_pairs(h, w)

            # Compute p_im for each pair
            p_values = self._compute_p_im_matrix(ground_truths, pairs)

            # Cache the results
            if self.config.use_cache and self.cache_manager is not None:
                self.cache_manager.save_p_im_data(
                    image_id=image_id,
                    pairs=pairs,
                    p_values=p_values,
                    n_ground_truths=len(ground_truths),
                    image_shape=(h, w)
                )

        # Compute c_im for current segmentation
        c_values = self._compute_c_im_values(segmentation, pairs)

        # Apply PRI formula: c_im × p_im + (1 - c_im) × (1 - p_im)
        contributions = c_values * p_values + (1 - c_values) * (1 - p_values)

        # Average over all pairs
        pri_score = np.mean(contributions)

        return float(pri_score)

    def evaluate_batch(
        self,
        image_id: str,
        segmentations: List[np.ndarray],
        ground_truths: List[np.ndarray]
    ) -> List[float]:
        """
        Evaluate multiple segmentations efficiently.

        Reuses cached p_im matrix for all segmentations.

        Args:
            image_id: Image identifier
            segmentations: List of segmentation labels
            ground_truths: List of ground truth segmentations

        Returns:
            pri_scores: List of PRI scores
        """
        return [
            self.evaluate(image_id, seg, ground_truths)
            for seg in segmentations
        ]
