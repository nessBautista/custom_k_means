"""
PRI (Probabilistic Rand Index) Module

True PRI calculation using sampled pixel pairs with caching.

Implements the paper's TRUE PRI formula:
    PRI(S, {Gk}) = (1/T) Σ [c_im × p_im + (1 - c_im) × (1 - p_im)]

Example:
    >>> from src.pri import TruePRIEvaluator, PRIConfig, PRICacheManager
    >>> from src.data import BSDDataset, BSDConfig
    >>>
    >>> # Setup
    >>> config = PRIConfig(n_samples=10000)
    >>> cache_mgr = PRICacheManager(Path("cache/pri_cache.json"))
    >>> evaluator = TruePRIEvaluator(config, cache_mgr)
    >>>
    >>> # Load ground truths
    >>> dataset = BSDDataset(BSDConfig())
    >>> ground_truths = dataset.load_all_ground_truths('12074')
    >>>
    >>> # Evaluate
    >>> pri_score = evaluator.evaluate('12074', segmentation, ground_truths)
    >>> print(f"PRI: {pri_score:.4f}")
"""

from .config import PRIConfig
from .pri_cache import PRICacheManager
from .pri_evaluator import TruePRIEvaluator

__all__ = [
    'PRIConfig',
    'PRICacheManager',
    'TruePRIEvaluator',
]
