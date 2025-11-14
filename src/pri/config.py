"""
PRI Configuration

Configuration for True Probabilistic Rand Index calculation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union


@dataclass
class PRIConfig:
    """
    Configuration for True PRI calculation using sampled pixel pairs.

    Attributes:
        use_sampling: Whether to use sampled pairs (True) or all pairs (False)
        n_samples: Number of pixel pairs to sample (Strategy A)
        use_cache: Enable caching of p_im matrices
        cache_path: Path to cache file (relative to project root)
        random_seed: Random seed for reproducible sampling
    """
    use_sampling: bool = True
    """Use sampled pairs instead of all pairs (recommended for large images)"""

    n_samples: int = 10000
    """Number of pixel pairs to sample (default: 10,000)"""

    use_cache: bool = True
    """Enable caching of p_im matrices to avoid recomputation"""

    cache_path: Union[str, Path] = "cache/pri_cache.json"
    """Path to cache file (relative to project root)"""

    random_seed: int = 42
    """Random seed for reproducible pair sampling"""

    def __post_init__(self):
        """Validate configuration."""
        if self.n_samples <= 0:
            raise ValueError(f"n_samples must be positive, got {self.n_samples}")

        if isinstance(self.cache_path, str):
            self.cache_path = Path(self.cache_path)
