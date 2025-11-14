"""
PRI Cache Manager

JSON-based caching for p_im matrices to avoid recomputation.
Caches sampled pixel pairs and their p_im values per image.
"""

import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import numpy as np


class PRICacheManager:
    """
    Manager for caching p_im matrices in JSON format.

    Stores sampled pixel pairs and their p_im values for each image,
    avoiding expensive recomputation when evaluating multiple segmentations.

    Cache structure:
    {
        "cache_version": "1.0",
        "images": {
            "12074": {
                "pairs": [[0, 1547], [234, 5678], ...],
                "p_values": [0.8, 0.6, 0.4, ...],
                "n_ground_truths": 5,
                "image_shape": [321, 481],
                "created": "2025-01-11T10:30:00"
            }
        }
    }
    """

    def __init__(self, cache_path: Path):
        """
        Initialize cache manager.

        Args:
            cache_path: Path to cache JSON file
        """
        self.cache_path = cache_path
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict:
        """Load cache from JSON file."""
        if not self.cache_path.exists():
            return {
                "cache_version": "1.0",
                "images": {}
            }

        try:
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            # If cache is corrupted, start fresh
            return {
                "cache_version": "1.0",
                "images": {}
            }

    def _save_cache(self):
        """Save cache to JSON file."""
        # Ensure cache directory exists
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.cache_path, 'w') as f:
            json.dump(self._cache, f, indent=2)

    def has_cached_p_im(self, image_id: str) -> bool:
        """
        Check if p_im matrix is cached for an image.

        Args:
            image_id: Image identifier (e.g., '12074')

        Returns:
            True if cached, False otherwise
        """
        return image_id in self._cache["images"]

    def get_p_im_data(self, image_id: str) -> Optional[Dict]:
        """
        Get cached p_im data for an image.

        Args:
            image_id: Image identifier

        Returns:
            Dictionary with:
                - pairs: List of [i, m] pixel pair indices
                - p_values: List of corresponding p_im values
                - n_ground_truths: Number of ground truths used
                - image_shape: [H, W] shape
            Or None if not cached
        """
        if not self.has_cached_p_im(image_id):
            return None

        return self._cache["images"][image_id]

    def save_p_im_data(
        self,
        image_id: str,
        pairs: np.ndarray,
        p_values: np.ndarray,
        n_ground_truths: int,
        image_shape: Tuple[int, int]
    ):
        """
        Save p_im data for an image.

        Args:
            image_id: Image identifier
            pairs: Sampled pixel pairs, shape (N, 2)
            p_values: Corresponding p_im values, shape (N,)
            n_ground_truths: Number of ground truths used
            image_shape: (H, W) image dimensions
        """
        self._cache["images"][image_id] = {
            "pairs": pairs.tolist(),
            "p_values": p_values.tolist(),
            "n_ground_truths": int(n_ground_truths),
            "image_shape": list(image_shape),
            "created": datetime.now().isoformat()
        }

        self._save_cache()

    def invalidate(self, image_id: str):
        """
        Invalidate (remove) cache for specific image.

        Args:
            image_id: Image identifier to invalidate
        """
        if image_id in self._cache["images"]:
            del self._cache["images"][image_id]
            self._save_cache()

    def clear_all(self):
        """Clear entire cache."""
        self._cache = {
            "cache_version": "1.0",
            "images": {}
        }
        self._save_cache()

    def get_stats(self) -> Dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        n_images = len(self._cache["images"])
        return {
            "n_cached_images": n_images,
            "cached_image_ids": list(self._cache["images"].keys()),
            "cache_path": str(self.cache_path),
            "cache_exists": self.cache_path.exists()
        }
