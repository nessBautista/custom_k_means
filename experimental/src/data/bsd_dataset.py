"""
BSD300 Dataset Handler

Provides interface for loading images and ground truth segmentations
from the Berkeley Segmentation Dataset (BSD300/BSD500).

Dataset Structure:
    bsd500/
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/          # Paper's 10 test images
    └── ground_truth/
        ├── train/
        ├── val/
        └── test/          # .mat files with ~5 annotations each

Paper Reference: Islam et al. (2021) uses 10 specific test images (Table I)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional, Union
import numpy as np
import cv2
import scipy.io as sio


# Paper's 10 test images from Table I
# NOTE: These images are from BSD500's train (7) and val (3) splits, not test.
#       The dataset loader automatically searches across all splits.
#       Train: 12074, 42044, 176035, 176039, 178054, 216066, 353013
#       Val:   86016, 147091, 160068
PAPER_IMAGE_IDS = [
    '12074',   # I1  (train)
    '42044',   # I2  (train)
    '86016',   # I3  (val)
    '147091',  # I4  (val)
    '160068',  # I5  (val)
    '176035',  # I6  (train)
    '176039',  # I7  (train)
    '178054',  # I8  (train, best PRI: 0.8556)
    '216066',  # I9  (train)
    '353013'   # I10 (train)
]


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class BSD300Config:
    """
    Configuration for BSD300 dataset loading.

    The dataset path is relative to the custom_k_means project root.
    """
    dataset_path: Union[str, Path] = "src/data/bsd500"
    """Path to BSD500 dataset root directory.
    Can be absolute or relative to project root."""

    split: str = "test"
    """Dataset split to use: 'train', 'val', or 'test'"""

    image_ids: Optional[List[str]] = None
    """Specific image IDs to load. If None, loads all images in split."""

    normalize: bool = True
    """Whether to normalize images to [0, 1] range"""

    def __post_init__(self):
        """Validate configuration."""
        self.dataset_path = Path(self.dataset_path)

        if self.split not in ['train', 'val', 'test']:
            raise ValueError(
                f"split must be 'train', 'val', or 'test', got '{self.split}'"
            )

    def get_images_dir(self) -> Path:
        """Get path to images directory for current split."""
        return self.dataset_path / 'images' / self.split

    def get_ground_truth_dir(self) -> Path:
        """Get path to ground truth directory for current split."""
        return self.dataset_path / 'ground_truth' / self.split


# ============================================================================
# Dataset Class
# ============================================================================

class BSD300Dataset:
    """
    Berkeley Segmentation Dataset (BSD300) handler.

    Provides iteration interface for loading images and ground truth
    segmentations. Each image typically has 5 human annotations.

    Example:
        >>> from src.data.bsd_dataset import BSD300Dataset, BSD300Config
        >>>
        >>> # Load paper's 10 test images
        >>> config = BSD300Config()
        >>> dataset = BSD300Dataset(config).get_paper_images()
        >>>
        >>> print(f"Dataset size: {len(dataset)}")
        >>> # Dataset size: 10
        >>>
        >>> # Iterate over images
        >>> for image_id, image, ground_truths in dataset:
        >>>     print(f"{image_id}: {image.shape}, {len(ground_truths)} GTs")
        >>>
        >>> # Access by index
        >>> image_id, image, gts = dataset[0]
    """

    def __init__(self, config: BSD300Config):
        """
        Initialize BSD300 dataset.

        Args:
            config: Dataset configuration
        """
        self.config = config
        self._image_list = self._build_image_list()

    def _build_image_list(self) -> List[Tuple[str, Path]]:
        """
        Build list of (image_id, image_path) tuples.

        When specific image_ids are provided, searches across all splits
        (train/val/test) since papers often use images from different splits.

        Returns:
            image_list: List of (image_id, path) for all images in split
        """
        images_dir = self.config.get_images_dir()

        if not images_dir.exists():
            raise FileNotFoundError(
                f"Images directory not found: {images_dir}\n"
                f"Please ensure BSD500 dataset is in {self.config.dataset_path}"
            )

        # Get list of image IDs
        if self.config.image_ids is not None:
            # Use specified image IDs - search across all splits
            image_ids = self.config.image_ids
            image_list = []

            # Search in all splits (train, val, test)
            all_splits = ['train', 'val', 'test']

            for image_id in image_ids:
                found = False

                # First try the configured split
                image_path = images_dir / f'{image_id}.jpg'
                if image_path.exists():
                    image_list.append((image_id, image_path))
                    found = True
                else:
                    # Search in other splits
                    for split in all_splits:
                        if split == self.config.split:
                            continue  # Already checked
                        alt_path = self.config.dataset_path / 'images' / split / f'{image_id}.jpg'
                        if alt_path.exists():
                            image_list.append((image_id, alt_path))
                            found = True
                            break

                if not found:
                    print(f"Warning: Image {image_id} not found in any split")

            if not image_list:
                raise ValueError(
                    f"No images found with IDs: {image_ids}\n"
                    f"Searched in splits: {all_splits}"
                )
        else:
            # Get all .jpg files in the specified split directory
            image_paths = sorted(images_dir.glob('*.jpg'))
            image_ids = [p.stem for p in image_paths]

            image_list = []
            for image_id, image_path in zip(image_ids, image_paths):
                image_list.append((image_id, image_path))

            if not image_list:
                raise ValueError(
                    f"No images found in {images_dir}"
                )

        return image_list

    def __len__(self) -> int:
        """Get number of images in dataset."""
        return len(self._image_list)

    def __getitem__(self, idx: int) -> Tuple[str, np.ndarray, List[np.ndarray]]:
        """
        Get image and ground truths by index.

        Args:
            idx: Index of image to load

        Returns:
            image_id: Image identifier (e.g., '12074')
            image: RGB image of shape (H, W, 3), normalized to [0, 1]
            ground_truths: List of ground truth segmentations,
                          each of shape (H, W) with integer labels

        Example:
            >>> dataset = BSD300Dataset(BSD300Config())
            >>> image_id, image, gts = dataset[0]
            >>> print(f"ID: {image_id}")
            >>> print(f"Image: {image.shape}, range [{image.min():.2f}, {image.max():.2f}]")
            >>> print(f"Ground truths: {len(gts)} annotations")
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")

        image_id, image_path = self._image_list[idx]

        # Load image
        image = self.load_image(image_path)

        # Load ground truths
        ground_truths = self.load_ground_truths(image_id)

        return image_id, image, ground_truths

    def __iter__(self):
        """Iterate over all images in dataset."""
        for idx in range(len(self)):
            yield self[idx]

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """
        Load and preprocess image from file.

        Based on: research/intuition/custom_kmeans/utils.py

        Args:
            image_path: Path to image file

        Returns:
            image: RGB image of shape (H, W, 3)
                  Values in [0, 1] if normalize=True, else [0, 255]

        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image cannot be loaded
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image using OpenCV
        img = cv2.imread(str(image_path))

        if img is None:
            raise ValueError(f"Could not load image from {image_path}")

        # Convert BGR to RGB (OpenCV loads as BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1] if requested
        if self.config.normalize:
            img = img.astype(np.float32) / 255.0

        return img

    def load_ground_truths(self, image_id: str) -> List[np.ndarray]:
        """
        Load ground truth segmentations for an image.

        BSD500 provides multiple human annotations per image stored in
        .mat files. Typically 5 annotations per image.

        Searches across all splits (train/val/test) if not found in
        the configured split, to match the image loading behavior.

        Based on: research/intuition/custom_kmeans/utils.py

        Args:
            image_id: Image identifier (e.g., '12074')

        Returns:
            ground_truths: List of segmentation masks, each (H, W)
                          with integer labels [0, N_segments-1]

        Example:
            >>> gts = dataset.load_ground_truths('12074')
            >>> print(f"Found {len(gts)} annotations")
            >>> # Found 5 annotations
            >>> print(f"First GT: {gts[0].shape}, {gts[0].max()} segments")
        """
        gt_dir = self.config.get_ground_truth_dir()
        gt_path = gt_dir / f'{image_id}.mat'

        # If not found in configured split, search in other splits
        if not gt_path.exists():
            all_splits = ['train', 'val', 'test']
            for split in all_splits:
                if split == self.config.split:
                    continue
                alt_gt_path = self.config.dataset_path / 'ground_truth' / split / f'{image_id}.mat'
                if alt_gt_path.exists():
                    gt_path = alt_gt_path
                    break

        ground_truths = []

        if not gt_path.exists():
            print(f"Warning: Ground truth not found for {image_id} in any split")
            return ground_truths

        try:
            # Load .mat file
            mat_data = sio.loadmat(str(gt_path))

            # BSD500 .mat structure: groundTruth[0][i]['Segmentation']
            if 'groundTruth' not in mat_data:
                print(f"Warning: No 'groundTruth' key in {gt_path}")
                return ground_truths

            gt_array = mat_data['groundTruth'][0]

            # Extract each annotation
            for i in range(len(gt_array)):
                # Access nested structure: gt_array[i]['Segmentation'][0, 0]
                segmentation = gt_array[i]['Segmentation'][0, 0]
                ground_truths.append(segmentation.astype(np.int32))

        except Exception as e:
            print(f"Warning: Could not load ground truth from {gt_path}: {e}")

        return ground_truths

    def get_paper_images(self) -> 'BSD300Dataset':
        """
        Get dataset with only the 10 images from paper's Table I.

        Paper Reference: Islam et al. (2021), Table I
        Image IDs: 12074, 42044, 86016, 147091, 160068,
                  176035, 176039, 178054, 216066, 353013

        Returns:
            dataset: New BSD300Dataset instance with only paper's images

        Example:
            >>> # Load full test set
            >>> dataset = BSD300Dataset(BSD300Config(split='test'))
            >>> print(len(dataset))  # Could be 100+ images
            >>>
            >>> # Get only paper's 10 images
            >>> paper_dataset = dataset.get_paper_images()
            >>> print(len(paper_dataset))  # Exactly 10 images
        """
        # Create new config with paper's image IDs
        paper_config = BSD300Config(
            dataset_path=self.config.dataset_path,
            split=self.config.split,
            image_ids=PAPER_IMAGE_IDS,
            normalize=self.config.normalize
        )

        return BSD300Dataset(paper_config)

    def get_image_by_id(self, image_id: str) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Load specific image by ID.

        Args:
            image_id: Image identifier (e.g., '12074')

        Returns:
            image: RGB image (H, W, 3)
            ground_truths: List of ground truth segmentations

        Raises:
            ValueError: If image_id not found in dataset

        Example:
            >>> dataset = BSD300Dataset(BSD300Config()).get_paper_images()
            >>> image, gts = dataset.get_image_by_id('12074')
        """
        for idx, (img_id, _) in enumerate(self._image_list):
            if img_id == image_id:
                _, image, ground_truths = self[idx]
                return image, ground_truths

        raise ValueError(
            f"Image ID '{image_id}' not found in dataset. "
            f"Available IDs: {[img_id for img_id, _ in self._image_list]}"
        )

    def get_statistics(self) -> dict:
        """
        Get dataset statistics.

        Returns:
            stats: Dictionary with dataset information

        Example:
            >>> dataset = BSD300Dataset(BSD300Config()).get_paper_images()
            >>> stats = dataset.get_statistics()
            >>> print(f"Images: {stats['n_images']}")
            >>> print(f"Avg GTs per image: {stats['avg_ground_truths']:.1f}")
        """
        n_images = len(self)
        image_ids = [img_id for img_id, _ in self._image_list]

        # Count ground truths
        total_gts = 0
        for image_id, _, gts in self:
            total_gts += len(gts)

        avg_gts = total_gts / n_images if n_images > 0 else 0

        return {
            'n_images': n_images,
            'split': self.config.split,
            'image_ids': image_ids,
            'total_ground_truths': total_gts,
            'avg_ground_truths': avg_gts,
            'dataset_path': str(self.config.dataset_path)
        }

    def print_summary(self):
        """Print dataset summary."""
        stats = self.get_statistics()

        print("=" * 70)
        print("BSD300 Dataset Summary")
        print("=" * 70)
        print(f"Dataset path:  {stats['dataset_path']}")
        print(f"Split:         {stats['split']}")
        print(f"Images:        {stats['n_images']}")
        print(f"Ground truths: {stats['total_ground_truths']} total "
              f"({stats['avg_ground_truths']:.1f} per image)")
        print()
        print("Image IDs:")
        for i, img_id in enumerate(stats['image_ids'], 1):
            print(f"  I{i:2d}. {img_id}")
        print("=" * 70)


# ============================================================================
# Helper Functions
# ============================================================================

def check_dataset_exists(dataset_path: Union[str, Path] = "src/data/bsd500") -> bool:
    """
    Check if BSD500 dataset exists at given path.

    Args:
        dataset_path: Path to dataset root

    Returns:
        exists: True if dataset structure is valid
    """
    dataset_path = Path(dataset_path)

    required_dirs = [
        dataset_path / 'images',
        dataset_path / 'groundTruth',
    ]

    return all(d.exists() for d in required_dirs)


def get_expected_dataset_structure() -> str:
    """
    Get expected BSD500 dataset directory structure.

    Returns:
        structure: String describing expected structure
    """
    return """
Expected BSD500 Dataset Structure:
-----------------------------------
bsd500/
├── images/
│   ├── train/           # Training images
│   ├── val/             # Validation images
│   └── test/            # Test images (paper uses these)
│       ├── 12074.jpg
│       ├── 42044.jpg
│       ├── ... (more test images)
└── groundTruth/
    ├── train/
    ├── val/
    └── test/            # Ground truth .mat files
        ├── 12074.mat    # Contains ~5 human annotations
        ├── 42044.mat
        ├── ... (more ground truth files)

Each .mat file contains:
    groundTruth[0][i]['Segmentation'][0, 0] -> (H, W) array
    where i ∈ [0, num_annotations-1] (typically 5)
"""
