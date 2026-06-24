"""
skeletonize.py
Converts a binary road mask into a thin 1-pixel-wide skeleton (centerline).
Uses scikit-image skeletonization.
"""

import numpy as np
from skimage.morphology import skeletonize


def mask_to_skeleton(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary road mask (0/1 or 0/255) into a skeletonized version.

    Args:
        mask: 2D numpy array, road pixels = 1 (or 255), background = 0

    Returns:
        2D numpy array (same shape), skeleton pixels = 1, rest = 0
    """
    binary_mask = (mask > 0).astype(bool)
    skeleton = skeletonize(binary_mask)
    return skeleton.astype(np.uint8)


if __name__ == "__main__":
    # quick sanity test with a synthetic mask (a thick horizontal road)
    test_mask = np.zeros((100, 100), dtype=np.uint8)
    test_mask[45:55, 10:90] = 1  # thick horizontal strip

    skeleton = mask_to_skeleton(test_mask)

    print(f"Original mask road pixels: {test_mask.sum()}")
    print(f"Skeleton pixels: {skeleton.sum()}")
    print(f"Skeleton shape: {skeleton.shape}")