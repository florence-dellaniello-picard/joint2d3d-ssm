"""
Author: Florence Dell'Aniello Picard
Metric utilities: dice score, mean surface distance (MSD) and Hausdorff distance (HD).
"""

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt

def dice_score(pred, target, threshold=0.5):
    """Compute Dice score between two binary volumes."""
    pred   = (pred   > threshold).astype(np.float32)
    target = (target > threshold).astype(np.float32)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum()
    return 2.0 * intersection / union

def hausdorff_and_msd(a, b, threshold=0.5):
    """Hausdorff distance (95th percentile) and Mean Surface Distance between two binary volumes, computed via distance transforms."""
    a_bin = (a > threshold).astype(bool)
    b_bin = (b > threshold).astype(bool)

    def surface(mask):
        return mask & ~binary_erosion(mask)

    surf_a = surface(a_bin)
    surf_b = surface(b_bin)

    # Distance transform
    dt_a = distance_transform_edt(~surf_a)   # dist to surface of a
    dt_b = distance_transform_edt(~surf_b)   # dist to surface of b

    # Distances from each surface to the other
    dist_a_to_b = dt_b[surf_a] 
    dist_b_to_a = dt_a[surf_b] 

    # Hausdorff 95
    hd95 = max(np.percentile(dist_a_to_b, 95),
               np.percentile(dist_b_to_a, 95))

    # Mean Surface Distance (symmetric)
    msd = (dist_a_to_b.mean() + dist_b_to_a.mean()) / 2.0

    return hd95, msd