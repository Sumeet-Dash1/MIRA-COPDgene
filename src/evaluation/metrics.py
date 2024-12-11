import numpy as np
from scipy.spatial.distance import directed_hausdorff

def compute_dice_coefficient(seg1, seg2):
    """
    Compute the Dice Similarity Coefficient (DSC) between two binary masks.
    Parameters:
        seg1 (np.ndarray): Binary segmentation 1.
        seg2 (np.ndarray): Binary segmentation 2.
    Returns:
        float: Dice Similarity Coefficient.
    """
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    return 2 * intersection / (np.sum(seg1 > 0) + np.sum(seg2 > 0))

def compute_hausdorff_distance(seg1, seg2):
    """
    Compute the Hausdorff Distance (HD) between two binary masks.
    Parameters:
        seg1 (np.ndarray): Binary segmentation 1.
        seg2 (np.ndarray): Binary segmentation 2.
    Returns:
        float: Hausdorff Distance.
    """
    coords1 = np.argwhere(seg1 > 0)
    coords2 = np.argwhere(seg2 > 0)

    if coords1.size == 0 or coords2.size == 0:
        return float('inf')  # No overlapping elements

    hd1 = directed_hausdorff(coords1, coords2)[0]
    hd2 = directed_hausdorff(coords2, coords1)[0]
    return max(hd1, hd2)

def compute_average_volume_difference(seg1, seg2, voxel_volume=1.0):
    """
    Compute the Average Volume Difference (AVD) between two binary masks.
    Parameters:
        seg1 (np.ndarray): Binary segmentation 1.
        seg2 (np.ndarray): Binary segmentation 2.
        voxel_volume (float): Volume of a single voxel (default is 1.0).
    Returns:
        float: Average Volume Difference.
    """
    vol1 = np.sum(seg1 > 0) * voxel_volume
    vol2 = np.sum(seg2 > 0) * voxel_volume
    return abs(vol1 - vol2) / ((vol1 + vol2) / 2)