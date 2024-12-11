import numpy as np

def extract_windows(volume, window_size, stride):
    """
    Extract overlapping windows from a 3D volume.
    Parameters:
        volume (np.ndarray): 3D volume to extract windows from.
        window_size (tuple): Size of the window (width, height, depth).
        stride (tuple): Stride for the sliding window.
    Returns:
        list: List of windows and their corresponding coordinates.
    """
    windows = []
    coords = []
    w, h, d = volume.shape
    ww, wh, wd = window_size
    sw, sh, sd = stride

    for x in range(0, w - ww + 1, sw):
        for y in range(0, h - wh + 1, sh):
            for z in range(0, d - wd + 1, sd):
                windows.append(volume[x:x + ww, y:y + wh, z:z + wd])
                coords.append((x, y, z))
    return windows, coords


def merge_windows(volume_shape, windows, coords):
    """
    Merge overlapping windows into the full volume.
    Parameters:
        volume_shape (tuple): Shape of the original volume.
        windows (list of np.ndarray): List of windows to merge.
        coords (list of tuple): Corresponding coordinates of each window.
    Returns:
        np.ndarray: Reconstructed 3D volume.
    """
    merged_volume = np.zeros(volume_shape, dtype=np.float64)
    weight_map = np.zeros(volume_shape, dtype=np.float64)

    for window, (x, y, z) in zip(windows, coords):
        ww, wh, wd = window.shape
        merged_volume[x:x + ww, y:y + wh, z:z + wd] += window
        weight_map[x:x + ww, y:y + wh, z:z + wd] += 1

    # Normalize by the weight map to account for overlapping windows
    merged_volume /= np.maximum(weight_map, 1)
    return merged_volume