import numpy as np

def compute_weights(fixed_image, registered_images, window_coords, window_size, metric="ncc"):
    """
    Compute weights for each registered image in a given window based on a similarity metric.
    Parameters:
        fixed_image (np.ndarray): Fixed image.
        registered_images (list of np.ndarray): List of registered images.
        window_coords (tuple): Coordinates of the window (x, y, z).
        window_size (tuple): Size of the window (width, height, depth).
        metric (str): Similarity metric ("ncc", "mse", "entropy").
    Returns:
        list: Weights for each registered image.
    """
    x, y, z = window_coords
    ww, wh, wd = window_size

    fixed_window = fixed_image[x:x + ww, y:y + wh, z:z + wd]
    weights = []

    for registered in registered_images:
        registered_window = registered[x:x + ww, y:y + wh, z:z + wd]

        if metric == "ncc":
            weight = np.corrcoef(fixed_window.ravel(), registered_window.ravel())[0, 1]
        elif metric == "mse":
            weight = -np.mean((fixed_window - registered_window) ** 2)  # Lower MSE is better
        elif metric == "entropy":
            p = np.histogram(registered_window, bins=256, density=True)[0]
            weight = -np.sum(p * np.log(p + 1e-10))  # Lower entropy is better
        else:
            raise ValueError(f"Unknown metric: {metric}")

        weights.append(weight)

    # Normalize weights
    weights = np.array(weights)
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-10)
    return weights

from src.utils.sliding_window import extract_windows, merge_windows

def fuse_window(window_labels, weights, fusion_strategy="weighted_vote"):
    """
    Fuse labels within a window using a specified fusion strategy.
    Parameters:
        window_labels (list of np.ndarray): List of label maps for the window.
        weights (list of float): Weights for each label map.
        fusion_strategy (str): Fusion strategy ("majority_vote", "weighted_vote").
    Returns:
        np.ndarray: Fused label map for the window.
    """
    from src.segmentation.fusion_methods import majority_vote, weighted_vote

    if fusion_strategy == "majority_vote":
        return majority_vote(window_labels)
    elif fusion_strategy == "weighted_vote":
        return weighted_vote(window_labels, weights)
    else:
        raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

def window_based_segmentation(fixed_image, registered_images, label_maps, window_size, stride, fusion_strategy, metric):
    """
    Perform window-based segmentation with dynamic weights and fusion strategy.
    Parameters:
        fixed_image (np.ndarray): Fixed image.
        registered_images (list of np.ndarray): Registered images for similarity computation.
        label_maps (list of np.ndarray): Transformed label maps for fusion.
        window_size (tuple): Size of the sliding window (width, height, depth).
        stride (tuple): Stride for sliding window.
        fusion_strategy (str): Fusion strategy ("majority_vote", "weighted_vote").
        metric (str): Similarity metric for weight computation.
    Returns:
        np.ndarray: Final fused label map.
    """
    # Extract windows and coordinates
    windows, coords = extract_windows(fixed_image, window_size, stride)

    # Fuse windows
    fused_windows = []
    for coord in coords:
        weights = compute_weights(fixed_image, registered_images, coord, window_size, metric)
        x, y, z = coord
        ww, wh, wd = window_size

        window_labels = [
            lm[x:x + ww, y:y + wh, z:z + wd]
            for lm in label_maps
        ]
        fused_window = fuse_window(window_labels, weights, fusion_strategy)
        fused_windows.append(fused_window)

    # Merge fused windows back into the full volume
    fused_volume = merge_windows(fixed_image.shape, fused_windows, coords)
    return fused_volume