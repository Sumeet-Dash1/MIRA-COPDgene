import numpy as np


def majority_vote(labels):
    """
    Perform majority voting on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays.
    Returns:
        np.ndarray: Majority-voted label map.
    """
    # Stack label maps into a single 4D array
    stacked_labels = np.stack(labels, axis=-1)

    # Compute the majority vote along the last axis
    majority_labels = np.apply_along_axis(
        lambda x: np.bincount(x.astype(int)).argmax(), axis=-1, arr=stacked_labels
    )

    return majority_labels


def weighted_vote(labels, weights):
    """
    Perform weighted voting on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays.
        weights (list of float): List of weights corresponding to the labels.
    Returns:
        np.ndarray: Weighted-voted label map.
    """
    if len(labels) != len(weights):
        raise ValueError("Number of labels and weights must be the same.")

    # Stack labels and apply weights
    stacked_labels = np.stack(labels, axis=-1)
    weighted_labels = np.zeros_like(stacked_labels[..., 0], dtype=np.float64)

    for i, weight in enumerate(weights):
        weighted_labels += weight * stacked_labels[..., i]

    # Compute the final label based on maximum weighted vote
    return np.argmax(weighted_labels, axis=-1)


def staple_fusion(labels, max_iterations=10, tolerance=1e-5):
    """
    Perform STAPLE fusion on a list of label arrays.
    Parameters:
        labels (list of np.ndarray): List of label arrays.
        max_iterations (int): Maximum number of iterations.
        tolerance (float): Convergence threshold.
    Returns:
        np.ndarray: STAPLE-fused label map.
    """
    labels = np.stack(labels, axis=-1)  # Stack labels into a 4D array
    n_voxels = labels.shape[:-1]
    n_labels = labels.shape[-1]

    # Initialize probabilities and sensitivities
    p = np.ones(n_voxels) * 0.5  # Prior probabilities
    sensitivities = np.ones(n_labels) * 0.8  # Sensitivity of each labeler
    specificities = np.ones(n_labels) * 0.8  # Specificity of each labeler

    for iteration in range(max_iterations):
        # E-step: Compute weights
        weights = np.zeros_like(labels, dtype=np.float64)
        for i in range(n_labels):
            weights[..., i] = sensitivities[i] * labels[..., i] + \
                              specificities[i] * (1 - labels[..., i])

        weights /= np.sum(weights, axis=-1, keepdims=True)

        # M-step: Update probabilities, sensitivities, and specificities
        new_p = np.mean(weights, axis=-1)
        new_sensitivities = np.mean(weights * labels, axis=(0, 1, 2)) / np.mean(weights, axis=(0, 1, 2))
        new_specificities = np.mean(weights * (1 - labels), axis=(0, 1, 2)) / np.mean(weights, axis=(0, 1, 2))

        # Check for convergence
        if np.max(np.abs(new_p - p)) < tolerance:
            break

        p = new_p
        sensitivities = new_sensitivities
        specificities = new_specificities

    # Final fused label map
    return (p > 0.5).astype(np.int32)


def probability_fusion(labels):
    """
    Perform probability-based label fusion by selecting the label with the maximum probability.
    Parameters:
        labels (list of np.ndarray): List of 3D label arrays (e.g., (256, 128, 256)).
    Returns:
        np.ndarray: Fused label map (3D array).
    """
    # Stack labels into a single 4D array: (x, y, z, n_labels)
    stacked_labels = np.stack(labels, axis=-1)  # Shape: (x, y, z, n_labels)

    # Get unique labels from all arrays
    unique_labels = np.unique(stacked_labels)  # Assume integer labels

    # Initialize a probability array: (x, y, z, n_unique_labels)
    probabilities = np.zeros(stacked_labels.shape[:-1] + (len(unique_labels),), dtype=np.float64)

    # Compute probabilities for each unique label
    for i, label in enumerate(unique_labels):
        probabilities[..., i] = np.mean(stacked_labels == label, axis=-1)

    # Select the label with the maximum probability for each voxel
    fused_label = unique_labels[np.argmax(probabilities, axis=-1)]  # Shape: (x, y, z)

    return fused_label