import os
import numpy as np
import nibabel as nib

def load_labels(label_dir):
    """
    Load all transformed label maps from a directory.
    Parameters:
        label_dir (str): Directory containing the label maps.
    Returns:
        list of np.ndarray: List of label arrays.
    """
    labels = []
    for file in sorted(os.listdir(label_dir)):
        if file.endswith(".hdr"):
            # Load the label map (assuming Analyze format)
            img = nib.load(os.path.join(label_dir, file))
            labels.append(img.get_fdata())
    return labels

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

def save_label(output_path, label_map, reference_image):
    """
    Save the majority-voted label map as a NIfTI file.
    Parameters:
        output_path (str): Path to save the resulting label map.
        label_map (np.ndarray): Majority-voted label map.
        reference_image (nib.Nifti1Image): Reference image for metadata.
    """
    output_img = nib.Nifti1Image(label_map.astype(np.int16), reference_image.affine)
    nib.save(output_img, output_path)

def majority_voting_segmentation(fixed_image, label_dir, output_path):
    """
    Perform majority voting segmentation for a fixed image.
    Parameters:
        fixed_image (str): Path to the fixed image (reference).
        label_dir (str): Directory containing transformed label maps.
        output_path (str): Path to save the majority-voted segmentation.
    """
    # Load reference image (fixed image)
    ref_img = nib.load(fixed_image)

    # Load transformed labels
    labels = load_labels(label_dir)

    # Perform majority voting
    majority_labels = majority_vote(labels)

    # Save the result
    save_label(output_path, majority_labels, ref_img)
    print(f"Majority voting segmentation saved to {output_path}")

