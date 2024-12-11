### **Helper Scripts**

#### `src/utils/helpers.py`
import nibabel as nib
import numpy as np

def compute_dice_coefficient(seg1, seg2):
    """Compute Dice Similarity Coefficient between two binary masks."""
    intersection = np.sum((seg1 > 0) & (seg2 > 0))
    return 2 * intersection / (np.sum(seg1 > 0) + np.sum(seg2 > 0))

def calculate_dice_score(img1_data, img2_data, ignore_background=True):
    """
    Calculate the Dice score between two NIfTI images.
    
    Parameters:
    nii_image1(np.ndarray:): first NIfTI image (e.g., ground truth).
    nii_image2 (np.ndarray:): second NIfTI image (e.g., predicted labels).
    ignore_background (bool): Whether to ignore the background label (default: True).
    
    Returns:
    dice_scores (dict): Dictionary of Dice scores for each label present in the images.
    """    
    # Ensure the two images have the same shape
    if img1_data.shape != img2_data.shape:
        raise ValueError("The two NIfTI images must have the same shape.")
    
    # Calculate the Dice score for each label
    dice_scores = {}
    labels = np.unique(img1_data)  # Find all unique labels in the first image
    
    for label in labels:
        if label == 0 and ignore_background:
            continue  # Skip background (assuming label 0 is background)
        
        # Create binary masks for the current label in both images
        img1_mask = (img1_data == label)
        img2_mask = (img2_data == label)
        
        # Calculate the intersection and union
        intersection = np.sum(img1_mask & img2_mask)
        union = np.sum(img1_mask) + np.sum(img2_mask)
        
        # Calculate Dice score (handle division by zero)
        if union == 0:
            dice_score = 1.0  # Perfect match if both masks are empty
        else:
            dice_score = (2. * intersection) / union
        
        # Store the Dice score for the current label
        dice_scores[label] = dice_score
    
    return dice_scores

def normalize_image(image):
    """Normalize image intensities to zero mean and unit variance."""
    return (image - np.mean(image)) / np.std(image)

def load_nifti(file_path):
    """
    Load a NIfTI file and return the image data and affine transformation matrix.
    Parameters:
        file_path (str): Path to the NIfTI file.
    Returns:
        np.ndarray: 3D image data.
        np.ndarray: Affine transformation matrix.
    """
    img = nib.load(file_path)
    return img.get_fdata(), img.affine


def save_nifti(filepath, data, reference):
    """Save a NIFTI file with reference metadata."""
    nib.save(nib.Nifti1Image(data, affine=reference.affine), filepath)

def compute_correlation_weights(fixed_image, registered_images):
    """
    Compute correlation-based weights for registered images.
    Parameters:
        fixed_image (np.ndarray): The validation image (fixed image).
        registered_images (list of np.ndarray): List of elastically registered images.
    Returns:
        np.ndarray: Normalized correlation-based weights (1D array).
    """
    # Ensure fixed_image and registered_images are 3D arrays
    fixed_image = np.squeeze(fixed_image)
    registered_images = [np.squeeze(img) for img in registered_images]

    # Compute Pearson correlation coefficients
    correlations = []
    for reg_img in registered_images:
        # Compute Pearson correlation coefficient
        corr = np.corrcoef(fixed_image.ravel(), reg_img.ravel())[0, 1]
        correlations.append(max(0, corr))  # Use only positive correlations

    # Normalize weights to sum to 1
    weights = np.array(correlations) / (np.sum(correlations) + 1e-10)

    return weights