from skimage import exposure
import numpy as np
from skimage import morphology
import matplotlib.pyplot as plt
import cv2

def hu_data(volume, rescale_slope=1.0, rescale_intercept=-1024.0, 
                        background_value=-2000, target_range=(-1, 1)):
    """
    Converts a 3D CT volume to Hounsfield Units (HU) and normalizes for a U-Net model.

    Parameters:
        volume (numpy.ndarray): The 3D volume to preprocess.
        rescale_slope (float): Rescale slope from the DICOM metadata (default: 1.0).
        rescale_intercept (float): Rescale intercept from the DICOM metadata (default: -1024.0).
        background_value (int): Background intensity value to replace (default: -2000).
        target_range (tuple): Desired range for normalization (e.g., (-1, 1)).

    Returns:
        numpy.ndarray: The preprocessed volume in Hounsfield Units (HU).
    """
    # Ensure the volume is a NumPy array
    volume = np.asarray(volume)

    # Step 1: Convert to Hounsfield Units (HU)
    hu_volume = volume * rescale_slope + rescale_intercept

    # Replace background value (-2000) with the minimum HU value (-1024)
    hu_volume[volume == background_value] = -1024

    # Step 2: Clip HU values to a typical range for CT scans (e.g., -1000 to 400 HU)
    hu_volume = np.clip(hu_volume, -1000, 400)

    # # Step 3: Normalize HU values to the target range (e.g., [-1, 1])
    # min_val, max_val = target_range
    # hu_normalized = (hu_volume - (-1000)) / (400 - (-1000))  # Normalize to [0, 1]
    # hu_normalized = hu_normalized * (max_val - min_val) + min_val  # Scale to target range

    return hu_volume

def convert_to_signed_4bit(ct_image):
    """
    Converts a CT image in Hounsfield Units (HU) to a signed 4-bit representation.

    Parameters:
        ct_image (numpy.ndarray): The 3D CT image in HU.

    Returns:
        numpy.ndarray: The signed 4-bit CT image with values in the range [-8, 7].

    Notes:
        - The CT image is normalized symmetrically based on the maximum absolute value.
        - Values are clipped to fit within the signed 4-bit range.
    """
    # Find the maximum absolute value to normalize symmetrically
    max_abs_value = max(abs(ct_image.min()), abs(ct_image.max()))

    # Normalize to the range [-8, 7]
    ct_image_signed_4bit = np.round((ct_image / max_abs_value) * 7).astype(np.int8)

    # Clip values to ensure they fit within the signed 4-bit range [-8, 7]
    ct_image_signed_4bit = np.clip(ct_image_signed_4bit, -8, 7)

    return ct_image_signed_4bit

def convert_signed_4bit_to_hu(ct_image_4bit, max_abs_value = 1000):
    """
    Converts a signed 4-bit CT image back to the original HU range.

    Parameters:
        ct_image_4bit (numpy.ndarray): The signed 4-bit CT image (range: [-8, 7]).
        max_abs_value (float, optional): The maximum absolute HU value used during the initial conversion (default: 1000).

    Returns:
        numpy.ndarray: The CT image converted back to the original HU range.

    Notes:
        - The conversion assumes the original range was symmetrically normalized using the given max_abs_value.
    """
    # Reverse scaling
    ct_image_hu = (ct_image_4bit / 7) * max_abs_value

    return ct_image_hu

def convert_signed_4bit_to_unsigned_8bit(ct_image_4bit):
    """
    Converts a signed 4-bit CT image (range: [-8, 7]) to an unsigned 8-bit image (range: [0, 255]).
    
    Parameters:
        ct_image_4bit (numpy.ndarray): The signed 4-bit CT image (range: [-8, 7]).
    
    Returns:
        numpy.ndarray: The unsigned 8-bit image (range: [0, 255]).

    Raises:
        ValueError: If the input values are not within the range [-8, 7].

    Notes:
        - The signed 4-bit range is linearly scaled to fit the unsigned 8-bit range.
    """
    # Ensure the input range is valid
    if not np.all((ct_image_4bit >= -8) & (ct_image_4bit <= 7)):
        raise ValueError("Input image values must be in the range [-8, 7].")
    
    # Scale signed 4-bit range [-8, 7] to unsigned 8-bit range [0, 255]
    ct_image_8bit = ((ct_image_4bit + 8) / 15 * 255).astype(np.uint8)
    
    return ct_image_8bit

def flatten_peak_in_range(image, peak_range, target_range):
    """
    Flattens intensities within a specified peak range to a target range.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        peak_range (tuple): The intensity range of the peak to flatten (min, max).
        target_range (tuple): The target intensity range for flattening (min, max).

    Returns:
        numpy.ndarray: The image with the specified peak flattened to the target range.

    Notes:
        - The function scales intensities within the specified peak range to the target range linearly.
        - Pixels outside the peak range remain unchanged.
    """
    # Extract the min and max of the input peak range
    peak_min, peak_max = peak_range

    # Extract the min and max of the target range
    target_min, target_max = target_range

    # Copy the image to avoid modifying the original
    flattened_image = image.copy()

    # Scale intensities within the peak range to the target range
    mask = (flattened_image >= peak_min) & (flattened_image <= peak_max)
    flattened_image[mask] = ((flattened_image[mask] - peak_min) / (peak_max - peak_min)) * (target_max - target_min) + target_min

    return flattened_image


