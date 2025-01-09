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
        numpy.ndarray: The preprocessed volume.
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
    Converts a CT image in HU to a signed 4-bit representation with values in the range [-8, 7].

    Parameters:
        ct_image (numpy.ndarray): The 3D CT image in HU.

    Returns:
        numpy.ndarray: The signed 4-bit CT image with values in the range [-8, 7].
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
        max_abs_value (float): The maximum absolute value used during the initial conversion.

    Returns:
        numpy.ndarray: The CT image in the original HU range.
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
    """
    # Ensure the input range is valid
    if not np.all((ct_image_4bit >= -8) & (ct_image_4bit <= 7)):
        raise ValueError("Input image values must be in the range [-8, 7].")
    
    # Scale signed 4-bit range [-8, 7] to unsigned 8-bit range [0, 255]
    ct_image_8bit = ((ct_image_4bit + 8) / 15 * 255).astype(np.uint8)
    
    return ct_image_8bit

def normalize_image(image, in_range=(-2000, 2000), mid_range=(-1000, 1000)):
    """
    Normalizes a CT image array.

    Parameters
    ----------
    image : np.ndarray
        Input CT image as a NumPy array.
    in_range : tuple
        Intensity range for the entire image normalization (default: (-2000, 2000)).
    mid_range : tuple
        Intensity range for rescaling values above -2000 (default: (-1000, 1000)).

    Returns
    -------
    np.ndarray
        Normalized image as a NumPy array.
    """
    # Rescale intensities for the full range
    normalized_image = exposure.rescale_intensity(image, in_range='image', out_range=in_range)
    # Further rescale intensities for values above -2000
    normalized_image[normalized_image > -2000] = exposure.rescale_intensity(
        normalized_image[normalized_image > -2000],
        in_range='image',
        out_range=mid_range
    )
    return normalized_image

def apply_clahe(image, kernel_size=None):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to a CT image array.

    Parameters
    ----------
    image : np.ndarray
        Input CT image as a NumPy array.
    kernel_size : tuple, optional
        Kernel size for the CLAHE algorithm. Default is None, which uses the default skimage setting.

    Returns
    -------
    np.ndarray
        CLAHE-enhanced image as a NumPy array.
    """
    # Normalize image to range [0, 1] for CLAHE
    image_01 = exposure.rescale_intensity(image, in_range='image', out_range=(0, 1))
    # Apply CLAHE
    clahe_image = exposure.equalize_adapthist(image_01, kernel_size=kernel_size)
    # Rescale back to original intensity range
    clahe_image = exposure.rescale_intensity(
        clahe_image,
        in_range='image',
        out_range=(np.amin(image), np.amax(image))
    )
    return clahe_image

def segment_kmeans(image_array, K=3):
    """
    Segment the image using K-means clustering.

    Parameters
    ----------
    image_array : ndarray
        Input image as a NumPy array.
    K : int, optional
        Number of clusters for K-means, by default 3.

    Returns
    -------
    ndarray
        Segmented image as a NumPy array.
    """
    pixels = image_array.flatten().reshape(-1, 1).astype(np.float32)
    _, labels, centers = cv2.kmeans(
        pixels,
        K,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    segmented = labels.reshape(image_array.shape)
    return segmented

def fill_chest_cavity(image):
    """
    Fill the chest cavity to obtain the final gantry mask.

    Parameters
    ----------
    image : ndarray
        Input binary image.

    Returns
    -------
    ndarray
        Chest cavity filled binary mask.
    """
    image = image.astype(np.uint8)
    filled_image = np.zeros_like(image)
    for i in range(image.shape[2]):  # Iterate over slices along z-axis
        slice_ = image[:, :, i]
        contours, _ = cv2.findContours(slice_, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(slice_.shape, dtype="uint8")
        area = [cv2.contourArea(contour) for contour in contours]
        if len(area) == 0:
            continue
        index_contour = area.index(max(area))
        cv2.drawContours(mask, contours, index_contour, 255, -1)
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        filled_image[:, :, i] = mask
    return filled_image / 255

def remove_gantry(input_array, visualize=False):
    """
    Creates a segmentation mask for the input CT image (as a NumPy array) and removes the gantry.

    Parameters
    ----------
    input_array : ndarray
        Input CT image as a 3D NumPy array (shape: x, y, z).
    visualize : bool, optional
        Whether to visualize intermediate steps and results, by default False.

    Returns
    -------
    tuple
        - gantry_removed_array: 3D NumPy array, the CT image with the gantry removed.
        - gantry_mask: 3D binary mask used for gantry removal.
    """
    # Rescale intensity to [0, 255]
    rescaled_array = cv2.normalize(input_array, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)

    # Perform K-means segmentation
    segmented_array = segment_kmeans(rescaled_array)
    # Calculate the mean intensity for each segment
    segment_means = [
        rescaled_array[segmented_array == i].mean() for i in range(3)  # Assuming 3 segments
    ]

    # Find the segment with the highest mean intensity
    highest_mean_segment = np.argmax(segment_means)

    # Find the segment with the middle mean intensity
    sorted_indices = np.argsort(segment_means)  # Indices of the sorted means
    middle_mean_segment = sorted_indices[1]    # Middle mean segment

    # Generate the gantry mask for the segment with the middle mean
    lung_segment = segmented_array * (segmented_array == middle_mean_segment)
    # Generate the gantry mask for the segment with the lowest mean
    gantry_mask = segmented_array * (segmented_array == highest_mean_segment)
    gantry_mask_filled = fill_chest_cavity(gantry_mask)

    # gantry_mask_eroded = np.zeros_like(gantry_mask_filled)
    # for z in range(gantry_mask_filled.shape[2]):
    #     gantry_mask_eroded[:,:,z] = cv2.erode(gantry_mask_filled.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)

    # Calculate the minimum of the input array
    min_value = input_array.min()

    # Remove the gantry: parts inside the mask are kept as input_array,
    # parts outside the mask are set to the minimum value
    gantry_removed_array = np.where(gantry_mask_filled, input_array, min_value)
    lung_mask = np.where(gantry_mask_filled, lung_segment, 0)

    if visualize:
        mid_slice = input_array.shape[2] // 2  # Middle slice along z-axis
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(input_array[:, :, mid_slice], cmap="gray")
        ax[0].set_title("Original Image")
        ax[1].imshow(gantry_mask_filled[:, :, mid_slice], cmap="gray")
        ax[1].set_title("Gantry Mask")
        ax[2].imshow(gantry_removed_array[:, :, mid_slice], cmap="gray")
        ax[2].set_title("Gantry Removed")
        plt.show()

    return gantry_removed_array, lung_mask

def flatten_peak_in_range(image, peak_range, target_range):
    """
    Flattens the non-dominant peak in a specified intensity range.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        peak_range (tuple): The range of the non-dominant peak (min, max).
        target_range (tuple): The target range to flatten the peak (min, max).

    Returns:
        numpy.ndarray: Image with the specified peak flattened to the target range.
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


