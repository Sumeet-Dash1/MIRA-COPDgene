import os
from lungmask import LMInferer
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import nibabel as nib
from utils.preprocessing import hu_data, convert_signed_4bit_to_hu, convert_to_signed_4bit, remove_gantry
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import clear_border
import nrrd

def apply_lungmask_binary(image, output_path=None, model_name=None, vis=False):
    """
    Applies the LungMask model (default U-Net R231) to segment lungs from a 3D CT image
    and converts the output into a binary mask (1: Lung, 0: Background).

    Parameters:
        image_path (str): Path to the 3D CT image file (e.g., .nii or .nii.gz).
        output_path (str, optional): Path to save the binary lung mask. If None, the mask is not saved.
        model_name (str): The specific model to use ('R231' or 'R231CovidWeb'). If None, the default model is used.
        plot (bool): Whether to plot the original image and the binary lung mask (default: False).

    Returns:
        numpy.ndarray: Binary lung mask (3D array) with:
                       - 1: Lung (left or right)
                       - 0: Background
    """
    # Load the CT image using SimpleITK

    if isinstance(image, str):
        image = nib.load(image).get_fdata()


    hu_image = hu_data(image)  # Convert to numpy array (z, y, x)
    bit_4 = convert_to_signed_4bit(hu_image)
    bit_4_hu = convert_signed_4bit_to_hu(bit_4)

    # Initialize the lung mask inferer with the specified model
    if model_name is None:
        inferer = LMInferer()
    else:
        inferer = LMInferer(model=model_name)

    # Apply the model to segment the lungs
    segmentation = inferer.apply(np.transpose(bit_4_hu))

    segmentation = np.transpose(segmentation)
    # Convert the segmentation into a binary lung mask
    lung_mask = np.where(segmentation > 0, 1, 0).astype(np.uint8)  # 1: Lung, 0: Background

    # Save the binary lung mask if an output path is specified
    if output_path:
        # Create a NIfTI image using the lung mask and the original image's affine and header
        lung_mask_image = nib.Nifti1Image(lung_mask, affine=image.affine, header=image.header)
        
        # Save the NIfTI image
        nib.save(lung_mask_image, output_path)

    # Plot if requested
    if vis:
        slice_index = image.shape[2] // 2  # Middle slice
        plt.figure(figsize=(12, 6))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, slice_index], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Binary Lung Mask
        plt.subplot(1, 2, 2)
        plt.imshow(lung_mask[:, :, slice_index], cmap='gray')
        plt.title('Binary Lung Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return lung_mask

def apply_lungmask_thresholded(image, output_path=None, vis=False, axis_3 = False, axis_2 = False):
    """
    Applies the LungMask model (default U-Net R231) to segment lungs from a 3D CT image
    and converts the output into a binary mask (1: Lung, 0: Background).

    Parameters:
        image_path (str): Path to the 3D CT image file (e.g., .nii or .nii.gz).
        output_path (str, optional): Path to save the binary lung mask. If None, the mask is not saved.
        plot (bool): Whether to plot the original image and the binary lung mask (default: False).

    Returns:
        numpy.ndarray: Binary lung mask (3D array) with:
                       - 1: Lung (left or right)
                       - 0: Background
    """

    if isinstance(image, str):
        image = nib.load(image).get_fdata()  # Read image if it's a file path

    hu_image = hu_data(image) 
    bit_4 = convert_to_signed_4bit(hu_image)
    bit_4_hu = convert_signed_4bit_to_hu(bit_4)

    _, thresholded = cv.threshold(bit_4_hu, -400, 1, cv.THRESH_BINARY_INV)

    borderless = np.zeros_like(thresholded, dtype=np.uint8)
    for z in range(thresholded.shape[2]):
        borderless[:, :, z] = clear_border(thresholded[:,:,z])
        borderless[:, :, z] = binary_fill_holes(borderless[:, :, z])
        borderless[:, :, z] = cv.morphologyEx(borderless[:, :, z], cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
        borderless[:, :, z] = cv.morphologyEx(borderless[:, :, z], cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
    
    if axis_2 == True:
        for y in range(thresholded.shape[1]):
            borderless[:, y, :] = cv.morphologyEx(borderless[:, y, :], cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3)))
            borderless[:, y, :] = binary_fill_holes(borderless[:, y, :])

    if axis_3 == True:
        for x in range(thresholded.shape[0]):
            borderless[x, :, :] = cv.morphologyEx(borderless[x, :, :], cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11)))
            borderless[x, :, :] = binary_fill_holes(borderless[x, :, :])

    lung_mask = borderless
    # Save the binary lung mask if an output path is specified
    if output_path:
        # Create a NIfTI image using the lung mask and the original image's affine and header
        lung_mask_image = nib.Nifti1Image(lung_mask, affine=image.affine, header=image.header)
        
        # Save the NIfTI image
        nib.save(lung_mask_image, output_path)

    # Plot if requested
    if vis:
        slice_index = image.shape[2] // 2  # Middle slice
        plt.figure(figsize=(12, 6))
        
        # Original Image
        plt.subplot(1, 2, 1)
        plt.imshow(image[:, :, slice_index], cmap='gray')
        plt.title('Original Image')
        plt.axis('off')
        
        # Binary Lung Mask
        plt.subplot(1, 2, 2)
        plt.imshow(lung_mask[:, :, slice_index], cmap='gray')
        plt.title('Binary Lung Mask')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    return lung_mask

def convert_nrrd_to_nii_gz(nrrd_file, reference_nii_file, output_nii_file=None):
    """
    Converts a mask from a .nrrd file to a .nii.gz file using affine information
    from a reference .nii.gz file. Saves the result only if output_nii_file is provided.

    Parameters:
        nrrd_file (str): Path to the input .nrrd file.
        reference_nii_file (str): Path to the reference .nii.gz file.
        output_nii_file (str, optional): Path to save the output .nii.gz file. If None, does not save.

    Returns:
        nib.Nifti1Image: The NIfTI image object created from the .nrrd file.
    """
    # Load the .nrrd file
    mask_data, mask_header = nrrd.read(nrrd_file)

    mask_in_all = mask_data[0,:,:,:]
    mask_in_right = mask_data[1,:,:,:]
    mask_in_left = mask_data[2,:,:,:]
    mask_without_trachea = np.bitwise_and(mask_in_left, mask_in_right)
    mask_with_trachea = np.where(mask_in_all>0, 1, 0).astype(np.uint8)
        
    # Load the reference .nii.gz file
    reference_nii = nib.load(reference_nii_file)
    affine = reference_nii.affine  # Get affine transformation

    # Convert the mask data to a NIfTI image
    mask_with_trachea = nib.Nifti1Image(mask_with_trachea, affine=affine)
    mask_without_trachea = nib.Nifti1Image(mask_without_trachea, affine=affine)

    # Save the NIfTI image if output_nii_file is provided
    if output_nii_file:
        nib.save(mask_with_trachea, os.path.join(output_nii_file, "mask_with_trachea.nii.gz"))
        nib.save(mask_without_trachea, output_nii_file, "mask_without_trachea.nii.gz")
        print(f"Converted {nrrd_file} to {output_nii_file} using reference {reference_nii_file}")
    else:
        print(f"NIfTI image created but not saved (no output path provided).")

    return mask_with_trachea, mask_without_trachea