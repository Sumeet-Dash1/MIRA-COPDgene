import os
import pandas as pd
import nibabel as nib
import numpy as np

def convert_img_to_nii_with_metadata(img_path, output_path, dimensions, spacing):
    """
    Converts a raw .img file to .nii.gz using provided dimensions and spacing.
    """
    try:
        # Load raw data from .img file
        img_data = np.fromfile(img_path, dtype=np.float32)  # Assuming float32 for raw image data
        img_data = img_data.reshape(dimensions)  # Reshape based on provided dimensions

        # Create an affine matrix based on spacing (voxel size)
        affine = np.diag(spacing + [1])  # Spacing for x, y, z and 1 for homogeneous coordinates

        # Create a NIfTI image
        nii_image = nib.Nifti1Image(img_data, affine)

        # Save as .nii.gz
        nib.save(nii_image, output_path)
        print(f"Converted: {img_path} -> {output_path}")
    except Exception as e:
        print(f"Error converting {img_path}: {e}")

# Load metadata
metadata_path = "/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/copd_metadata.csv"
metadata = pd.read_csv(metadata_path)

# Process each row in the metadata table
for index, row in metadata.iterrows():
    # Define paths for the input .img and output .nii.gz files
    img_path = f"/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/Data/{row['Label']}/{row['Label']}_eBHCT.img"  # Update base path as needed
    output_path = f"/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/Data/Data_nii/{row['Label']}/{row['Label']}_eBHCT.nii.gz"  # Update base path as needed

    # Extract dimensions and spacing from metadata
    dimensions = [row['image_dims0'], row['image_dims1'], row['image_dims2']]
    spacing = [row['vspacing0'], row['vspacing1'], row['vspacing2']]
    
    # Check if the .img file exists
    if not os.path.exists(img_path):
        print(f"Image file not found: {img_path}. Skipping...")
        continue

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Convert the .img file to .nii.gz
    convert_img_to_nii_with_metadata(img_path, output_path, dimensions, spacing)

print("Conversion completed.")