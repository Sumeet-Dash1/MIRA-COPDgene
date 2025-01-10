#!/bin/bash

# Set paths and parameter files
fixed_image="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/data/copd4/copd4_iBHCT.nii.gz"  # Replace with the actual path to the fixed image
moving_image="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/data/copd4/copd4_eBHCT.nii.gz"  # Replace with the actual path to the moving image
output_dir="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/processed"
param_affine="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/parameter/Par0056rigid.txt"
param_elastic="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/parameter/Par0016.multibsplines.lung.sliding_modified.txt"

fixed_mask="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/mask/copd4/copd4_iBHCT_lung_mask.nii.gz"  # Replace with the actual path to the fixed mask
moving_mask="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/mask/copd4/copd4_eBHCT_lung_mask.nii.gz"  # Replace with the actual path to the moving mask
# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Check if the fixed and moving images exist
if [[ ! -f "$fixed_image" ]]; then
    echo "Fixed image not found: $fixed_image. Exiting..."
    exit 1
fi
if [[ ! -f "$moving_image" ]]; then
    echo "Moving image not found: $moving_image. Exiting..."
    exit 1
fi

# Run elastix with the fixed and moving images
../elastix/elastix \
    -f "$fixed_image" \
    -m "$moving_image" \
    -out "$output_dir" \
    -p "$param_affine" \
    -p "$param_elastic" \
    -fMask "$fixed_mask" \
    -mMask "$moving_mask"

echo "Processed moving image to fixed image. Results stored in $output_dir"