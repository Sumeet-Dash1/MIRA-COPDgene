#!/bin/bash

# Set paths and parameter files
data_path="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/Data"
output_dir="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/processed"
param_affine="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/scripts/Par0009/Parameter.affine.txt"
param_elastic="/Users/sumeetdash/MAIA/Semester_3/CODES/MIRA/Final_Project_MIRA/MIRA-COPDgene/scripts/Par0009/Parameter.bsplines.txt"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each folder in the data directory
for folder in "$data_path"/copd*; do
    # Skip if not a directory
    if [[ ! -d "$folder" ]]; then
        continue
    fi

    # Extract the folder name (e.g., copd1, copd2, etc.)
    folder_name=$(basename "$folder")

    # Define paths to inhale and exhale images
    inhale_image="$folder/${folder_name}_iBHCT.img"
    exhale_image="$folder/${folder_name}_eBHCT.img"

    # Check if both inhale and exhale images exist
    if [[ ! -f "$inhale_image" ]]; then
        echo "Inhale image not found: $inhale_image. Skipping..."
        continue
    fi
    if [[ ! -f "$exhale_image" ]]; then
        echo "Exhale image not found: $exhale_image. Skipping..."
        continue
    fi

    # Define the specific output folder for this registration
    specific_output_dir="$output_dir/$folder_name/"
    mkdir -p "$specific_output_dir"

    # Run elastix with the inhale as fixed and exhale as moving
    ../elastix/elastix \
        -f "$inhale_image" \
        -m "$exhale_image" \
        -out "$specific_output_dir" \
        -p "$param_affine" \
        -p "$param_elastic"

    echo "Processed exhale to inhale for $folder_name, results stored in $specific_output_dir"
done