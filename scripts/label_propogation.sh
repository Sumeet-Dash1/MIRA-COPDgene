# Set directories
fixed_image_dir="../Data/SPM/Input/Validation_Set/"
moving_image_dir="../Data/SPM/Input/Training_Set/"
registration_output_dir="../Data/SPM/processed/"
label_output_dir="../Data/transformed_labels/"
transformix_path="../elastix/transformix"

# Create the label output directory if it doesn't exist
mkdir -p "$label_output_dir"

# Loop through each fixed image directory
for fixed_folder in "$fixed_image_dir"IBSR_*; do
    fixed_base=$(basename "$fixed_folder")  # e.g., IBSR_11

    # Registration output for this fixed image
    fixed_output_dir="$registration_output_dir/$fixed_base"

    # Create a directory for transformed labels for this fixed image
    specific_label_output_dir="$label_output_dir/$fixed_base/"
    mkdir -p "$specific_label_output_dir"

    # Loop through each moving image directory
    for moving_folder in "$moving_image_dir"IBSR_*; do
        moving_base=$(basename "$moving_folder")  # e.g., IBSR_01

        # Skip if the moving and fixed images are the same
        if [[ "$fixed_base" == "$moving_base" ]]; then
            continue
        fi

        # Path to the labels file for the moving image
        moving_label="$moving_folder/${moving_base}_seg.nii"

        # Check if the labels file exists
        if [[ ! -f "$moving_label" ]]; then
            echo "Labels file not found for $moving_base. Skipping..."
            continue
        fi

        # Path to the TransformParameters file
        transform_param_file="$fixed_output_dir/${moving_base}_to_${fixed_base}/TransformParameters.1.txt"

        # Check if the TransformParameters file exists
        if [[ ! -f "$transform_param_file" ]]; then
            echo "TransformParameters file not found for $fixed_base -> $moving_base. Skipping..."
            continue
        fi

        # Define output for transformed label
        transformed_label_output_base="$specific_label_output_dir/${moving_base}_to_${fixed_base}_seg"

        # Run transformix to apply the transformation to the label file
        "$transformix_path" \
            -in "$moving_label" \
            -out "$specific_label_output_dir" \
            -tp "$transform_param_file"

        # Move the resulting files (result.hdr and result.img) to the final output path
        mv "$specific_label_output_dir/result.hdr" "${transformed_label_output_base}.hdr"
        mv "$specific_label_output_dir/result.img" "${transformed_label_output_base}.img"

        echo "Transformed label for $moving_base -> $fixed_base saved as ${transformed_label_output_base}.hdr and ${transformed_label_output_base}.img"
    done
done