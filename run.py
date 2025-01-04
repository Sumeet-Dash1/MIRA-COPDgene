import os
import SimpleITK as sitk
import numpy as np

def perform_registration(fixed_image_path, moving_image_path, parameter_file, output_transform_path):
    """
    Perform registration using SimpleITK and Elastix parameter file.

    Parameters:
        fixed_image_path (str): Path to the fixed image.
        moving_image_path (str): Path to the moving image.
        parameter_file (str): Path to the Elastix parameter file.
        output_transform_path (str): Path to save the resulting transform file.

    Returns:
        sitk.Transform: The resulting transformation.
    """
    # Load images
    fixed_image = sitk.ReadImage(fixed_image_path)
    moving_image = sitk.ReadImage(moving_image_path)
    
    # Initialize ElastixImageFilter
    elastix = sitk.ElastixImageFilter()
    elastix.SetFixedImage(fixed_image)
    elastix.SetMovingImage(moving_image)
    elastix.SetParameterMap(sitk.ReadParameterFile(parameter_file))
    
    # Perform registration
    elastix.Execute()
    
    # Save the transform file
    sitk.WriteParameterFile(elastix.GetTransformParameterMap()[0], output_transform_path)
    
    return elastix.GetTransformParameterMap()[0]

def transform_points(transform_file, points):
    """
    Transform points using the resulting transform.

    Parameters:
        transform_file (str): Path to the transform file.
        points (np.ndarray): Array of points (N x 3) to transform.

    Returns:
        np.ndarray: Transformed points.
    """
    # Read the transform file
    transform = sitk.ReadTransform(transform_file)
    
    # Apply transformation to each point
    transformed_points = np.array([transform.TransformPoint(pt) for pt in points])
    return transformed_points

def calculate_displacement(fixed_points, transformed_points):
    """
    Calculate displacement between fixed points and transformed points.

    Parameters:
        fixed_points (np.ndarray): Fixed image points (N x 3).
        transformed_points (np.ndarray): Transformed moving image points (N x 3).

    Returns:
        float: Mean displacement.
        float: Standard deviation of displacement.
    """
    displacements = np.linalg.norm(transformed_points - fixed_points, axis=1)
    return np.mean(displacements), np.std(displacements)

# Paths
fixed_image_path = "/Users/huytrq/Workspace/UdG/MIRA/Final/Training/copd1/copd1_iBHCT.img"
moving_image_path = "/Users/huytrq/Workspace/UdG/MIRA/Final/Training/copd1/copd1_eBHCT.img"
landmarks_fixed_path = "/Users/huytrq/Workspace/UdG/MIRA/Final/Training/copd1/copd1_300_iBH_xyz_r1.txt"
landmarks_moving_path = "/Users/huytrq/Workspace/UdG/MIRA/Final/Training/copd1/copd1_300_eBH_xyz_r1.txt"
model_dir = "/Users/huytrq/Workspace/UdG/MIRA/Final/ElastixModelZoo/models"
output_dir = "./simpleitk_results"

# Load landmarks
landmarks_fixed = np.loadtxt(landmarks_fixed_path)
landmarks_moving = np.loadtxt(landmarks_moving_path)

# Iterate through models
results = []
for root, _, files in os.walk(model_dir):
    for file in files:
        if file.endswith(".txt"):  # Parameter file
            parameter_file = os.path.join(root, file)
            output_transform_path = os.path.join(output_dir, f"{os.path.basename(file)}_transform.txt")
            
            print(f"Using model: {parameter_file}")
            
            # Perform registration
            transform_map = perform_registration(fixed_image_path, moving_image_path, parameter_file, output_transform_path)
            
            # Transform moving landmarks
            transformed_points = transform_points(output_transform_path, landmarks_moving)
            
            # Calculate displacement
            mean_displacement, std_displacement = calculate_displacement(landmarks_fixed, transformed_points)
            results.append((parameter_file, mean_displacement, std_displacement))
            print(f"Model: {parameter_file}, Mean Displacement: {mean_displacement:.3f}, Std: {std_displacement:.3f}")

# Find the best model
best_model = min(results, key=lambda x: x[1])
print(f"Best model: {best_model[0]}")
print(f"Mean Displacement: {best_model[1]:.3f}, Std: {best_model[2]:.3f}")