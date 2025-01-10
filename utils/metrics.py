import numpy as np

def target_registration_error(fixed_points, transformed_points, voxel_size):
    """
    Calculate the Target Registration Error (T.R.E.) between fixed points and transformed points.

    Parameters:
        fixed_points (np.ndarray): Array of fixed image points with shape (N, 3),
                                   where N is the number of points, and each point has 3 coordinates.
        transformed_points (np.ndarray): Array of transformed moving image points with shape (N, 3).
        voxel_size (tuple): The voxel size of the image in physical units (e.g., [1.0, 1.0, 1.0]).

    Returns:
        tuple: A tuple containing:
            - float: Mean T.R.E. in physical units (e.g., millimeters).
            - float: Standard deviation of T.R.E. in physical units.

    Notes:
        - T.R.E. is computed as the Euclidean distance between corresponding fixed and transformed points.
        - The calculation accounts for voxel size to provide the error in physical units.
    """
    # Convert points from voxel units to physical units
    fixed_points_phys = fixed_points * np.array(voxel_size)
    transformed_points_phys = transformed_points * np.array(voxel_size)
    
    # Calculate the distance between each pair of points in physical units
    distances_phys = np.linalg.norm(fixed_points_phys - transformed_points_phys, axis=1)
    
    # Calculate the mean and standard deviation of the distances
    tre_mean = np.mean(distances_phys)
    tre_std = np.std(distances_phys)
    return tre_mean, tre_std

if __name__ == "__main__":
    ## Check Target Registration Errors Metric
    import json
    dataset_info = json.load(open("./data/dataset_info.json"))
    for key, value in dataset_info.items():
        print(f"Processing {key}")
        
        ## Read the metadata
        size = value["size"]
        spacing = value["spacing"]
        disp_mean = value["disp_mean"]
        disp_std = value["disp_std"]
        print(f"Information: Size: {size}, Spacing: {spacing}, Displacement: Mean - {disp_mean} Std - {disp_std}")
        
        inhale_landmarks_path = f"./data/{key}/{key}_300_iBH_xyz_r1.txt"
        exhale_landmarks_path = f"./data/{key}/{key}_300_eBH_xyz_r1.txt"
        landmarks_inhale = np.loadtxt(inhale_landmarks_path)
        landmarks_exhale = np.loadtxt(exhale_landmarks_path)
        
        print(f"Landmarks: Inhale - {landmarks_inhale.shape}, Exhale - {landmarks_exhale.shape}")
    
        tre_mean, tre_std = target_registration_error(landmarks_inhale, landmarks_exhale, spacing)
        print(f"Mean T.R.E.: {tre_mean:.2f} mm")
        print(f"Std. Dev. T.R.E.: {tre_std:.2f} mm")
        print("-" * 50)