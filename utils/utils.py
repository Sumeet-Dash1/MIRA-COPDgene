import numpy as np

def parse_transformed_points(file_path, save_path=None):
    """
    Parses the input and transformed points from a Transformix points file.

    Parameters:
        file_path (str): Path to the Transformix points file. The file should contain lines with 
                         "InputIndex" and "OutputIndexFixed" that specify the points before and 
                         after transformation.
        save_path (str, optional): Path to save the transformed points as a text file.
                                   If None, the transformed points are not saved.

    Returns:
        tuple: A tuple of two numpy arrays:
            - input_points (numpy.ndarray): Array of input points (before transformation).
            - transformed_points (numpy.ndarray): Array of transformed points (after transformation).

    Notes:
        - The Transformix points file is expected to have lines in the format:
          `InputIndex = [x, y, z] OutputIndexFixed = [x', y', z']`.
        - The input and transformed points are extracted as integers and stored in separate arrays.
        - If `save_path` is provided, the transformed points are saved as a tab-delimited text file.
    """
    input_points = []
    transformed_points = []

    with open(file_path, 'r') as file:
        for line in file:
            if "InputIndex" in line and "OutputIndexFixed" in line:
                input_point = line.split("InputIndex = [")[1].split("]")[0].split()
                output_point = line.split("OutputIndexFixed = [")[1].split("]")[0].split()
                input_points.append([int(val) for val in input_point])
                transformed_points.append([int(val) for val in output_point])

    if save_path is not None:
        np.savetxt(save_path, transformed_points, delimiter='\t', fmt='%d')

    return np.array(input_points), np.array(transformed_points)
