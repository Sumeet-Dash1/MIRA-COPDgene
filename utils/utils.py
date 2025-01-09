import numpy as np

def parse_transformed_points(file_path, save_path=None):
    """
    Parse the input and output points from the Transformix points file.

    Parameters:
        file_path (str): Path to the Transformix points file.

    Returns:
        tuple: Numpy arrays of input and transformed points.
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
