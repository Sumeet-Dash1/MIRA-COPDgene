import numpy as np

def parse_transformed_points(file_path):
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
            if "InputPoint" in line and "OutputPoint" in line:
                input_point = line.split("InputPoint = [")[1].split("]")[0].split()
                output_point = line.split("OutputPoint = [")[1].split("]")[0].split()
                input_points.append([float(val) for val in input_point])
                transformed_points.append([float(val) for val in output_point])

    return np.array(input_points), np.array(transformed_points)