import math  # Importing the math module for mathematical operations

def euclidean_distance(point1, point2):
    """
    This function calculates the Euclidean distance between two points in a 2D space.

    Args:
        point1: A tuple representing the coordinates of the first point (x1, y1).
        point2: A tuple representing the coordinates of the second point (x2, y2).

    Returns:
        distance: The Euclidean distance between the two points.

    """
    x1, y1 = point1  # Extracting the x and y coordinates of the first point
    x2, y2 = point2  # Extracting the x and y coordinates of the second point

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)   # Calculating the Euclidean distance using the formula sqrt((x2 - x1)^2 + (y2 - y1)^2)

    return distance   # Returning the calculated Euclidean distance
