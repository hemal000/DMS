import cv2  # Importing the OpenCV library for computer vision tasks
from euclidean_dist import euclidean_distance  # Importing the euclidean_distance function from a custom module

YELLOW = (0, 255, 255)  # Defining the color yellow as an RGB tuple

def mouth_aspect_ratio(img, landmarks, top_indices, bottom_indices):
    """
    This function calculates the mouth aspect ratio (MAR) using the landmarks of the mouth.

    Args:
        img: The input image on which the mouth aspect ratio is calculated.
        landmarks: A list of facial landmarks.
        top_indices: A list of indices representing the landmarks of the top lip.
        top_indices: A list of indices representing the landmarks of the top lip.
        bottom_indices: A list of indices representing the landmarks of the bottom lip.

    Returns:
        lip_distance: The mouth aspect ratio.

    """
    # vertical
    lip_top = landmarks[top_indices[4]]  # Topmost point of the top lip
    lip_bottom = landmarks[bottom_indices[5]]  # Bottommost point of the bottom lip

    #cv2.line(img, lip_top, lip_bottom, YELLOW, 2)  # Drawing a line on the mouth vertically

    # euclidean distance
    lip_distance = euclidean_distance(lip_top, lip_bottom)  # Calculating the vertical distance of the mouth

    return lip_distance  # Returning the mouth aspect ratio
