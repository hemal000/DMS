import cv2  # Importing the OpenCV library for computer vision tasks
from euclidean_dist import euclidean_distance  # Importing the euclidean_distance function from a custom module

BLUE = (255, 0, 0)  # Defining the color blue as an RGB tuple

def eye_aspect_ratio(img, landmarks, right_indices, left_indices):
    """
    This function calculates the eye aspect ratio (EAR) using the landmarks of the eyes.

    Args:
        img: The input image on which the eye aspect ratio is calculated.
        landmarks: A list of facial landmarks.
        right_indices: A list of indices representing the landmarks of the right eye.
        left_indices: A list of indices representing the landmarks of the left eye.

    Returns:
        ratio: The eye aspect ratio.

    """
    # right eye
    # horizontal line
    rh_right = landmarks[right_indices[0]]  # Rightmost point of the right eye
    rh_left = landmarks[right_indices[8]]  # Leftmost point of the right eye
    # vertical line
    rv_top = landmarks[right_indices[12]]  # Topmost point of the right eye
    rv_bottom = landmarks[right_indices[4]]  # Bottommost point of the right eye

    #cv2.line(img, rh_right, rh_left, BLUE, 2)  # Drawing a line on the right eye horizontally
    #cv2.line(img, rv_top, rv_bottom, BLUE, 2)  # Drawing a line on the right eye vertically

    # left eye
    # horizontal line
    lh_right = landmarks[left_indices[0]]  # Rightmost point of the left eye
    lh_left = landmarks[left_indices[8]]  # Leftmost point of the left eye
    # vertical line
    lv_top = landmarks[left_indices[12]]  # Topmost point of the left eye
    lv_bottom = landmarks[left_indices[4]]  # Bottommost point of the left eye

    #cv2.line(img, lh_left, lh_right, BLUE, 2)  # Drawing a line on the left eye horizontally
    #cv2.line(img, lv_top, lv_bottom, BLUE, 2)  # Drawing a line on the left eye vertically

    # euclidean distance
    rh_distance = euclidean_distance(rh_right, rh_left)  # Calculating the horizontal distance of the right eye
    rv_distance = euclidean_distance(rv_top, rv_bottom)  # Calculating the vertical distance of the right eye

    lh_distance = euclidean_distance(lh_right, lh_left)  # Calculating the horizontal distance of the left eye
    lv_distance = euclidean_distance(lv_top, lv_bottom)  # Calculating the vertical distance of the left eye
    
    if lh_distance != 0 and lv_distance != 0:
        re_ratio = rh_distance / rv_distance  # Calculating the eye aspect ratio of the right eye
        le_ratio = lh_distance / lv_distance  # Calculating the eye aspect ratio of the left eye
    
    ratio = (re_ratio + le_ratio)/2  # Calculating the average eye aspect ratio
    return ratio  # Returning the eye aspect ratio
