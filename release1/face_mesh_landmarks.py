import cv2  # Importing the OpenCV library for computer vision tasks

def landmark_detection(img, results, draw=False):
    """
    Detects landmarks on the input image and returns their mesh coordinates.
    
    Parameters:
        img: The input image.
        results: The results of the landmark detection.
        draw: Flag indicating whether to draw the landmarks on the image. Defaults to False.
    
    Returns:
        list: The mesh coordinates of the detected landmarks.
    """
    img_height, img_width = img.shape[:2]   # Extracting the height and width of the image
    mesh_coord = [(int(point.x*img_width), int(point.y*img_height)) for point in results.multi_face_landmarks[0].landmark]  # Calculating the coordinates of the facial landmarks relative(normalizing) to the image size
    if draw:
        [cv2.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]   # Drawing circles at the coordinates of the facial landmarks on the image
    return mesh_coord    # Returning the list of facial landmark coordinates

