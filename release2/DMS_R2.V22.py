"""
This version updated 
Abnormal notefication is added
"""

# import libraries
import cv2   # Importing the OpenCV library for computer vision tasks
import numpy as np  # Importing the NumPy library for numerical operations
import mediapipe as mp  # Importing the Mediapipe library for face mesh detection
import time
import math

# initialize variables
is_distracted = False  # Flag to indicate if the driver is distracted 
is_yawning = False     # Flag to indicate if the driver is yawning
is_eyeclosed = False   # Flag to indicate if the driver eyes is closed
is_drowsy = False      # Flag to indicate if the driver is drowsy
distance = False
start_time = time.time()
time_duration = 10
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  # Indices of landmarks for the right eye 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]  # Indices of landmarks for the left eye
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]  # Indices of landmarks for the lower lips
UPPER_LIPS =[185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]  # Indices of landmarks for the for the upper lips
FACE_TOP =[10]
FACE_BOTTOM =[152]

YELLOW = (0, 255, 255)  # Defining the color yellow as an RGB tuple

# load the pre trained face mesh 
mp_face_mesh = mp.solutions.face_mesh  # Creating a face mesh object for detecting facial landmarks
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Create a face mesh object with minimum detection and tracking confidence thresholds

cap = cv2.VideoCapture(2)  # Opening a video file for capturing frames (or) Web cam

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Getting the width of the video frame
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Getting the height of the video frame

# define the ROI coordinates
roi_x = 1  # X-coordinate of the top-left corner of the ROI
roi_y = 1  # Y-coordinate of the top-left corner of the ROI
roi_width = 640 # width of the ROI
roi_height = 480  # height of the ROIq

######################################################################################################
# Text position percentages (adjust as needed)
bottom_right_x_percent = 0.85  # 5% from the right edge
bottom_right_y_percent = 0.98  # 5% from the bottom edge

bottom_left_x_percent = 0.05
bottom_left_y_percent = 0.89

top_left_x_percent = 0.05
top_left_y_percent = 0.07

left_margin_percent = 0.05  # 5% from the left edge for left text
right_margin_percent = 0.95  # 5% from the right edge for right text
######################################################################################################
# Calculate text positions based on ROI dimensions
bottom_right_x = int(roi_width * bottom_right_x_percent)
bottom_right_y = int(roi_height * bottom_right_y_percent)

bottom_left_x = int(roi_width * bottom_left_x_percent)
bottom_left_y = int(roi_height * bottom_left_y_percent)

top_left_x = int(roi_width*top_left_x_percent)
top_left_y = int(roi_height*top_left_y_percent)

left_margin = int(roi_width * left_margin_percent)
right_margin = int(roi_width * right_margin_percent)
######################################################################################################
# define the threshold value for the parameters

mouth_threshold = 38  # mouth threshold value
eye_threshold = 4  # eye threshold value
left_threshold = -17  # yaw rotation along y axis threshold value
right_threshold = 17  # yaw rotation along y axis threshold value
eye_closed_threshold = 3.4  # eye closed threshold value
face_length = 90
######################################################################################################

"""

# define the threshold value for the parameters
mouth_threshold = float(input("TY: "))   # mouth threshold value
eye_threshold = float(input("TE:"))  # eye threshold value
left_threshold = float(input("TDL:"))  # yaw rotation along y axis threshold value
right_threshold = float(input("TDR:"))  # yaw rotation along y axis threshold value
eye_closed_threshold = float(input("TEC:"))  # eye closed threshold value
######################################################################################################
"""




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


###########################################################################################################
def face_distance_ratio(img,landmarks,top_face_indices,bottom_face_indices):
    #vertical
    face_top = landmarks[top_face_indices[0]]
    face_bottom = landmarks[bottom_face_indices[0]]

    face = euclidean_distance(face_top,face_bottom)

    return face

###########################################################################################################




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
    
    re_ratio =0
    le_ratio =0
    
    if lh_distance != 0 and lv_distance != 0 and rh_distance !=0 and rv_distance !=0:
        re_ratio = rh_distance / rv_distance  # Calculating the eye aspect ratio of the right eye
        le_ratio = lh_distance / lv_distance  # Calculating the eye aspect ratio of the left eye
    
    ratio = (re_ratio + le_ratio)/2  # Calculating the average eye aspect ratio
    return ratio  # Returning the eye aspect ratio

def calculate_fps(i: int, t0: float, frame: any, show_fps: bool=False) -> float:
    """
    Calculate frames per second (FPS) and optionally display it on the frame.

    Args:
        i (int): Frame count.
        t0 (float): Start time.
        frame (any): Frame to display FPS on (if enabled).
        show_fps (bool): Flag to indicate whether to display FPS on the frame.

    Returns:
        float: Frames per second (FPS).
    """
    time.sleep(0.001)   # To prevent zero division error when calculating the FPS
    t_now = time.perf_counter()     # Get the current time
    
    fps = i / (t_now - t0)  # Calculate FPS
    # If FPS is 0, set it to 10 (to prevent division by zero)
    if fps == 0:
        fps = 10
    
    # If enabled, display FPS on the frame
    if show_fps:
        print(f"fps{round(fps)}")
    return fps

# store the video
op = cv2.VideoWriter("./op001.avi", cv2.VideoWriter_fourcc('X','V','I','D'), 20, (frame_width, frame_height))   # Creating a video writer object for saving the output video

while cap.isOpened():
    ret, frame = cap.read()  # Reading a frame from the video
    #frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        break  # If there are no more frames to read, exit the loop

    roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]   # Crop the frame to the ROI size
    frame_with_mesh = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2RGB)  # Flipping the frame horizontally and converting color space from BGR to RGB
    frame_without_mesh = roi_frame.copy()
    frame_with_mesh.flags.writeable = False   # Setting the writeable flag of the frame to False for improving performance
    results = face_mesh.process(frame_with_mesh)   # Processing the frame to detect facial landmarks using the face mesh model
    frame_with_mesh.flags.writeable = True   # Setting the writeable flag of the frame back to True 
    # convert from RGB to BGR
    frame_with_mesh = cv2.cvtColor(frame_with_mesh, cv2.COLOR_RGB2BGR)  # Converting the color space of the frame from RGB to BGR
    img_h, img_w, img_c = frame_with_mesh.shape  # Getting the height, width, and number of channels of the frame
    
    face_3d = []  # List to store 3D coordinates of facial landmarks
    face_2d = []  # List to store 2D coordinates of facial landmarks

    # Check if there are multiple face landmarks detected in the results
    if results.multi_face_landmarks:
        # Iterate over each face landmark detected
        for face_landmarks in results.multi_face_landmarks:
            # Iterate over each landmark point in the face
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                    x, y = int(lm.x*img_w), int(lm.y*img_h)  # Converting normalized coordinates to pixel coordinates
                    face_2d.append([x, y])   # Adding the 2D coordinates to the list
                    face_3d.append([x, y, lm.z])  # Adding the 3D coordinates to the list

            face_2d = np.array(face_2d, dtype=np.float64)  # Converting the list of 2D coordinates to a NumPy array
            face_3d = np.array(face_3d, dtype=np.float64)  # Converting the list of 3D coordinates to a NumPy array
            ##########################################################################################################
            i = 0
            t0 = time.perf_counter()
            # Simulated frame
            args = {"show_fps": True}  # Example argument, replace it with your actual argument dictionary
            fps = calculate_fps(i, t0, frame, args["show_fps"])  # Calculate FPS

            #cv2.putText(img=frame, text=f"FPS:{str(round(fps))}", org=(380,390), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=(0, 255, 0), thickness=2)
            cv2.putText(frame_with_mesh, f"FPS:{str(round(fps))}", (bottom_right_x, bottom_right_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            ##########################################################################################################


            # the camera matrix
            focal_length = 1 * img_w  # Calculating the focal length based on the image width
            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0, 0, 1]])   # Creating the camera matrix
            
            dist_matrix = np.zeros((4, 1), dtype=np.float64)  # Creating a zero-filled distance matrix
            ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)  # Solving the PnP problem to estimate the pose of the face
            rmat, jac = cv2.Rodrigues(rot_vec)  # Converting the rotation vector to a rotation matrix
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)  # Decomposing the rotation matrix to obtain Euler angles

            x = angles[0] * 360  # Calculating the rotation degree around the x-axis
            y = angles[1] * 360  # Calculating the rotation degree around the y-axis
            print(f"x:{x},y:{y}")  # Printing the rotation degrees

            #cv2.putText(frame_with_mesh, f"X:{x:.2f}", (10, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #cv2.putText(frame_with_mesh, f"Y:{y:.2f}", (10, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

            cv2.putText(frame_with_mesh, f"DB:",(top_left_x,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            cv2.putText(frame_with_mesh, f"X:{x:.2f}", (bottom_left_x, bottom_left_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(frame_with_mesh, f"Y:{y:.2f}", (bottom_left_x,bottom_left_y + 0 ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)



            if y < left_threshold or y > right_threshold:  # Checking if the rotation degree around the y-axis indicates distraction 
                is_distracted = True
                #cv2.putText(frame_with_mesh, "Distracted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
            else:
                is_distracted = False
            
            frame_height, frame_width = frame.shape[:2]  # Getting the height and width of the frame


            # Check if there are multiple face landmarks detected in the results
            if results.multi_face_landmarks:
                mesh_coords = landmark_detection(frame_with_mesh, results, True)  # Detecting and drawing landmarks on the frame
                eye_dist = eye_aspect_ratio(frame_with_mesh, mesh_coords, RIGHT_EYE, LEFT_EYE)  # Calculating the eye aspect ratio
                mouth_dist = mouth_aspect_ratio(frame_with_mesh, mesh_coords, UPPER_LIPS, LOWER_LIPS)  # Calculating the mouth aspect ratio
                face_distance = face_distance_ratio(frame_with_mesh,mesh_coords,FACE_TOP,FACE_BOTTOM )

                print(f"face_distance: {face_distance}")
                cv2.putText(frame_with_mesh, f"FD:{face_distance:.2f}", (bottom_left_x, bottom_left_y -40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,215,0), 2)

                print(f"Eye_distance: {eye_dist}, Mouth_Dist: {mouth_dist}.")  # Printing Eye aspect ratio and Mouth aspect ratio

                #cv2.putText(frame_with_mesh, f"M:{mouth_dist:.2f}", (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                #cv2.putText(frame_with_mesh, f"E:{eye_dist:.2f}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 20, 255), 2)

                cv2.putText(frame_with_mesh, f"E:{eye_dist:.2f}", (bottom_left_x, bottom_left_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(frame_with_mesh, f"M:{mouth_dist:.2f}", (bottom_left_x,bottom_left_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (147, 20, 255), 2)

                if eye_dist > eye_threshold and eye_dist < eye_closed_threshold:  # Checking if the eye aspect ratio indicates drowsiness
                    #cv2.putText(frame, "Drowsy", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    text = "Drowsy"
                    is_drowsy = True
                    pass
                else:
                    is_drowsy = False
                if eye_dist > eye_closed_threshold:
                    #cv2.putText(frame, "Eye Closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    is_eyeclosed = True
                    text = "Eye Closed"
                    pass
                else: 
                    is_eyeclosed = False
                    pass
 
                if mouth_dist > mouth_threshold:   # Checking if the mouth aspect ratio indicates yawning
                    #cv2.putText(frame, "Yawning", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    text = "Yawning"
                    is_yawning = True
                    pass
                else:
                    is_yawning = False

                if face_distance <= face_length :
                    cv2.putText(frame_with_mesh, "Abnormal", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                    distance = True
                    pass
                else:
                    distance = False



                if is_distracted:
                    cv2.putText(frame_with_mesh, "Distracted", (top_left_x + 28,top_left_y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    
                elif is_yawning:
                    cv2.putText(frame_with_mesh, "Yawning", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                elif is_drowsy:
                    cv2.putText(frame_with_mesh, "Drowsy", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                elif is_eyeclosed:
                    cv2.putText(frame_with_mesh, "Eye Closed", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame

                # Draw rectangles on the output frame based on the calculated coordinates
                if not is_yawning and not is_eyeclosed and not is_drowsy and not is_distracted and not distance :
                    cv2.putText(frame_with_mesh, "Normal", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)


                '''
                if is_distracted or is_yawning or is_eyeclosed or is_drowsy:
                # Check if the timer has expired
                    if time.time() - start_time >= time_duration:
                        # Perform the desired action when the timer expires
                        print("Alert: Distracted for more than 10 seconds!")
                        cv2.putText(frame, f"{text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2) 
                    else:
                        # Continue counting the time
                        pass
                else:
                    # Reset the timer
                    start_time = time.time()    
                # Display the timer on the screen
                timer_text = "Timer: {:.1f}s".format(time_duration - (time.time() - start_time))    
                #cv2.putText(frame, f"Timer: {timer_text}" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text to the frame
                #cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                '''
    # Check if there are no face landmarks detected in the results
    if not results.multi_face_landmarks:
        cv2.putText(frame_with_mesh, "No Face Detected", (top_left_x + 28,top_left_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
        pass
    
    display_frame = np.hstack((frame_with_mesh, frame_without_mesh))
    op.write(display_frame)  # Writing the frame to the output video
    cv2.namedWindow("Driver Behaviour System ", cv2.WINDOW_NORMAL)

    cv2.rectangle(display_frame, (roi_x, roi_y), (roi_x + roi_width, roi_y + roi_height), (255, 0, 0), 2)  # ROI rectangle

    cv2.setWindowProperty("Driver Behaviour System ", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Driver Behaviour System ", display_frame)  # Displaying the frame

    key = cv2.waitKey(24) & 0xFF  # Waiting for a key press   
    if key == ord('q'):  # Checking if the 'q' key is pressed
        break

op.release()
cap.release()  # Releasing the video capture object
cv2.destroyAllWindows()  # Closing all the window