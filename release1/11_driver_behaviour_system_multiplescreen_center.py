# import libraries
import cv2   # Importing the OpenCV library for computer vision tasks
import numpy as np  # Importing the NumPy library for numerical operations
import mediapipe as mp  # Importing the Mediapipe library for face mesh detection
from face_mesh_landmarks import landmark_detection   # Importing a custom function for landmark detection
from eye_ratio import eye_aspect_ratio  # Importing a custom function for calculating eye aspect ratio
from mouth_ratio import mouth_aspect_ratio  # Importing a custom function for calculating mouth aspect ratio
import time

# initialize variables
is_distracted = False  # Flag to indicate if the driver is distracted 
is_yawning = False     # Flag to indicate if the driver is yawning
is_eyeclosed = False   # Flag to indicate if the driver eyes is closed
is_drowsy = False      # Flag to indicate if the driver is drowsy
start_time = time.time()
time_duration = 10
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]  # Indices of landmarks for the right eye 
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]  # Indices of landmarks for the left eye
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]  # Indices of landmarks for the lower lips
UPPER_LIPS =[185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]  # Indices of landmarks for the for the upper lips


# load the pre trained face mesh 
mp_face_mesh = mp.solutions.face_mesh  # Creating a face mesh object for detecting facial landmarks
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # Create a face mesh object with minimum detection and tracking confidence thresholds

cap = cv2.VideoCapture(0)  # Opening a video file for capturing frames (or) Web cam

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Getting the width of the video frame
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Getting the height of the video frame

# define the ROI coordinates
roi_x = 100  # X-coordinate of the top-left corner of the ROI
roi_y = 100  # Y-coordinate of the top-left corner of the ROI
roi_width = 400  # width of the ROI
roi_height = 300  # height of the ROIq

######################################################################################################
# define the threshold value for the parameters
mouth_threshold = 13.5  # mouth threshold value
eye_threshold = 3.5  # eye threshold value
left_threshold = -18  # yaw rotation along y axis threshold value
right_threshold = 10  # yaw rotation along y axis threshold value
eye_closed_threshold = 4  # eye closed threshold value
######################################################################################################

# store the video
op = cv2.VideoWriter("./op001.avi", cv2.VideoWriter_fourcc('X','V','I','D'), 20, (frame_width, frame_height))   # Creating a video writer object for saving the output video

while cap.isOpened():
    ret, frame = cap.read()  # Reading a frame from the video
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if not ret:
        break  # If there are no more frames to read, exit the loop

    roi_frame = frame[roi_y:roi_y+roi_height, roi_x:roi_x+roi_width]   # Crop the frame to the ROI size
    frame_with_mesh = cv2.cvtColor(cv2.flip(roi_frame, 1), cv2.COLOR_BGR2RGB)  # Flipping the frame horizontally and converting color space from BGR to RGB
    frame_without_mesh = roi_frame.copy()
    frame_without_mesh = cv2.cvtColor(cv2.flip(roi_frame, 1), cv2.COLOR_BGR2RGB) 
    frame_with_mesh.flags.writeable = False   # Setting the writeable flag of the frame to False for improving performance
    results = face_mesh.process(frame_with_mesh)   # Processing the frame to detect facial landmarks using the face mesh model
    frame_with_mesh.flags.writeable = True   # Setting the writeable flag of the frame back to True 
    # convert from RGB to BGR
    frame_with_mesh = cv2.cvtColor(frame_with_mesh, cv2.COLOR_RGB2BGR)  # Converting the color space of the frame from RGB to BGR
    frame_without_mesh = cv2.cvtColor(frame_without_mesh, cv2.COLOR_RGB2BGR) 
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
                print(f"Eye_distance: {eye_dist}, Mouth_Dist: {mouth_dist}.")  # Printing Eye aspect ratio and Mouth aspect ratio

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

                if is_distracted:
                    cv2.putText(frame_with_mesh, "Distracted", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                elif is_yawning:
                    cv2.putText(frame_with_mesh, "Yawning", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                elif is_drowsy:
                    cv2.putText(frame_with_mesh, "Drowsy", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                elif is_eyeclosed:
                    cv2.putText(frame_with_mesh, "Eye Closed", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame

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
        cv2.putText(frame_with_mesh, "No Face Detected", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
        pass
    
    display_frame = np.hstack((frame_with_mesh, frame_without_mesh))
    op.write(display_frame)  # Writing the frame to the output video
    cv2.namedWindow("Driver Behaviour System ", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Driver Behaviour System ", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.imshow("Driver Behaviour System ", display_frame)  # Displaying the frame

    key = cv2.waitKey(24) & 0xFF  # Waiting for a key press   
    if key == ord('q'):  # Checking if the 'q' key is pressed
        break
op.release()
cap.release()  # Releasing the video capture object
cv2.destroyAllWindows()  # Closing all the window



  
