import cv2
import numpy as np
import mediapipe as mp
from face_mesh_landmarks import landmark_detection
from eye_ratio import eye_aspect_ratio
from mouth_ratio import mouth_aspect_ratio
import time

# initialize variables
is_distracted = False
is_yawning = False
is_eyeclosed = False
is_drowsy = False
distraction_start_time = time.time()
distraction_time_duration = 10
drowsy_start_time = time.time()
drowsy_time_duration = 5
yawning_start_time = time.time()
yawning_time_duration = 8
eyeclosed_start_time = time.time()
eyeclosed_time_duration = 3
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

roi_x = 100
roi_y = 100
roi_width = 400
roi_height = 300

mouth_threshold = 45
eye_threshold = 4.0
y_rotation = 15
eye_closed_threshold = 4.6

op = cv2.VideoWriter("./op001.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 20, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    roi_frame = frame[roi_y:roi_y + roi_height, roi_x:roi_x + roi_width]

    frame = cv2.rotate(roi_frame, cv2.ROTATE_180)

    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame.flags.writeable = False

    results = face_mesh.process(frame)

    frame.flags.writeable = True
    frame = cv2.rotate(frame, cv2.ROTATE_180)

    img_h, img_w, img_c = frame.shape

    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                if idx in RIGHT_EYE or idx in LEFT_EYE or idx in LOWER_LIPS or idx in UPPER_LIPS:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w
            cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                   [0, focal_length, img_w / 2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            ret, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
            rmat, jac = cv2.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360

            if y < -y_rotation or y > y_rotation:
                is_distracted = True
                if time.time() - distraction_start_time >= distraction_time_duration:
                    cv2.putText(frame, f"Distracted", (180, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                else:
                    pass
            else:
                is_distracted = False
                distraction_start_time = time.time()
            dist_timer_text = "dist: {:.1f}s".format(distraction_time_duration - (time.time() - distraction_start_time))
            cv2.putText(frame, f"Timer: {dist_timer_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if results.multi_face_landmarks:
                mesh_coords = landmark_detection(frame, results, False)
                eye_dist = eye_aspect_ratio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                mouth_dist = mouth_aspect_ratio(frame, mesh_coords, UPPER_LIPS, LOWER_LIPS)
                print(f"Eye_distance: {eye_dist}, Mouth_Dist: {mouth_dist}.")

                if eye_dist > eye_threshold:
                    text = "Drowsy"
                    is_drowsy = True
                    if time.time() - drowsy_start_time >= drowsy_time_duration:
                        cv2.putText(frame, f"Drowsy", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        pass
                else:
                    is_drowsy = False
                    drowsy_start_time = time.time()
                drowsy_timer_text = "Drowsy: {:.1f}s".format(drowsy_time_duration - (time.time() - drowsy_start_time))
                cv2.putText(frame, f"Timer: {drowsy_timer_text}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                if eye_dist > eye_closed_threshold:
                    is_eyeclosed = True
                    if time.time() - eyeclosed_start_time >= eyeclosed_time_duration:
                        cv2.putText(frame, f"Eye Closed", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    text = "Drowsy"
                    is_drowsy = True
                    if time.time() - drowsy_start_time >= drowsy_time_duration:
                         cv2.putText(frame, f"Drowsy", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                        # countinue counting drowsy time
                        pass
                else:
                    is_drowsy = False
                    drowsy_start_time = time.time()  # Reset the drowsy start time
                # Display the timer on the screen
                drowsy_timer_text = "Drowsy: {:.1f}s".format(drowsy_time_duration - (time.time() - drowsy_start_time))    
                cv2.putText(frame, f"Timer: {drowsy_timer_text}" , (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text to the frame
                
                if eye_dist > eye_closed_threshold:
                    #cv2.putText(frame, "Eye Closed", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    is_eyeclosed = True
                    if time.time() - eyeclosed_start_time >= eyeclosed_time_duration:
                        cv2.putText(frame, f"Eye Closed", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                         # countinue counting eye closed time
                         pass
                else: 
                    is_eyeclosed = False
                    eyeclosed_start_time = time.time()
                # Display the timer on the screen
                eyec_timer_text = "eyeC: {:.1f}s".format(eyeclosed_time_duration - (time.time() - eyeclosed_start_time))    
                cv2.putText(frame, f"Timer: {eyec_timer_text}" , (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text to the frame
                
                
                if mouth_dist > mouth_threshold:   # Checking if the mouth aspect ratio indicates yawning
                    #cv2.putText(frame, "Yawning", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    is_yawning = True
                    if time.time() - yawning_start_time > yawning_time_duration:
                        cv2.putText(frame, f"Yawning", (180, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    else:
                         # countinue counting yawn time
                         pass
                else:
                    is_yawning = False
                    yawning_start_time = time.time()
                    # Display the timer on the screen
                yawn_timer_text = "yawn: {:.1f}s".format(yawning_time_duration - (time.time() - yawning_start_time))    
                cv2.putText(frame, f"Timer: {yawn_timer_text}" , (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text to the frame
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
                cv2.putText(frame, f"Timer: {timer_text}" , (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # Adding text to the frame
                #cv2.putText(frame, text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
                    '''
    # Check if there are no face landmarks detected in the results
    if not results.multi_face_landmarks:
        cv2.putText(frame, "No Face Detected", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # Adding text to the frame
        pass

    op.write(frame)  # Writing the frame to the output video      
    cv2.imshow("Driver Behaviour System ", frame)  # Displaying the frame

    key = cv2.waitKey(24) & 0xFF  # Waiting for a key press   
    if key == ord('q'):  # Checking if the 'q' key is pressed
        break
op.release()
cap.release()  # Releasing the video capture object
cv2.destroyAllWindows()  # Closing all the window
