"""
Description: Create the model for the Mexican Sign Language 
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
First created: Monday 24 january, 2022
"""

import cv2
import numpy as np
import os
import mediapipe as mp
import LSM_utils as utils

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities


colors = [(245,117,16), (117,245,16), (16,117,245)]


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['A', 'B', 'C', 'D'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

start_folder = 0

for action in actions: 
    frames=os.listdir("../frames/{}".format(action))
    for sequence in range(1,int(len(frames)/sequence_length)+1):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass


#Create a VideoCapture object and read from input file
# Loop through actions

for action in actions:
    frames_videos=len(os.listdir("MP_Data/{}".format(action)))

    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        # NEW LOOP
        
            # Loop through sequences aka videos
            for sequence in range(1, frames_videos+1):
                # Loop through video length aka sequence length
                for frame_num in range(start_folder, start_folder+sequence_length):
                    
                    path_video = "../frames/{}/{}{}.jpg".format(action,action,frame_num)
                    cap = cv2.VideoCapture(path_video)

                    # Read feed
                    ret, frame = cap.read()
                    # Make detections
                    image, results = utils.mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
                    
                    # NEW Apply wait logic
                    if frame_num == start_folder: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                        #cv2.waitKey(500)
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        # Show to screen
                        cv2.imshow('OpenCV Feed', image)
                    
                    # NEW Export keypoints
                    keypoints = utils.extract_keypoints(results)
                    print("keypoints: {}".format(keypoints.shape))
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num-start_folder))
                    np.save(npy_path, keypoints)
                    start_folder = frame_num+1 if frame_num >= (start_folder+sequence_length-1) else start_folder
                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                start_folder = 0 if sequence==frames_videos else start_folder    
    cap.release()
    cv2.destroyAllWindows()