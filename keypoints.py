"""
Description: Extract keypoints for the model
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import os
import mediapipe as mp
import time
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import argparse

# Import user-defined libraries
import LSM_utils as utils

# Initialize the MediaPipe Holistic model and drawing utils
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define the label mapping
label_map = {"A": 0,"B":1,"C":2,"D":3, 'None':4}

# Define the path to your video folder
# Input the arguments and parse them
parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('-v','--videos', type=str, help='path to videos')
args = parser.parse_args()
video_folder = "{}".format(args.videos)

# Define the sequence length
seq_length = 30

# Initialize the training data and labels
train_data = []
labels = []

# Initialize the timer to get the time to extract keypoints
start = time.time()

# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # Loop through each subfolder in the video folder
    videos = os.listdir(video_folder)
    videos.sort()

    for subfolder in videos:

        subfolder_path = os.path.join(video_folder, subfolder)

        if os.path.isdir(subfolder_path):
            
            # Loop through each video file in the subfolder
            for video_file in os.listdir(subfolder_path):
                video_path = os.path.join(subfolder_path, video_file)

                # Read the video file
                cap = cv2.VideoCapture(video_path)

                # Initialize the keypoints list for every video
                keypoints_list = []
                                
                # Loop through each frame in the video
                while True:
                    ret, frame = cap.read()
                  
                    if not ret:
                        break
                 
                    # Make detections
                    image, results = utils.mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
               
                    # Export keypoints
                    keypoints = utils.extract_keypoints(results)
                    keypoints_list.append(keypoints)

                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Remove the last keypoints of the list to make it even to 30 frames
                while len(keypoints_list)%seq_length!=0:
                    keypoints_list.pop()

                # Split the frames into sequences of length seq_length
                for i in range(0, len(keypoints_list), seq_length):
                    # Store the keypoints in 30 frame sequences
                    seq = keypoints_list[i:i+seq_length]

                    # Add the sequence to the training data
                    train_data.append(seq)

                    # Add the label for the sequence
                    labels.append(label_map[subfolder])
                    

    cap.release()
    cv2.destroyAllWindows()    



print("keypoint time: {}".format((time.time()-start)))

# Add the training data to an array
train_data = np.array(train_data)

# Convert the labels to categorical format
labels = to_categorical(labels).astype(int)

# Path to the keypoints with the date and time
path_keypoints = "../saved_data/keypoints_{}".format(datetime.now())
path_labels = "../saved_data/labels_{}".format(datetime.now())

# Save the keypoints
np.save(path_keypoints, train_data)
np.save(path_labels, labels)

