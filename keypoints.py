"""
Description: Create the model for the Mexican Sign Language 
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
First created: Monday 24 january, 2022
"""

import cv2
import numpy as np
import math
import os
import mediapipe as mp
import LSM_utils as utils
import time
from tensorflow.keras.utils import to_categorical


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define the label mapping
label_map = {"A": 0,"B":1,"C":2,"D":3}

# Define the path to your video folder
video_folder = "../../../videos_cuadrados"

# Define the sequence length
seq_length = 30

# Initialize the training data and labels
train_data = []
train_labels = []
keypoints_list = []
labels = []
start = time.time()
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.4, min_tracking_confidence=0.4) as holistic:
    # Loop through each subfolder in the video folder
    videos = os.listdir(video_folder)
    videos.sort()
    #print("videos: ",videos)
    for subfolder in videos:
        subfolder_path = os.path.join(video_folder, subfolder)
        print("subfolder: ",subfolder)
        if os.path.isdir(subfolder_path):
            
            # Loop through each video file in the subfolder
            for video_file in os.listdir(subfolder_path):
                video_path = os.path.join(subfolder_path, video_file)
                print("video: ", video_path)
                # Read the video file
                cap = cv2.VideoCapture(video_path)
                frames = []
                                
                # Loop through each frame in the video
                while True:
                    ret, frame = cap.read()
                    
                    #print("h: {},w: {}".format(h,w))
                    if not ret:
                        break
                    # Resize the frame to 224x224 and add it to the list of frames
                    #frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    #frame = (frame - mean) / std
                    #frame = utils.resize_img(frame, height, width)
                    #h, w = frame.shape[:2]
                    frame=cv2.resize(frame, (480,480))
                    #h2, w2 = frame.shape[:2]
                    #print("h2: {},w2: {}".format(h2,w2))
                    frames.append(frame)
                    #print("resize:", math.floor(w/(h/height)))
                    #print("resize2:", math.floor(h/(w/width)))
                    # Make detections
                    image, results = utils.mediapipe_detection(frame, holistic)
                    # Draw landmarks
                    utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
                    #h3, w3 = image.shape[:2]
                    #print("h3: {},w3: {}".format(h3,w3))
                    # Export keypoints
                    keypoints = utils.extract_keypoints(results)
                    #print("keypoints",keypoints)
                    print("shape",keypoints.shape)
                    keypoints_list.append(keypoints)

                    cv2.imshow('OpenCV Feed', image)

                    # Break gracefully
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                while len(frames)%seq_length!=0:
                    frames.pop()
                    keypoints_list.pop()
                #print("frames: ",len(frames))
                #print("keypoints list:",len(keypoints_list))
                #print("labels:",labels)
                
                # Split the frames into sequences of length seq_length
                for i in range(0, len(frames), seq_length):
                    # NEW Export keypoints
                    #print("i:",i)
                    if len(frames)%seq_length==0:
                        
                        seq = keypoints_list[i:i+seq_length]
                        #print(len(seq))
                        
                        # Add the sequence and label to the training data and labels
                        train_data.append(seq)
                        #print(len(train_data))
                        # Add the label for the frame
                        labels.append(label_map[subfolder])
                        #print("labels len", len(labels))

    cap.release()
    cv2.destroyAllWindows()    

train_data = np.array(train_data)
print(train_data.shape)


# Convert the labels to categorical format
labels = to_categorical(labels).astype(int)
print(labels.shape)

# Save the keypoints
np.save("prueba_keypoints2", train_data)
np.save("prueba_labels2", labels)

print("keypoint time: {}".format((time.time()-start)))