import cv2
import numpy as np
import math
import os
import mediapipe as mp
import LSM_utils as utils
import time
from tensorflow.keras.utils import to_categorical
import pandas as pd
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from itertools import chain
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.models import *
from tensorflow.keras.layers import LSTM, Dense, GRU

mp_hands = mp.solutions.hands
#mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define the path to your video folder
video_folder = "/home/yosthingc/Documents/VideosBaseDatosCompletaFinal/VideoBaseDatos2/Abecedario_Faltante"

csv_path = "/home/yosthingc/Documents/PEF_LSM/github/LSM_data_Hands_model.csv"

# Set mediapipe model 
with mp_hands.Hands(max_num_hands=1, min_tracking_confidence=0.3) as hands:
    # Loop through each subfolder in the video folder
    videos = os.listdir(video_folder)
    videos.sort()
    #scaler = MinMaxScaler()
    total = time.time()
    count_total=0
    
    for subfolder in videos:
        print("subfolder: ", subfolder)
        subfolder_path = os.path.join(video_folder, subfolder)
        for hand in os.listdir(subfolder_path):
            print("hand",hand)
            hand_path = os.path.join(subfolder_path, hand)
            if os.path.isdir(hand_path):
                    
                # Loop through each video file in the subfolder
                for video_file in os.listdir(hand_path):
                    print("video file:", video_file)
                    video_path = os.path.join(hand_path, video_file)
                    # Initialize the timer to get the time to extract keypoints
                    start = time.time()
                    # Read the video file
                    cap = cv2.VideoCapture(video_path)
                    width=100
                    # Initialize the keypoints list for every video
                    #keypoints_list = []
                    #frames = []
                    count=0
                    total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    print(total_frames)
                    # Loop through each frame in the video
                    while True:
                        ret, frame = cap.read()

                        if not ret:
                            print("break")
                            break
                        print(count)
                        frame,height = utils.resize_img(frame, cap,width)
                        if hand == 'right':
                            frame = cv2.flip(frame, 1)
                        #frames.append(frame)
                        # Make detections
                        image, results = utils.mediapipe_detection(frame, hands)
                        #print(results.multi_hand_landmarks)
                            
                        # Draw landmarks
                        utils.draw_styled_landmarks(image, results, mp_drawing, mp_hands)

                        # Export keypoints
                        keypoints = utils.extract_keypoints(results) 
                        #print(keypoints)                           
                        if count==(total_frames//30*30):

                            count_total+=count
                            break
                        if keypoints is not None:
                            keypoints = utils.scale_points(keypoints)
                            
                            keypoints = np.append(subfolder,keypoints)
                            utils.toCSV(csv_path,keypoints, mp_hands)
                            count+=1
                        cv2.imshow('OpenCV Feed', image)
                        
                        # Break gracefully
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    """if count%30!=0:
                        print("menor a 30")"""
                    cap.release()
                    cv2.destroyAllWindows()    
                    print("keypoint time: {}".format((time.time()-start)))
                
print("total time: {}".format((time.time()-total)))