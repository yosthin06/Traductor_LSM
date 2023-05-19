"""
Description: Extract keypoints and save them in a csv file
Author: Yosthin Galindo, Santiago Garza
Contact: yosthin.galindo@udem.edu, santiago.garzam@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import os
import mediapipe as mp
import time
import argparse

# Import user-defined libraries
import LSM_utils as utils

# Initialize the MediaPipe solutions and utilities
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Input the arguments and parse them
parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('-f','--folder', help='Path to the folder with the database videos')
args = parser.parse_args()

# Define the path to your video folder
# Example: "../LSM_database_videos"
video_folder = args.folder

# Path to the csv file
csv_path = "data_folder/LSM_database.csv"

# Set mediapipe model 
with mp_holistic.Holistic(model_complexity=1,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    
    # Get the list of subfolders in the video folder
    videos = os.listdir(video_folder)
    videos.sort()
    # Initialize time counter
    total = time.time()
    
    # Loop through each subfolder in the video folder
    for subfolder in videos:
        subfolder_path = os.path.join(video_folder, subfolder)

        # Loop through every hand folder (left and right)
        for hand in os.listdir(subfolder_path):
            hand_path = os.path.join(subfolder_path, hand)

            # Condition to dismiss any other type of file that is not a folder
            if os.path.isdir(hand_path):
                    
                # Loop through each video file in the hand subfolder
                for video_file in os.listdir(hand_path):
                    video_path = os.path.join(hand_path, video_file)
                    # Initialize the timer to get the time to extract keypoints
                    start = time.time()
                    # Read the video file
                    cap = cv2.VideoCapture(video_path)
                    width=100
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
                        # Make detections
                        image, results = utils.mediapipe_detection(frame, holistic)
                            
                        # Draw landmarks
                        utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

                        # Export keypoints
                        keypoints = utils.extract_keypoints(results) 

                        # Condition to get even number of frames that are multiples of 30
                        if count==(total_frames//30*30):
                            break
                        
                        # Condition to check the keypoints array has a value
                        if keypoints is not None:
                            keypoints = utils.scale_points(keypoints)
                            # Append the sign letter or word to the keypoints array
                            keypoints = np.append(subfolder,keypoints)
                            utils.toCSV(csv_path,keypoints, mp_holistic)
                            count+=1

                        # Show the image
                        cv2.imshow('OpenCV Feed', image)
                        
                        # Break gracefully
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    
                    # Release the image and destroy the windows
                    cap.release()
                    cv2.destroyAllWindows()  
                    # Print the time of every iteration of extracting keypoints  
                    print("keypoint time: {}".format((time.time()-start)))

# Print the total time for extracting keypoints    
print("total time: {}".format((time.time()-total)))