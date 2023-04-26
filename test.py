"""
Description: Library for the functions of the Mexican Sign Language Translator
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tensorflow.keras.models import *
import os

# Import user-defined libraries
import LSM_utils as utils

# Input the arguments and parse them
parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('-m','--model', type=str, help='location of the pretrained model')
args = parser.parse_args()

# Load the pretrained model
#model = load_model('../saved_data/{}'.format(args.model))
model = load_model("../saved_data/{}".format(args.model))
model.summary()

# Initialize the actions list
#actions = np.array(['A','B','C','D','None'])
video_folder = "../../../base_de_datos_3"
videos = os.listdir(video_folder)
videos.sort()


# Initialise the MediaPipe model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Set the colors for the probabilities
colors = [(245,117,16), (117,245,16), (16,117,245), (218, 247, 166), (255, 195, 0 )]

# Initialize lists and variables
sequence = []
sentence = []
threshold = 0.95
predictions=[]
frames=[]
i=0
# Start to capture the real-time feed from the camera
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
# Set mediapipe model 
with mp_holistic.Holistic(model_complexity=2,min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()
        if not ret:
            break
        frame_normal = np.copy(frame)
        #frame = utils.resize_img(frame, 100, cap)
        frames.append(frame)
        print("len frames: ",len(frames))
        i+=1
        print("i: ",i)
        # Make detections
        image, results = utils.mediapipe_detection(frame, holistic)
        
        # Draw landmarks
        utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
        
        # Prediction logic
        keypoints = utils.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Condition to predict only when 30 frames has passed        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            print(videos[np.argmax(res)],res[np.argmax(res)])
            
            
        #  Prediction logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if videos[np.argmax(res)] != sentence[-1]:
                            sentence.append(videos[np.argmax(res)])
                    else:
                        sentence.append(videos[np.argmax(res)])

            # Condition to only show the last 5 predicted labels
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Visualize probabilities
            image = utils.prob_viz(res, videos, image, colors)

        # Visualize the predicted label    
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    