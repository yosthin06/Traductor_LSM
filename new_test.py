# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tensorflow.keras.models import *
import os
import pandas as pd
 

# Import user-defined libraries
import LSM_utils as utils

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define the path to your video folder
video_folder = "/home/yosthingc/Documents/VideosBaseDatosCompletaFinal/VideoBaseDatos2/Abecedario_Faltante"

csv_path = "/home/yosthingc/Documents/PEF_LSM/github/LSM_data_completa_2_con_mov.csv"

# Load the pretrained model
model = load_model("/home/yosthingc/Documents/PEF_LSM/github/saved_data/my_best_model_2023-04-19 14:05:00.877168.h5")
model.summary()

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)
#df=df.sort_values(by=["Sign"]).reset_index(drop=True)
#print(df)
labels = np.unique(df["Sign"])
print(labels)

# Set the colors for the probabilities
colors = [(245,117,16), (117,245,16), (16,117,245), (218, 247, 166), (255, 195, 0 )]

# Initialize lists and variables
sequence = []
sentence = []
threshold = 0.90
predictions=[]
frames=[]
i=0
width=200
x=True
while x==True:
    print("Elija: \n0. Derecha\n1. Izquierda")
    mano = int(input())
    if mano == 0:
        print("derecha")
        x=False
    elif mano==1:
        print("izquierda")
        x=False
    else:
        print("ERROR, vuelva a intentar")
# Start to capture the real-time feed from the camera
cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)
print("fps:", fps)
# Set mediapipe model 
with mp_holistic.Holistic(model_complexity=0,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
    if mano == 0:
        while cap.isOpened():


            # Read feed
            ret, frame = cap.read()
            if not ret:
                break
            frame_normal = np.copy(frame)
            #frame = cv2.resize(frame,(100,177))
            frame,height = utils.resize_img(frame, cap,width)
            frame = cv2.flip(frame, 1)
            #frames.append(frame)
            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Draw landmarks
            #utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

            # Prediction logic
            keypoints = utils.extract_keypoints(results)
            keypoints = np.array(utils.scale_points(keypoints,width,height)).flatten()
            
            #keypoints = np.reshape(keypoints, (keypoints.shape[0],1, keypoints.shape[1]))
            sequence.append(keypoints)
            sequence = sequence[-1:]
            #print(len(sequence))

            # Condition to predict only when 30 frames has passed        
            #if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            print(np.argmax(res))
            print(labels[np.argmax(res)],res[np.argmax(res)])


            #  Prediction logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if labels[np.argmax(res)] != sentence[-1]:
                            sentence.append(labels[np.argmax(res)])
                    else:
                        sentence.append(labels[np.argmax(res)])

            # Condition to only show the last 5 predicted labels
            if len(sentence) > 5: 
                sentence = sentence[-5:]

                # Visualize probabilities
                #image = prob_viz(res, videos, image, colors)

            # Visualize the predicted label    
            cv2.rectangle(frame_normal, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame_normal, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', frame_normal)

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        while cap.isOpened():


            # Read feed
            ret, frame = cap.read()
            if not ret:
                break
            frame_normal = np.copy(frame)
            frame,height = utils.resize_img(frame, cap,width)
            #frames.append(frame)
            # Make detections
            image, results = utils.mediapipe_detection(frame, holistic)

            # Draw landmarks
            #utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)

            # Prediction logic
            keypoints = utils.extract_keypoints(results)
            keypoints = np.array(utils.scale_points(keypoints,width,height)).flatten()
            sequence.append(keypoints)
            sequence = sequence[-1:]

            # Condition to predict only when 30 frames has passed        
            #if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            print(labels[np.argmax(res)],res[np.argmax(res)])


            #  Prediction logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if labels[np.argmax(res)] != sentence[-1]:
                            sentence.append(labels[np.argmax(res)])
                    else:
                        sentence.append(labels[np.argmax(res)])

                # Condition to only show the last 5 predicted labels
                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Visualize probabilities
                #image = prob_viz(res, videos, image, colors)

            # Visualize the predicted label    
            cv2.rectangle(frame_normal, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(frame_normal, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # Show to screen
            cv2.imshow('OpenCV Feed', frame_normal)

            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()