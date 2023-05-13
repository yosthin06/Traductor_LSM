# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp
import argparse
from tensorflow.keras.models import *
import os
import pandas as pd
from flask import Flask, render_template, Response, request

# Import user-defined libraries
import LSM_utils as utils

app = Flask(__name__,template_folder='./templates'  )


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

global switch, mano,camera
mano=0
switch=1
print("init", switch)
# Load the pretrained model
model = load_model("/home/yosthingc/Documents/PEF_LSM/github/saved_data/my_best_model_2023-04-19 14:05:00.877168.h5")

csv_path = "/home/yosthingc/Documents/PEF_LSM/github/LSM_data_completa_2_con_mov.csv"

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)
#df=df.sort_values(by=["Sign"]).reset_index(drop=True)
#print(df)
labels = np.unique(df["Sign"])
print(labels)


def gen_frames():
    
    sequence = []
    sentence = []
    threshold = 0.90
    predictions=[]
    frames=[]
    i=0
    width=100
    
    camera = cv2.VideoCapture(0)
    with mp_holistic.Holistic(model_complexity=0,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:

        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame_normal = np.copy(frame)
            
                frame,height = utils.resize_img(frame, camera,width)
                if mano==0:
                    frame = cv2.flip(frame, 1)
                # Make detections
                frame, results = utils.mediapipe_detection(frame, holistic)
                # Draw landmarks
                utils.draw_styled_landmarks(frame, results, mp_drawing, mp_holistic)
                # Prediction logic
                keypoints = utils.extract_keypoints(results)
                #print("before:\n{}".format(keypoints))
                keypoints = np.array(utils.scale_points(keypoints)).flatten()
                sequence.append(keypoints)
                sequence = sequence[-1:]

                # Condition to predict only when 30 frames has passed        
                #if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                predictions.append(np.argmax(res))
                #print(np.argmax(res))
                #print(labels[np.argmax(res)],res[np.argmax(res)])
                #  Prediction logic
                if np.unique(predictions[-1:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if labels[np.argmax(res)] != sentence[-1]:
                                sentence.append(labels[np.argmax(res)])
                        else:
                            sentence.append(labels[np.argmax(res)])

                # Condition to only show the last 5 predicted labels
                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Visualize the predicted label    
                cv2.rectangle(frame_normal, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame_normal, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                #print("after:\n{}".format(keypoints))
                ret, buffer = cv2.imencode('.jpg', frame_normal)
                
                frame_normal = buffer.tobytes()
                
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame_normal + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch, camera
    if request.method == 'POST':
        if request.form.get('derecha') == 'Derecha':
            global mano
            mano=0
        elif  request.form.get('izquierda') == 'Izquierda':
            mano=1 
        
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
