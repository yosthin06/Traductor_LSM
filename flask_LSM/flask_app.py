"""
Description: Creation of the Flask app web
Author: Yosthin Galindo, Santiago Garza
Contact: yosthin.galindo@udem.edu, santiago.garzam@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import pandas as pd
from flask import Flask, render_template, Response, request
import sys
import tensorflow as tf
 
# Inserting the path to the previous folder to import utils
sys.path.insert(0, '..')

# Import user-defined libraries
import LSM_utils as utils

# Create the Flask app
app = Flask(__name__,template_folder='./templates')

# Initialize the MediaPipe solutions and utilities
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Initialize the dominant hand of the user as right hand
mano=0

# Set the device to use CPU for m1 Macs
tf.config.set_visible_devices([], 'GPU')

# Load the pretrained model
model = load_model("../data_folder/LSM_model.h5")

# Path to the csv file
csv_path = "../data_folder/LSM_database.csv"

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)

# Obtain the labels of the signs
labels = np.unique(df["Sign"])

# Function to capture the frames and process them with the model
def gen_frames():
    
    # Initialize the lists and variables needed
    sentence = []
    threshold = 0.90
    predictions=[]
    width=100
    
    # Start capturing the video
    camera = cv2.VideoCapture(0)
    with mp_holistic.Holistic(model_complexity=1,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
        
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                # Create a copy of the frame to show it at the end
                frame_normal = np.copy(frame)

                # Resize the frame to lower the processing time
                frame,height = utils.resize_img(frame, camera,width)

                # Condition to flip the frame if the dominant hand chosen is the right hand
                if mano==0:
                    frame = cv2.flip(frame, 1)

                # Make detections
                frame, results = utils.mediapipe_detection(frame, holistic)

                # Draw landmarks
                utils.draw_styled_landmarks(frame, results, mp_drawing, mp_holistic)

                # Extract the keypoints 
                keypoints = utils.extract_keypoints(results)
                
                # Condition to predict only when the chosen hand is detected
                if all(keypoints[30:72])!=0:
                    # Scale the points to the minimum and maximum points detected with MediaPipe
                    keypoints = np.array(utils.scale_points(keypoints)).flatten().reshape((1,72))
                
                    # Predict the gesture and store it in a list       
                    res = model.predict(np.expand_dims(keypoints, axis=0))[0]
                    predictions.append(np.argmax(res))

                    # Condition to print the prediction every 3 frames
                    if len(predictions)%3==0:
                        #  Condition to validate that the last 3 predictions are the same
                        if np.unique(predictions[-3:])[0]==np.argmax(res): 
                            # Condition to validate that the prediction is over the threshold
                            if res[np.argmax(res)] > threshold: 
                                sentence.append(labels[np.argmax(res)])
                                
                    # Condition to only show the last 10 predicted labels
                    if len(sentence) > 10: 
                        sentence = sentence[-10:]

                # Visualize the predicted label    
                cv2.rectangle(frame_normal, (0,0), (640, 40), (245, 117, 16), -1)
                cv2.putText(frame_normal, ' '.join(sentence), (3,30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Encode and buffer the frame to show it in the Flask app
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
        # If the "Derecha" button in the website is pressed the variable 
        # mano will be 0 indicating that is the right hand
        if request.form.get('derecha') == 'Derecha':
            global mano
            mano=0
        # If the "Izquierda" button in the website is pressed the variable 
        # mano will be 1 indicating that is the left hand
        elif  request.form.get('izquierda') == 'Izquierda':
            mano=1 
        if request.form.get('borrar') == 'Borrar':
            sentence = []       
     
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, port=8080, host='0.0.0.0')
