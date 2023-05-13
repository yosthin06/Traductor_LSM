from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from tensorflow.keras.models import *
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

# Import user-defined libraries
import LSM_utils as utils

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
#global  mano
#mano=0

model = load_model("/home/yosthingc/Documents/PEF_LSM/github/saved_data/my_best_model_2023-04-19 14:05:00.877168.h5")

csv_path = "/home/yosthingc/Documents/PEF_LSM/github/LSM_data_completa_2_con_mov.csv"

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)
#df=df.sort_values(by=["Sign"]).reset_index(drop=True)
#print(df)
labels = np.unique(df["Sign"])

class CamApp(App):
    def build(self):
        #global mano
        self.img1 = Image()
        layout = BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("Cv2 Image")
        Clock.schedule_interval(self.update,1.0/30.0)
        #print(mano)
        return layout
    
    def update(self,dt):
        global mano, sequence, sentence, threshold, predictions
        mano = 0
        sequence = []
        sentence = []
        threshold = 0.90
        predictions=[]
        frames=[]
        i=0
        width=100
        camera = self.capture
        # Fix indentation of the with block
        with mp_holistic.Holistic(model_complexity=0,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            #while True:
            success, frame = camera.read()
            if success:
            
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
                if np.unique(predictions[-5:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if labels[np.argmax(res)] != sentence[-1]:
                                sentence.append(labels[np.argmax(res)])
                        else:
                            sentence.append(labels[np.argmax(res)])

                # Condition to only show the last 5 predicted labels
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
                print(sentence)
                
                buf1=cv2.flip(frame,0)
                buf=buf1.tobytes()
                texture1=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
                #if working on raspberry pi use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer
                texture1.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
                #display image from the texture
                self.img1.texture = texture1
            
if __name__=='__main__':
    CamApp().run()