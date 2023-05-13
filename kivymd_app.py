from kivymd.app import MDApp
#from kivymd.uix.widget import Widget
from kivymd.uix.boxlayout import MDBoxLayout
from kivy.uix.image import Image
from kivymd.uix.button import MDRaisedButton
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp

# Import user-defined libraries
import LSM_utils as utils

#mp_holistic = mp.solutions.holistic # Holistic model
#mp_drawing = mp.solutions.drawing_utils # Drawing utilities

class CamApp(MDApp):
    
    def build(self):
        self.img1=Image()
        layout=MDBoxLayout(orientation='vertical')
        layout.add_widget(self.img1)
        layout.add_widget(MDRaisedButton(
            text="Click here",
            pos_hint={"center_x":0.5,"center_y":0.5},
            size_hint=(None,None)
        ))
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("Cv2 Image")
        Clock.schedule_interval(self.load_video,1.0/30)
        return layout
    
    def load_video(self,*args):
        self.mp_holistic = mp.solutions.hands
        self.holistic = self.mp_holistic.Hands(static_image_mode=False,max_num_hands=1,min_detection_confidence=0.2, min_tracking_confidence=0.2)

        #while True:
            #success, frame = self.capture.read()
        #with mp_holistic.Holistic(model_complexity=0,min_detection_confidence=0.2, min_tracking_confidence=0.2) as holistic:
        ret, frame = self.capture.read()
        if ret:
            frame=cv2.flip(frame,0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            #frame,height = utils.resize_img(frame, self.capture,100)
            # Make detections
            #frame, results = utils.mediapipe_detection(frame, self.holistic)
            # Draw landmarks
            #utils.draw_styled_landmarks(frame, results, mp_drawing, mp_holistic)

            #Frame initialize
            #self.image_frame = frame
            buf=frame.tobytes()
            texture1=Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='rgb')
            #if working on raspberry pi use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer
            texture1.blit_buffer(buf,colorfmt='rgb',bufferfmt='ubyte')
            #display image from the texture
            self.img1.texture = texture1
            
"""
    def update(self, dt):
        #for webcam input
        cap = self.capture
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            #while True:
            ret, image = cap.read()
                
            image.flags.writeable = False
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            results= holistic.process(image)

            #Draw the face mesh annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                    
                #Draw pose connections
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                        )
                # Draw left hand connections
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                        ) 

            
    
            buf1=cv2.flip(image,0)
            buf=buf1.tobytes()
            texture1=Texture.create(size=(image.shape[1],image.shape[0]),colorfmt='bgr')
            #if working on raspberry pi use colorfmt='rgba' here instead, but stick with "bgr" in blit_buffer
            texture1.blit_buffer(buf,colorfmt='bgr',bufferfmt='ubyte')
            #display image from the texture
            self.img1.texture = texture1
"""
if __name__=='__main__':
    CamApp().run()
    cv2.destroyAllWindows()
