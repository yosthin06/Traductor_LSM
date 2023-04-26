from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp

# Import user-defined libraries
import LSM_utils as utils

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles # Drawing styles
mp_face_mesh = mp.solutions.face_mesh

class CamApp(App):
    
    def build(self):
        self.img1=Image()
        layout=BoxLayout()
        layout.add_widget(self.img1)
        #opencv2 stuffs
        self.capture = cv2.VideoCapture(0)
        #cv2.namedWindow("Cv2 Image")
        Clock.schedule_interval(self.update,1.0/33.0)
        return layout
    
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

if __name__=='__main__':
    CamApp().run()
    cv2.destroyAllWindows()

