from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
import cv2
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
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
        drawing_spec = mp_drawing.DrawingSpec(thickness=1,circle_radius=1)
        cap = self.capture
        with mp_face_mesh.FaceMesh(
                max_num_faces=3,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as face_mesh:
            ret, image = cap.read()
            image.flags.writeable = False
            image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

            results= face_mesh.process(image)

            #Draw the face mesh annotations on the image
            image.flags.writeable = True
            image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
                    mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
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

