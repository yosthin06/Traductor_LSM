import kivy
import cv2
import mediapipe as mp
from kivy.app import App
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.clock import Clock

class HandTracker(Image):
    def _init_(self, capture, fps, **kwargs):
        super(HandTracker, self)._init_(**kwargs)
        self.capture = capture
        self.fps = fps
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
        Clock.schedule_interval(self.update, 1.0 / self.fps)

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(frame)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        cx, cy = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), cv2.FILLED)
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='rgb')
            texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
            self.texture = texture

class HandTrackingApp(App):
    def build(self):
        capture = cv2.VideoCapture(0)
        self.hand_tracker = HandTracker(capture, fps=30)
        return self.hand_tracker

    #def on_stop(self):
        #self.hand_tracker.hands.close()

if __name__ == '_main_':
    HandTrackingApp().run()