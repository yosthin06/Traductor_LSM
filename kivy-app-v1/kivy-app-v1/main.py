import cv2
import numpy as np
import mediapipe as mp
import argparse
import tensorflow as tf
import os
from kivy.app import App
from kivy.uix.screenmanager import ScreenManager, Screen

class Home(Screen):
    pass

class Hand(Screen):
    pass

class Cam(Screen):
    def __init__(self, **kwargs):
        super(Cam, self).__init__(**kwargs)
        # Load the TensorFlow model
        self.model = tf.keras.models.load_model('model.h5')
        self.model.summary()
        mp_holistic = mp.solutions.holistic # Holistic model
        labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'R', 'S', 'T', 'U', 'V', 'W', 'Y']

class MyScreenManager(ScreenManager):
    pass

class LSMApp(App):
    def build(self):
        return MyScreenManager()

if __name__ == '__main__':
    LSMApp().run()
