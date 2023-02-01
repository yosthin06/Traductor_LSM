"""
Description: Create the model for the Mexican Sign Language 
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
First created: Monday 24 january, 2022
"""

import numpy as np
import os
import mediapipe as mp
from sklearn.model_selection import train_test_split
import tensorflow
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import TensorBoard
import LSM_utils as utils
import argparse
import time

parser = argparse.ArgumentParser(description='Enter the arguments')
parser.add_argument('--epochs', type=int, help='epochs for training')
parser.add_argument('--test_size', type=float, help='test size')
parser.add_argument('--train', type=bool, help='retrain model')

args = parser.parse_args()

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities



colors = [(245,117,16), (117,245,16), (16,117,245)]


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions = np.array(['A', 'B', 'C', 'D'])

# Thirty videos worth of data
no_sequences = 30

# Videos are going to be 30 frames in length
sequence_length = 30

start_folder = 0

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    frames_videos=len(os.listdir("MP_Data/{}".format(action)))
    for sequence in range(1, frames_videos+1):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, shuffle=True, stratify=y)
print("x_train: {}".format(X_train))
print("y_train: {}".format(y_train))

print("x_test: {}".format(X_test))
print("y_test: {}".format(y_test))

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

start = time.time()
model = utils.model_creation(actions, X_train, y_train, tb_callback, X_test, y_test, sequence_length, epochs=args.epochs, train=args.train)
print("model time: {}".format(time.time()-start))

