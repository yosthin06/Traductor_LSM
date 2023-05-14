import cv2
import numpy as np
import math
import os
import mediapipe as mp
import LSM_utils as utils
import time
from tensorflow.keras.utils import to_categorical
import pandas as pd
import csv
import tensorflow as tf
from sklearn.model_selection import train_test_split
from itertools import chain
from keras.callbacks import ModelCheckpoint
from datetime import datetime
from tensorflow.keras.models import *
from tensorflow.keras.layers import LSTM, Dense, GRU

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define the path to your video folder
video_folder = "/home/yosthingc/Documents/VideosBaseDatosCompletaFinal/VideoBaseDatos2/Abecedario_Faltante"

csv_path = "/home/yosthingc/Documents/PEF_LSM/github/LSM_data_completa_2_con_mov.csv"

# Read CSV file for Training the model using Pandas
df = pd.read_csv(csv_path, header=0)

labels = np.unique(df["Sign"])

df["Sign"] = pd.Categorical(df["Sign"])
df["Sign"] = df.Sign.cat.codes

# Copy Label and Feature for training
y = df.pop("Sign")
x = df.copy()

# Copied Features turn to Array by using NumPy
x = np.array(x)
x = np.reshape(x, (x.shape[0],1, x.shape[1]))
num_classes=len(labels)
y = to_categorical(y, num_classes)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2,stratify=y)

model = Sequential()
model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(1,72)))
model.add(GRU(128, return_sequences=True, activation='relu'))
model.add(GRU(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))
model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.summary()
filepath = '../saved_data/my_best_model_{}.h5'.format(datetime.now())
checkpoint = ModelCheckpoint(filepath=filepath, 
                     monitor='loss',
                     verbose=1, 
                     save_best_only=True,
                     mode='min')
callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.05,verbose=1, patience=10)

model.fit(x_train, y_train, epochs=100, callbacks=[callback,checkpoint],validation_data=(x_test, y_test))