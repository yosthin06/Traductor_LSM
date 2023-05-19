"""
Description: Library for the functions of the Mexican Sign Language Translator
Author: Yosthin Galindo, Santiago Garza
Contact: yosthin.galindo@udem.edu, santiago.garzam@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import os
import csv
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import Dense, GRU
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from datetime import datetime

def mediapipe_detection(image, model):
    """
    Utilize the MediaPipe model selected to detect the keypoints
    """
    image.flags.writeable = False                  # Image is no longer writeable
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results, mp_drawing, mp_holistic):
    """
    Draw the keypoints detected by MediaPipe in the image
    """
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
    

def extract_keypoints(results):
    """
    Extract the desired keypoints and store them in an array
    """
    # Detect pose keypoints from 0 to 14
    count = 0
    pose=np.array([])
    if results.pose_landmarks:
        for res in results.pose_landmarks.landmark:
            pose = np.append(pose,np.array([[res.x, res.y]])).flatten()
            count += 1
            if count == 15:
                break
    else: pose=np.zeros(15*2)
    # Detect left hand keypoints
    lh = np.array([[res.x, res.y] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*2)
    # Return the two arrays of keypoints concatenated in one array
    return np.concatenate([pose, lh])

def column_names(model):
    """
    Create the list of the column names of the csv files
    """
    # Create Dataframe with the column names
    coords = ["x", "y"]

    ## List of of column names
    list_column_names = []
    # Append the sign letter or word
    list_column_names.append("Sign")

    # Pose 
    count=0
    # Accesing to the names of the pose keypoints through 0 to 14
    for landmark in model.PoseLandmark:
        for coord in coords:
            list_column_names.append(str(landmark) + "_" + str(count) + "_" + coord)
        count+=1
        if count == 15:
            break

    # Left Hand
    count=0
    # Accesing to the names of the left hand keypoints 
    for landmark in model.HandLandmark:
        for coord in coords:
            list_column_names.append("L"+str(landmark) + "_" + str(count) + "_" + coord)
        count+=1   
    
    return list_column_names

def resize_img(img,cap,new_width):
    """
    Resize the image to the desired size considering the aspect ratio
    """
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ratio = height/width
    new_height = int(ratio*new_width)
    resized_img = cv2.resize(img,(new_width,new_height))
    return resized_img, new_height

#Function to create CSV file or add dataset to the existed CSV file
def toCSV(filecsv,keypoints, mp_holistic):
    """
    Save the keypoints in a csv file
    """
    if os.path.isfile(filecsv):
        with open(filecsv, 'a+', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(keypoints)

    else:
        with open(filecsv, 'w', newline='') as file:
            # Create a writer object from csv module
            writer = csv.writer(file)
            list_column_names = column_names(mp_holistic)
            writer.writerow(list_column_names)
            writer.writerow(keypoints)

def scale_points(x):
    """
    Scale the keypoints respecting the minimum and maximum keypoint detected
    """
    x = x.astype("float")
    x_points = x[0::2]
    y_points = x[1::2]
    x_max=np.max(x_points)
    x_min=np.min(x_points)
    y_max=np.max(y_points)
    y_min=np.min(y_points)
    width = (x_max-x_min)
    height=(y_max-y_min)
    keypoints_x=np.divide(np.subtract(x_points,x_min),width)
    keypoints_y=np.divide(np.subtract(y_points,y_min),height)   
    return list(zip(keypoints_x, keypoints_y))

def model_creation(X_train, y_train, X_test, y_test, epochs, train, **pretrained_model):
    """
    Create or load the model for the LSM translation
    """
    if train==0:
        model = load_model(pretrained_model)
        model.summary()
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))
    
    else:
        
        model = Sequential()
        model.add(GRU(64, return_sequences=True, activation='relu', input_shape=(1,72)))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(GRU(64, return_sequences=False, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(32, activation='relu'))
        model.add(Dense(y_train.shape[1], activation='softmax'))
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        model.summary()
        filepath = 'data_folder/final_version2.h5'
        checkpoint = ModelCheckpoint(filepath=filepath, 
                            monitor='loss',
                            verbose=1, 
                            save_best_only=True,
                            mode='min')
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss',min_delta=0.05,verbose=1, patience=10)

        model.fit(X_train, y_train, epochs=epochs, callbacks=[callback,checkpoint],validation_data=(X_test, y_test))
        
        yhat = model.predict(X_test)
        ytrue = np.argmax(y_test, axis=1).tolist()
        yhat = np.argmax(yhat, axis=1).tolist()
        print(multilabel_confusion_matrix(ytrue, yhat))
        print(accuracy_score(ytrue, yhat))

        # Evaluate the model on the test data
        loss, accuracy = model.evaluate(X_test, y_test)

        # Print the test loss and accuracy
        print("Test loss:", loss)
        print("Test accuracy:", accuracy)


    return model

