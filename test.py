"""
Description: Library for the functions of the Mexican Sign Language Translator
Author: Yosthin Galindo
Contact: yosthin.galindo@udem.edu
"""

# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp

# Import user-defined libraries
import LSM_utils as utils
 
# Load the pretrained model
model = load_model('my_best_model_solo_manos.h5')
model.summary()

# Initialize the actions list
actions = np.array(['A','B','C','D','None'])

# Initialise the MediaPipe model
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Set the colors for the probabilities
colors = [(245,117,16), (117,245,16), (16,117,245)]

# Initialize lists and variables
sequence = []
sentence = []
threshold = 0.9
predictions=[]

# Start to capture the real-time feed from the camera
cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = utils.mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        utils.draw_styled_landmarks(image, results, mp_drawing, mp_holistic)
        
        # Prediction logic
        keypoints = utils.extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]

        # Condition to predict only when 30 frames has passed        
        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            predictions.append(np.argmax(res))
            
            
        #  Prediction logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    if len(sentence) > 0: 
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            # Condition to only show the last 5 predicted labels
            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Visualize probabilities
            image = utils.prob_viz(res, actions, image, colors)

        # Visualize the predicted label    
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    