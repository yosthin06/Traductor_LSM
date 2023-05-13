# Import standard libraries
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import *



# Import user-defined libraries
import LSM_utils as utils


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

camera = cv2.VideoCapture(0)
with mp_holistic.Holistic(model_complexity=0,min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:

    while True:
        success, frame = camera.read()
        height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
        print(f"height:{height},width:{width}")
        if not success:
            break
        else:
            
            # Make detections
            frame, results = utils.mediapipe_detection(frame, holistic)
            # Draw landmarks
            utils.draw_styled_landmarks(frame, results, mp_drawing, mp_holistic)
            # Export keypoints
            keypoints = utils.extract_keypoints(results) 
            #keypoints = utils.scale_points(keypoints)
            x_points = keypoints[0::2]
            y_points = keypoints[1::2]
            x_max=int(np.max(x_points)*640)
            x_min=int(np.min(x_points)*640)
            y_max=int(np.max(x_points)*480)
            y_min=int(np.min(x_points)*480)
            print("xmin:{},xmax:{},ymin:{},ymax:{}".format(x_min,x_max,y_min,y_max))
            
            start_point = (x_min,y_min)
            
            # Ending coordinate, here (220, 220)
            # represents the bottom right corner of rectangle
            end_point = (x_max, y_max)
            # Blue color in BGR
            color = (255, 0, 0)
            
            # Line thickness of 2 px
            thickness = 2
            
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            frame=cv2.rectangle(frame, start_point, end_point, color, thickness)
            cv2.imshow('OpenCV Feed', frame)
                        
            # Break gracefully
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    camera.release()
    cv2.destroyAllWindows() 
            