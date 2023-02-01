import cv2
import os
import numpy as np
import LSM_utils as utils
#capture = cv2.VideoCapture('videos_LSM/D/D1.mp4')
 


actions = np.array(['A', 'B', 'C', 'D'])

data_augmentation = utils.data_augmenter()

if os.path.exists("../../Traductor_LSM/frames"):
    for action in actions: 
        frameNr = 0
        videos=os.listdir("new_videos/{}".format(action))
        start_folder = len(os.listdir("PEF_LSM/github/frames/{}".format(action)))
        print(start_folder)
        for num_video in range(1,len(videos)+1):
            path_video = "new_videos/{}/{}{}.mp4".format(action,action,num_video)
            cap = cv2.VideoCapture(path_video)
            while(True):
                #for video in videos:
                success, frame = cap.read()
                print("success {}".format(success))
                if success:
                    cv2.imwrite(f'../../Traductor_LSM/frames/{action}/{action}{frameNr+start_folder-1}.jpg', frame)
            
                else:
                    break
            
                frameNr = frameNr+1

        print("success in {}".format(action))
        cap.release()
        cv2.destroyAllWindows()

else:
    for action in actions: 
        frameNr = 0
        videos=os.listdir("new_videos/{}".format(action))
        start_folder = len(os.listdir("PEF_LSM/github/frames/{}".format(action)))
        print(start_folder)
        for num_video in range(1,len(videos)+1):
            path_video = "new_videos/{}/{}{}.mp4".format(action,action,num_video)
            cap = cv2.VideoCapture(path_video)
            while(True):
                #for video in videos:
                success, frame = cap.read()
                print("success {}".format(success))
                if success:
                    cv2.imwrite(f'../../Traductor_LSM/frames/{action}/{action}{frameNr+start_folder-1}.jpg', frame)
            
                else:
                    break
            
                frameNr = frameNr+1

        print("success in {}".format(action))
        cap.release()
        cv2.destroyAllWindows()