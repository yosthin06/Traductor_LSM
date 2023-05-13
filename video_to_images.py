import cv2
import os 

video_folder = "../../../base_de_datos_3"
videos = os.listdir(video_folder)
videos.sort()
label_map = {label:num for num, label in enumerate(videos)}


for subfolder in videos:
    print("subfolder: ", subfolder)
    os.makedirs(os.path.join('../frames2/',subfolder))
    subfolder_path = os.path.join(video_folder, subfolder)
    frameNr = 0
    # Loop through each video file in the subfolder
    for video_file in os.listdir(subfolder_path):
        print("video_file:", video_file)
        video_path = os.path.join(subfolder_path, video_file)
        cap = cv2.VideoCapture(video_path)
        i=0
        while True:
            ret, frame = cap.read()

            if not ret:
                break
            if i == 30:
                break
            cv2.imwrite(f'../frames2/{subfolder}/{subfolder}{frameNr}.jpg', frame)
            i += 1
            frameNr = frameNr+1

    cap.release()
    cv2.destroyAllWindows()