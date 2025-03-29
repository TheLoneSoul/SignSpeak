import cv2
import os

#dataset path
data_path = './data'

#checking & creating the data directory if it's not existed
if not os.path.exists(data_path):
    os.makedirs(data_path)

total_class = 26
data_size = 1000

#open default webcam to capture frames
cap = cv2.VideoCapture(0)

#loop to create directories for each class
for num in range(total_class):
    if not os.path.exists(os.path.join(data_path, str(num))):
        os.makedirs(os.path.join(data_path, str(num)))

    print('Collecting images for class {}'.format(num))

    done = False
    while True:
        #captures a frame from webcam
        ret, frame = cap.read()
        cv2.putText(frame, "Press 'C' to capture images..", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 4,
                    cv2.LINE_AA)
        #displays the current frame with the instructions
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('c'):
            break