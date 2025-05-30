import os
import pickle
import mediapipe as mp
import cv2
import matplotlib.pyplot as plt

#initialize mediapipe hand module
mediapipe_hands = mp.solutions.hands
mediapipe_drawing = mp.solutions.drawing_utils
mediapipe_drawing_styles = mp.solutions.drawing_styles

#hand object for static image processing with minimum confidence
hands = mediapipe_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#dataset path
data_path = './data'

#List to store extracted landmark data
data = []
labels = []

#Loop through each directory and each images in the current directory
for direct in os.listdir(data_path):
    for image_path in os.listdir(os.path.join(data_path, direct)):

        data_aux = [] #List for storing normalized landmark data of the images
        x_aux = [] #List to store x coordinates
        y_aux = [] #List to store y coordinates

        #Reads and converts the image in rgb
        image = cv2.imread(os.path.join(data_path, direct, image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #Process the image to detect hand landmarks
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_aux.append(x)
                    y_aux.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_aux))
                    data_aux.append(y - min(y_aux))

            if (len(data_aux) == 42):
                data.append(data_aux)
                labels.append(direct)

file = open('dataset.pickle', 'wb')
pickle.dump({'data': data, 'labels': labels}, file)
file.close()