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