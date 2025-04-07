import pickle
import cv2
import mediapipe as mp
import numpy as np

#Load models
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

#Video capture initialization
cap = cv2.VideoCapture(0)

#MediaPipe hand detection solution
mediapipe_hands = mp.solutions.hands
mediapipe_drawing = mp.solutions.drawing_utils
mediapipe_drawing_styles = mp.solutions.drawing_styles

#Hand model with static image and low detection threshold
hands = mediapipe_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

#Label mapping for prediction
labels_dict = {0: 'A', 1: 'B', 2: 'T', 3: 'D', 4: 'S', 5: 'M', 6: 'N'}