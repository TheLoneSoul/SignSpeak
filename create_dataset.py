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
