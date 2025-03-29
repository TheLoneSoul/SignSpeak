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