import cv2
import os

#dataset path
data_path = './data'

#checking & creating the data directory if it's not existed
if not os.path.exists(data_path):
    os.makedirs(data_path)