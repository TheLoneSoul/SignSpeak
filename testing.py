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

#Check if camera is open/close
while cap.isOpened():
    # Prepare auxiliary lists for normalized landmark data
    data_aux = []
    x_aux = []
    y_aux = []

    # Read a frame from webcam
    ret, frame = cap.read()

    # Get frame dimensions
    H, W, _ = frame.shape

    # Convert frame to RGB as required by MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand landmarks
    results = hands.process(frame_rgb)

    # Draws detected hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mediapipe_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mediapipe_hands.HAND_CONNECTIONS,  # hand connections
                mediapipe_drawing_styles.get_default_hand_landmarks_style(),
                mediapipe_drawing_styles.get_default_hand_connections_style())

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