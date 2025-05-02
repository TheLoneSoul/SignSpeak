import pickle
import cv2
import mediapipe as mp
import numpy as np
import time

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
hands = mediapipe_hands.Hands(static_image_mode=False, max_num_hands= 1, min_detection_confidence=0.3)

#Label mapping for prediction
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
               10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
               18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'Bk'}

# Word formation variables
word = ""
sentence = ""
last_detected_time = None
current_character = None
selection_effect_time = 0

#Check if the camera is open/close
while cap.isOpened():
    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break # Exit when frame not read correctly

    # Prepare auxiliary lists for normalized landmark data
    data_aux = []
    x_aux = []
    y_aux = []

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

        # Loop through landmarks again to extract and normalize coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect raw landmark coordinates
            for lm in hand_landmarks.landmark:
                x_aux.append(lm.x)
                y_aux.append(lm.y)

            #Normalize coordinates
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_aux))
                data_aux.append(lm.y - min(y_aux))

        #Skips the frame when data size doesn't match
        if (len(data_aux) != 42):
            continue

        # Convert normalized landmarks into bounding box coordinates
        x1 = int(min(x_aux) * W) - 10
        x2 = int(max(x_aux) * W) + 10
        y1 = int(min(y_aux) * H) - 10
        y2 = int(max(y_aux) * H) + 10

        # Predict the gesture using a trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])] #Get predicted class index and map to character

        # Start timer for selection delay
        if last_detected_time is None:
            last_detected_time = time.time()
        current_character = predicted_character

        # Waits 2 seconds to confirm selection
        if time.time() - last_detected_time >= 2:
            if current_character == "Bk": #Handle Backspace
                word = word[:-1]
                sentence = sentence[:-1]
            elif current_character == " ": #Handle Space
                word = ""
                sentence += " "
            else:
                word += current_character #Append character to word
                sentence += current_character #Append character to sentence

            # Save sentence to a file
            with open("sentence.txt", "w") as f:
                f.write(sentence)
            last_detected_time = None
            selection_effect_time = time.time()

        # Draw the bounding box and prediction label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    else:
        last_detected_time = None #Resets the timer when no hand is detected

    # Display the current word and sentence on screen
    cv2.putText(frame, f'Word: {word}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    cv2.putText(frame, f'Sentence: {sentence}', (20, 300), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    # Selection effect
    if time.time() - selection_effect_time < 0.5:
        cv2.rectangle(frame, (50, 150), (500, 300), (0, 255, 0), -1)
        cv2.putText(frame, f'Selected: {current_character}', (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5, cv2.LINE_AA)

    # Show the final frame with annotations
    cv2.imshow('Hand Gesture to Word', frame)

    # Keyboard controls
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # ESC key to exit
        break

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()