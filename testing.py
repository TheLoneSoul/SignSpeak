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

        # Loop through landmarks again to extract and normalize coordinates
        for hand_landmarks in results.multi_hand_landmarks:
            # Collect raw landmark positions
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_aux.append(x)
                y_aux.append(y)
            # Normalize landmarks relative to minimum x and y
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_aux))
                data_aux.append(y - min(y_aux))

        # Convert normalized landmarks into bounding box coordinates
        x1 = int(min(x_aux) * W) - 10
        x2 = int(max(x_aux) * W) - 10
        y1 = int(min(y_aux) * H) - 10
        y2 = int(max(y_aux) * H) - 10

        # Predict the gesture using a trained model
        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        # Draw bounding box and prediction label on the frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

    # Show the final frame with annotations
    cv2.imshow('frame', frame)
    cv2.waitKey(1)

# Release the video capture and destroy all OpenCV windows
cap.release()
cv2.destroyAllWindows()