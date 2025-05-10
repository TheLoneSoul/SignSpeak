from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import base64
import time
from googletrans import Translator
from typing import Dict

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.3)

labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
    18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: ' ', 27: 'Bk'
}

class TranslationRequest(BaseModel):
    sentence: str
    lang: str

# Session storage
session_data: Dict[str, Dict[str, any]] = {}

@app.get("/")
async def index():
    return HTMLResponse(open("static/index.html").read())

@app.get("/health")
async def health():
    return JSONResponse(status_code=200, content={"message": "OK"})

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("static/favicon.ico", media_type="image/x-icon", headers={"Cache-Control": "public, max-age=31353600, immutable"})

@app.post("/translate")
async def translate(request: TranslationRequest):
    translator = Translator()
    translated = translator.translate(request.sentence, dest=request.lang).text
    return {"translated": translated}

@app.post("/process_frame")
async def process_frame(file: UploadFile = File(...), session_id: str = Form(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image"}

    if session_id not in session_data:
        session_data[session_id] = {
            "sentence": "",
            "last_detected_time": None,
            "current_character": None,
            "selection_effect_time": 0,
            "last_active": time.time()
        }

    user = session_data[session_id]

    frame = cv2.resize(frame, (640, 480))
    data_aux = []
    x_aux, y_aux = [], []
    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_aux.append(lm.x)
                y_aux.append(lm.y)
            for lm in hand_landmarks.landmark:
                data_aux.append(lm.x - min(x_aux))
                data_aux.append(lm.y - min(y_aux))

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        if len(data_aux) != 42:
            return {"image": "", "sentence": user["sentence"]}

        x1 = int(min(x_aux) * W) - 10
        y1 = int(min(y_aux) * H) - 10
        x2 = int(max(x_aux) * W) + 10
        y2 = int(max(y_aux) * H) + 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_index = int(prediction[0])
        predicted_character = labels_dict[predicted_index]

        if user["last_detected_time"] is None:
            user["last_detected_time"] = time.time()
        user["current_character"] = predicted_character

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

        if time.time() - user["last_detected_time"] >= 2:
            if predicted_character == "Bk":
                user["sentence"] = user["sentence"][:-1]
            elif predicted_character == " ":
                user["sentence"] += " "
            else:
                user["sentence"] += predicted_character
            user["last_detected_time"] = None
            user["selection_effect_time"] = time.time()
    else:
        user["last_detected_time"] = None

    if time.time() - user["selection_effect_time"] < 0.5:
        cv2.putText(frame, f'Selected: {user["current_character"]}', (0, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 5, cv2.LINE_AA)

    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
    img_str = base64.b64encode(buffer).decode('utf-8')

    user["last_active"] = time.time()

    return {
        "image": img_str,
        "sentence": user["sentence"]
    }