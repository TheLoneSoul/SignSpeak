# 🧠 SignSpeak – Real-Time Sign Language to Speech

<p align="center">
  <img src="static/signs.PNG" width="450"/>
</p>

## 🚀 Overview
**SignSpeak** is a real-time **American Sign Language (ASL) to Text and Speech system** that helps bridge communication between the deaf/mute community and others. It captures hand gestures using a webcam, converts them into text, and then transforms the text into speech — even in multiple languages.

---

## ✨ Features
- 🎥 **Real-Time Detection** – Captures gestures every 2–3 seconds  
- 🔤 **A–Z ASL Recognition** – Supports all 26 alphabets  
- ➕ **Special Gestures** – Includes:
  - Space (word separation)
  - Backspace (error correction)  
- 📝 **Live Text Formation** – Builds sentences dynamically  
- 🔊 **Text-to-Speech** – Converts text into voice output  
- 🌍 **Multilingual Support** – Translate and speak in different languages  

---

## 🛠️ Tech Stack
- **Python**
- **OpenCV** (Computer Vision)
- **MediaPipe** (Hand Tracking)
- **Scikit-learn** (Machine Learning)

---

## 🤖 Model Details
- Trained using a **Random Forest Classifier**
- Built on a **custom dataset** captured manually  
- Optimized for:
  - High accuracy with small datasets  
  - Fast prediction speed  
  - Robust gesture recognition  

---

## ⚙️ How It Works
1. Capture live video from webcam  
2. Detect hand landmarks using MediaPipe  
3. Extract features from hand positions  
4. Predict gesture using trained model  
5. Convert gesture → text  
6. Edit using space/backspace gestures  
7. Convert final text → speech  

---

## 📂 Project Structure
```
├── static/
│   └── signs.PNG
├── main.py
├── model_train.py
├── create_dataset.py
├── collection_img.py
├── dataset.pickle
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## ▶️ Run the Project
```
pip install -r requirements.txt
python main.py
```

---

## 🌟 Future Improvements
- Full word/sentence recognition (beyond alphabets)  
- Deep learning model (CNN/LSTM)  
- Mobile app version  
- Better performance in complex backgrounds  

---

## 💡 Impact
SignSpeak demonstrates how **AI + Computer Vision** can solve real-world communication problems and create inclusive technology for everyone.

---

## 👨‍💻 Author
**Aabhas Bhandari**
