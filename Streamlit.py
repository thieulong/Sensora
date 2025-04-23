import streamlit as st
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import speech_recognition as sr
import threading
import time
from transformers import AutoTokenizer
from LSTM_Load_Dataset import EmotionLandmarkDataset
from collections import defaultdict
import matplotlib.pyplot as plt

SEQ_LENGTH = 30
INPUT_SIZE = 1434
HIDDEN_SIZE = 128
NUM_LAYERS = 2
LANDMARK_MODEL_PATH = "models/landmarks_lstm.pt"
TEXT_MODEL_PATH = "models/text_cnn.pt"
DATA_DIR = "./data/npz"
TEXT_EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

face_emotion_counts = defaultdict(int)
text_emotion_counts = defaultdict(int)
last_pie_update = time.time()

dataset = EmotionLandmarkDataset(DATA_DIR)
label_map = dataset.label_map
id2label = {idx: label for label, idx in label_map.items()}
NUM_CLASSES = len(label_map)

class EmotionLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

landmark_model = EmotionLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
landmark_model.load_state_dict(torch.load(LANDMARK_MODEL_PATH, map_location=DEVICE))
landmark_model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

class TextEmotionCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextEmotionCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.global_avg_pool(x)
        x = x.squeeze(2)
        return self.fc(x)

text_model = TextEmotionCNN(tokenizer.vocab_size, 128, len(TEXT_EMOTIONS)).to(DEVICE)
text_model.load_state_dict(torch.load(TEXT_MODEL_PATH, map_location=DEVICE))
text_model.eval()

def predict_text_emotion(text):
    encoded = tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        output = text_model(encoded["input_ids"])
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
    return TEXT_EMOTIONS[pred_idx], dict(zip(TEXT_EMOTIONS, map(float, probs)))

transcribed_text = ""
text_emotion = ""
text_confidence = {}
last_spoken_time = time.time()
lock = threading.Lock()

st.set_page_config(page_title="Sensora - Real-Time Emotion Detection", layout="wide")
st.title("Sensora: Emotion Recognition")

left_col, mid_col, right_col = st.columns([3, 0.05, 1])

with left_col:
    FRAME_WINDOW = left_col.image([])
    transcript_placeholder = left_col.empty()
    emotion_placeholder = left_col.empty()

with mid_col:
    st.markdown(
        """
        <div style='border-left: 2px solid #d3d3d3; height: 80vh; margin: auto;'></div>
        """,
        unsafe_allow_html=True
    )

with right_col:
    facial_pie_title = right_col.empty()
    facial_pie_placeholder = right_col.empty()
    speech_pie_title = right_col.empty()
    speech_pie_placeholder = right_col.empty()

def passive_transcribe_loop():
    global transcribed_text, text_emotion, text_confidence, last_spoken_time

    recognizer = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        recognizer.adjust_for_ambient_noise(source)

    while True:
        with mic as source:
            try:
                audio = recognizer.listen(source, phrase_time_limit=5)
                text = recognizer.recognize_google(audio)
                with lock:
                    transcribed_text = text
                    text_emotion, text_confidence = predict_text_emotion(text)
                    text_emotion_counts[text_emotion] += 1
                    last_spoken_time = time.time()
            except:
                with lock:
                    transcribed_text = ""
                    text_emotion = ""
                    text_confidence = {}

        with lock:
            if time.time() - last_spoken_time > 1:
                transcribed_text = ""
                text_emotion = ""
                text_confidence = {}

threading.Thread(target=passive_transcribe_loop, daemon=True).start()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

sequence_buffer = []
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    display = frame.copy()

    if results.multi_face_landmarks:
        landmarks = []
        face_landmarks = results.multi_face_landmarks[0]

        mp_drawing.draw_landmarks(display, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)

        for lm in face_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        h, w, _ = frame.shape
        xs = [int(lm.x * w) for lm in face_landmarks.landmark]
        ys = [int(lm.y * h) for lm in face_landmarks.landmark]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        if len(landmarks) == INPUT_SIZE:
            sequence_buffer.append(landmarks)

        if len(sequence_buffer) == SEQ_LENGTH:
            x_seq = np.array(sequence_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, INPUT_SIZE)
            x_tensor = torch.tensor(x_seq).to(DEVICE)

            with torch.no_grad():
                output = landmark_model(x_tensor)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                emotion = id2label[pred_idx]
                face_emotion_counts[emotion] += 1

            cv2.putText(display, f"Facial Emotion: {emotion.upper()}", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            sequence_buffer.pop(0)

    display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
    resized_display = cv2.resize(display, (1024, 650)) 
    FRAME_WINDOW.image(resized_display)

    with lock:
        if transcribed_text:
            transcript_placeholder.markdown(
                f"<div style='font-size: 1.6rem; font-weight: bold;'>You: <span style='font-weight: normal;'>{transcribed_text}</span></div>",
                unsafe_allow_html=True
            )
        if text_emotion:
            scores_inline = " | ".join(f"{e}: {s:.0%}" for e, s in text_confidence.items())
            emo_display = f"""
            <div style='font-size: 1.4rem;'>
                <b>Speech Emotion:</b> {text_emotion.upper()}<br>
                <span style='font-size: 1.4rem; color: gray;'>{scores_inline}</span>
            </div>
            """
            emotion_placeholder.markdown(emo_display, unsafe_allow_html=True)

        else:
            transcript_placeholder.markdown(
                "<div style='font-size: 1.6rem; color: gray;'>Listening...</div>",
                unsafe_allow_html=True
            )
            emotion_placeholder.empty()

    now = time.time()
    if now - last_pie_update >= 1:
        facial_pie_title.markdown("<div style='text-align: center; font-size: 1.3rem; margin-top: 0px;'><b>Facial Emotion Distribution</b></div>", unsafe_allow_html=True)
        if face_emotion_counts:
            fig1, ax1 = plt.subplots()
            labels = list(face_emotion_counts.keys())
            sizes = list(face_emotion_counts.values())
            ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13})
            ax1.axis('equal')
            facial_pie_placeholder.pyplot(fig1)
        else:
            facial_pie_placeholder.info("No data available.")

        speech_pie_title.markdown("<div style='text-align: center; font-size: 1.3rem; margin-top: 10px;'><b>Speech Emotion Distribution</b></div>", unsafe_allow_html=True)
        if text_emotion_counts:
            fig2, ax2 = plt.subplots()
            labels = list(text_emotion_counts.keys())
            sizes = list(text_emotion_counts.values())
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 13})
            ax2.axis('equal')
            speech_pie_placeholder.pyplot(fig2)
        else:
            speech_pie_placeholder.info("No data available.")

        last_pie_update = now

cap.release()
cv2.destroyAllWindows()
