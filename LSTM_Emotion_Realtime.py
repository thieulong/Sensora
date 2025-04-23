import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
from LSTM_Load_Dataset import EmotionLandmarkDataset

SEQ_LENGTH = 30
INPUT_SIZE = 1434
HIDDEN_SIZE = 128
NUM_LAYERS = 2
MODEL_PATH = "models/landmarks_lstm.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data/npz"

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

model = EmotionLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

sequence_buffer = []

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        if len(landmarks) == INPUT_SIZE:
            sequence_buffer.append(landmarks)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0],
            mp_face_mesh.FACEMESH_TESSELATION,
            drawing_spec, drawing_spec
        )

        if len(sequence_buffer) == SEQ_LENGTH:
            sequence_np = np.array(sequence_buffer, dtype=np.float32).reshape(1, SEQ_LENGTH, INPUT_SIZE)
            x = torch.tensor(sequence_np).to(DEVICE)
            with torch.no_grad():
                output = model(x)
                probs = torch.softmax(output, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                emotion = id2label[pred_idx]

            y = 30
            cv2.putText(frame, f"Predicted: {emotion.upper()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            for emo, score in zip(id2label.values(), probs):
                y += 25
                text = f"{emo}: {score:.2f}"
                cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

            sequence_buffer.pop(0)

    cv2.imshow("Real-Time Emotion Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
