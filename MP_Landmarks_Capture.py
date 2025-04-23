import cv2
import mediapipe as mp
import numpy as np
import os

label = "neutral"
sequence_length = 50   # number of frames per sample
num_sequences = 30     # total samples per emotion
save_dir = "data/npz/"+label      # folder to store samples
os.makedirs(save_dir, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0)
print(f"Capturing {num_sequences} sequences of '{label}'...")

sequence = []
saved = 0

while saved < num_sequences:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        landmarks = []
        for lm in results.multi_face_landmarks[0].landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
        sequence.append(landmarks)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION
        )

        if len(sequence) == sequence_length:
            path = os.path.join(save_dir, f"{label}_{saved}.npz")
            np.savez_compressed(path, landmarks=np.array(sequence), label=label)
            print(f"Saved: {path}")
            sequence = []
            saved += 1

    cv2.imshow("Recording", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
