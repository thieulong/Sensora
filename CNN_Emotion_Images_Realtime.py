import cv2
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms, models
from LSTM_Load_Dataset import EmotionLandmarkDataset  

MODEL_PATH = "models/images_cnn.pt"
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision.datasets import ImageFolder
class_names = ImageFolder("data/img/").classes  
NUM_CLASSES = len(class_names)

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    input_img = transform(display)
    input_img = input_img.unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(input_img)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = np.argmax(probs)
        emotion = class_names[pred_idx]

    cv2.putText(display, f"Emotion: {emotion.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    for i, (label, score) in enumerate(zip(class_names, probs)):
        y = 60 + i * 25
        text = f"{label}: {score:.2f}"
        cv2.putText(display, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Real-Time Emotion Detection", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
