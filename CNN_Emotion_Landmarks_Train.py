# This model achieved 83.53% in Val Acc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from LSTM_Load_Dataset import EmotionLandmarkDataset

DATA_DIR = "./data/npz"
BATCH_SIZE = 16
VAL_SPLIT = 0.2
EPOCHS = 500
LEARNING_RATE = 0.0001
MODEL_PATH = "models/landmarks_cnn.pt"
MAX_BATCHES = 30

class EmotionCNNClassifier(nn.Module):
    def __init__(self, input_channels, num_classes):
        super(EmotionCNNClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=input_channels, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(x)

dataset = EmotionLandmarkDataset(DATA_DIR)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

SEQ_LEN = 50
FEATURE_DIM = 1434
INPUT_CHANNELS = SEQ_LEN  
NUM_CLASSES = len(dataset.label_map)

model = EmotionCNNClassifier(input_channels=SEQ_LEN, num_classes=NUM_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    batch_count = 0

    for batch_data, batch_labels in train_loader:
        if batch_count >= MAX_BATCHES:
            break

        batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
        batch_data = batch_data.permute(0, 1, 2)  

        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = torch.argmax(outputs, dim=1)
        correct += (predicted == batch_labels).sum().item()
        total += batch_labels.size(0)
        batch_count += 1

    train_acc = correct / total * 100

    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for val_data, val_labels in val_loader:
            val_data, val_labels = val_data.to(device), val_labels.to(device)
            val_data = val_data.permute(0, 1, 2)
            val_outputs = model(val_data)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_correct += (val_pred == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total * 100
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

print(f"Best CNN model saved to {MODEL_PATH}, Val Acc: {best_val_acc:.2f}%")
