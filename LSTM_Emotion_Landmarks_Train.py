# This model achieved 90.48% in Val Acc

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from LSTM_Load_Dataset import EmotionLandmarkDataset

DATA_DIR = "./data/npz"
BATCH_SIZE = 16
VAL_SPLIT = 0.2
EPOCHS = 500
MAX_BATCHES = 30
LEARNING_RATE = 0.0001
MODEL_PATH = "models/landmarks_lstm.pt"

class EmotionLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(EmotionLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

dataset = EmotionLandmarkDataset(DATA_DIR)
val_size = int(len(dataset) * VAL_SPLIT)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

INPUT_SIZE = 1434
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = len(dataset.label_map)

model = EmotionLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
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
            val_outputs = model(val_data)
            val_pred = torch.argmax(val_outputs, dim=1)
            val_correct += (val_pred == val_labels).sum().item()
            val_total += val_labels.size(0)

    val_acc = val_correct / val_total * 100

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_PATH)

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

print(f"Best model saved to {MODEL_PATH} with Val Acc: {best_val_acc:.2f}%")
