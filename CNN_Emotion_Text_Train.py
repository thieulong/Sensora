import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer
from datasets import load_dataset

MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 16
EMBEDDING_DIM = 128
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = "models/text_cnn.pt"

print("Loading dataset...")
dataset = load_dataset("emotion")

train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]
test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

print("Tokenizing...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

train_encodings = tokenize(train_texts)
test_encodings = tokenize(test_texts)

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}, self.labels[idx]

train_dataset = EmotionDataset(train_encodings, train_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(EmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=tokenizer.pad_token_id)
        self.conv1d = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.embedding(x)        # (batch_size, seq_len, embedding_dim)
        x = x.permute(0, 2, 1)       # (batch_size, embedding_dim, seq_len)
        x = self.conv1d(x)           # (batch_size, out_channels, seq_len)
        x = self.relu(x)
        x = self.global_avg_pool(x)  # (batch_size, out_channels, 1)
        x = x.squeeze(2)             # (batch_size, out_channels)
        x = self.fc(x)               # (batch_size, num_classes)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
num_classes = len(set(train_labels))

model = EmotionClassifier(vocab_size, EMBEDDING_DIM, num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

print("Training started...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0

    for batch, labels in train_loader:
        batch = {key: val.to(device) for key, val in batch.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch["input_ids"])
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

print(f"Saving model to {MODEL_SAVE_PATH}")
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print("Training complete.")
