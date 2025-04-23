import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from transformers import AutoTokenizer

MODEL_NAME = "bert-base-uncased"
MODEL_PATH = "models/text_cnn.pt"
EMBEDDING_DIM = 128
EMOTIONS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(texts):
    return tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

class EmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(EmotionClassifier, self).__init__()
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
        x = self.fc(x)
        return x
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionClassifier(vocab_size=tokenizer.vocab_size, embedding_dim=EMBEDDING_DIM, num_classes=len(EMOTIONS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict(sentence):
    encoded = tokenize([sentence])
    encoded = {key: val.to(device) for key, val in encoded.items()}

    with torch.no_grad():
        output = model(encoded["input_ids"])
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        prediction = torch.argmax(output, dim=1).item()

    return EMOTIONS[prediction], dict(zip(EMOTIONS, map(float, probs)))

while True:
    user_input = input("Enter a sentence (or 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    emotion, confidence = predict(user_input)
    print(f"Predicted: {emotion}")
    print(f"Confidence:")
    for emo, conf in confidence.items():
        print(f"  {emo}: {conf:.4f}")
