import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

DATA_DIR = "./data/npz"
BATCH_SIZE = 16
VAL_SPLIT = 0.2

class EmotionLandmarkDataset(Dataset):
    def __init__(self, root_dir, label_map=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        self.label_map = label_map or self._create_label_map()

        for label_name in os.listdir(root_dir):
            class_dir = os.path.join(root_dir, label_name)
            if not os.path.isdir(class_dir):
                continue
            for file in os.listdir(class_dir):
                if file.endswith(".npz"):
                    self.samples.append({
                        "path": os.path.join(class_dir, file),
                        "label": self.label_map[label_name]
                    })

    def _create_label_map(self):
        classes = sorted(os.listdir(self.root_dir))
        return {name: idx for idx, name in enumerate(classes)}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = np.load(sample["path"])
        landmarks = data["landmarks"]  
        label = sample["label"]

        if self.transform:
            landmarks = self.transform(landmarks)

        return torch.tensor(landmarks, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

if __name__ == "__main__":
    # Load dataset
    dataset = EmotionLandmarkDataset(DATA_DIR)

    # Print label map
    print("Label Map:", dataset.label_map)

    # Split into train/val
    val_size = int(len(dataset) * VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    # Dataloaders
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    # Quick test: show batch shapes
    for batch_data, batch_labels in train_loader:
        print("Batch data shape:", batch_data.shape)    # (B, seq_len, features)
        print("Batch labels shape:", batch_labels.shape)  # (B,)
        break
