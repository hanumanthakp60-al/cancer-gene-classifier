import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── 1. Check GPU ──────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ── 2. Load data ──────────────────────────────────────────────────────────────
print("Loading data...")
df = pd.read_parquet("tcga_5cancer_matrix.parquet")
X = df.drop(columns=["cancer_type"]).values.astype(np.float32)
y = df["cancer_type"].values

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print(f"Classes: {le.classes_}")
print(f"Shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ── 3. PyTorch Dataset ────────────────────────────────────────────────────────
class GeneDataset(Dataset):
    def __init__(self, X, y):
        # CNN expects (batch, channels, length)
        # We treat genes as a 1D sequence with 1 channel
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = GeneDataset(X_train, y_train)
test_dataset  = GeneDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

# ── 4. CNN Architecture ───────────────────────────────────────────────────────
class GeneCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(GeneCNN, self).__init__()

        # Block 1 — detect local gene patterns
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)   # 5000 → 1250
        )

        # Block 2 — detect higher level patterns
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)   # 1250 → 312
        )

        # Block 3 — detect complex combinations
        self.conv3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)   # 312 → 78
        )

        # Fully connected classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 78, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.classifier(x)
        return x

model = GeneCNN(input_size=5000, num_classes=5).to(device)
print(f"\nModel architecture:")
print(model)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total_params:,}")

# ── 5. Training setup ─────────────────────────────────────────────────────────
# Handle class imbalance with weighted loss
class_counts = np.bincount(y_train)
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# ── 6. Training loop ──────────────────────────────────────────────────────────
print("\nTraining CNN...")
epochs = 20
train_losses = []
test_losses  = []
train_accs   = []
test_accs    = []

for epoch in range(epochs):
    # ── Train ──
    model.train()
    epoch_loss = 0
    all_preds, all_labels = [], []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(y_batch.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_losses.append(epoch_loss / len(train_loader))
    train_accs.append(train_acc)

    # ── Evaluate ──
    model.eval()
    test_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y_batch.cpu().numpy())

    test_acc = accuracy_score(all_labels, all_preds)
    test_losses.append(test_loss / len(test_loader))
    test_accs.append(test_acc)

    scheduler.step()

    print(f"Epoch {epoch+1:2d}/{epochs} | "
          f"Train Loss: {train_losses[-1]:.4f} | "
          f"Train Acc: {train_acc*100:.1f}% | "
          f"Test Acc: {test_acc*100:.1f}%")

# ── 7. Final evaluation ───────────────────────────────────────────────────────
print("\n── Final Results ─────────────────────────────────────────")
print(classification_report(all_labels, all_preds, target_names=le.classes_))

# ── 8. Plot training curves ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train Loss", color="#3B8BD4")
ax1.plot(test_losses,  label="Test Loss",  color="#E8593C")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training vs Test Loss")
ax1.legend()

ax2.plot([a*100 for a in train_accs], label="Train Acc", color="#3B8BD4")
ax2.plot([a*100 for a in test_accs],  label="Test Acc",  color="#E8593C")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title("Training vs Test Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("plot11_cnn_training.png", dpi=150)
plt.close()
print("Saved: plot11_cnn_training.png")

# ── 9. Save model ─────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "gene_cnn_model.pth")
print("Saved: gene_cnn_model.pth")
print("\nPyTorch CNN complete!")
