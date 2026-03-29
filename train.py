import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Hyperparameters
# =========================
batch_size = 128
learning_rate = 0.001
epochs = 30
wrong_weight = 6.0           # stronger wrong penalty
reliability_scale = 0.3      # reduced influence

# =========================
# CIFAR-10 Dataset
# =========================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

train_dataset = datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# =========================
# SRNN Model
# =========================
class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()
        self.shared_fc = nn.Linear(128 * 4 * 4, 256)

        self.classifier = nn.Linear(256, 10)
        self.reliability_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.shared_fc(x))

        logits = self.classifier(x)

        # Temperature scaling (prevents saturation)
        reliability = torch.sigmoid(self.reliability_head(x) / 2.0)

        return logits, reliability


model = SRNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# =========================
# Training
# =========================
for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        logits, reliability = model(images)
        cls_loss = criterion(logits, labels)

        _, preds = torch.max(logits, 1)
        correct_mask = (preds == labels).float()

        # ----------------------------
        # Label-smoothed targets
        # Correct -> 0.95
        # Wrong   -> 0.05
        # ----------------------------
        target = correct_mask * 0.9 + 0.05

        bce = F.binary_cross_entropy(
            reliability.squeeze(),
            target,
            reduction='none'
        )

        weights = torch.where(correct_mask == 0, wrong_weight, 1.0)
        weighted_bce = (bce * weights).mean()

        loss = cls_loss + reliability_scale * weighted_bce

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Loss: {total_loss/len(train_loader):.4f} "
          f"Train Acc: {100*correct/total:.2f}%")

# =========================
# Evaluation
# =========================
model.eval()

correct = 0
total = 0
correct_reliability = []
wrong_reliability = []
high_conf_errors = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        logits, reliability = model(images)
        _, preds = torch.max(logits, 1)

        total += labels.size(0)
        correct += (preds == labels).sum().item()

        for i in range(len(labels)):
            rel_val = reliability[i].item()

            if preds[i] == labels[i]:
                correct_reliability.append(rel_val)
            else:
                wrong_reliability.append(rel_val)
                if rel_val > 0.8:
                    high_conf_errors += 1

print("\n===== TEST RESULTS =====")
print(f"Test Accuracy: {100*correct/total:.2f}%")
print(f"Total Wrong Predictions: {len(wrong_reliability)}")
print(f"Avg Reliability (Correct): {sum(correct_reliability)/len(correct_reliability):.4f}")
print(f"Avg Reliability (Wrong): {sum(wrong_reliability)/len(wrong_reliability):.4f}")
print(f"High-Reliability Errors (>0.8): {high_conf_errors}")