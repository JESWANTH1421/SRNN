import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model loaded on", device)

NUM_CLASSES = 10

# ============================
# 1️⃣ DATASET
# ============================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=False
)

print("Total test samples:", len(test_dataset))

# ============================
# 2️⃣ MODEL (Same as train.py)
# ============================

class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.flatten = nn.Flatten()

        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, NUM_CLASSES)
        )

        self.reliability_head = nn.Sequential(
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.features(x)
        flat = self.flatten(features)

        logits = self.classifier(flat)
        reliability = self.reliability_head(flat)

        return logits, reliability.squeeze()

model = SRNN().to(device)
model.load_state_dict(torch.load("srnn_cifar_model.pth", map_location=device))
model.eval()

print("Model successfully loaded!")

# ============================
# 3️⃣ COLLECT PREDICTIONS
# ============================

all_labels = []
all_preds = []
all_reliability = []
all_softmax_conf = []

softmax = nn.Softmax(dim=1)

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs, reliability = model(images)

        probs = softmax(outputs)
        conf, preds = torch.max(probs, dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_reliability.extend(reliability.cpu().numpy())
        all_softmax_conf.extend(conf.cpu().numpy())

all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_reliability = np.array(all_reliability)
all_softmax_conf = np.array(all_softmax_conf)

print("Evaluation data collected.\n")

# ============================
# 4️⃣ SELECTIVE PREDICTION
# ============================

def selective_prediction(confidence_scores, name):

    print(f"\n===== {name} Selective Prediction =====")
    print("Threshold | Coverage | Accuracy | Risk")

    for threshold in np.arange(0.0, 1.0, 0.1):

        selected = confidence_scores >= threshold

        if np.sum(selected) == 0:
            continue

        coverage = np.mean(selected) * 100

        selected_preds = all_preds[selected]
        selected_labels = all_labels[selected]

        accuracy = np.mean(selected_preds == selected_labels) * 100
        risk = 100 - accuracy

        print(f"{threshold:.1f}      | "
              f"{coverage:.2f}%   | "
              f"{accuracy:.2f}%   | "
              f"{risk:.2f}%")

# SRNN Reliability
selective_prediction(all_reliability, "SRNN Reliability")

# Softmax Confidence
selective_prediction(all_softmax_conf, "Softmax Confidence")

# ============================
# 5️⃣ ERROR ANALYSIS
# ============================

wrong = all_preds != all_labels
correct = all_preds == all_labels

avg_rel_correct = np.mean(all_reliability[correct])
avg_rel_wrong = np.mean(all_reliability[wrong])

high_conf_errors = np.sum((all_reliability > 0.8) & wrong)

print("\n===== ERROR ANALYSIS =====")
print("Total Wrong Predictions:", np.sum(wrong))
print("Avg Reliability (Correct):", round(avg_rel_correct, 4))
print("Avg Reliability (Wrong):", round(avg_rel_wrong, 4))
print("High-Reliability Errors (>0.8):", high_conf_errors)