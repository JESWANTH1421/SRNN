import torch
import torch.nn as nn
import torch.nn.functional as F

class SRNN(nn.Module):
    def __init__(self):
        super(SRNN, self).__init__()

        # =========================
        # Shared Feature Extractor
        # =========================
        self.feature = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU()
        )

        # =========================
        # Classification Head
        # =========================
        self.classifier = nn.Linear(64, 10)

        # =========================
        # Reliability Head
        # Input = features(64) + logits(10) + entropy(1) + max_prob(1)
        # Total = 76
        # =========================
        self.reliability = nn.Sequential(
            nn.Linear(76, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, 784)

        # Feature extraction
        h = self.feature(x)

        # Classification
        logits = self.classifier(h)

        # Softmax probabilities
        probs = F.softmax(logits, dim=1)

        # Max probability
        max_prob, _ = torch.max(probs, dim=1, keepdim=True)

        # Entropy (uncertainty measure)
        entropy = -torch.sum(
            probs * torch.log(probs + 1e-8),
            dim=1,
            keepdim=True
        )

        # Reliability input
        rel_input = torch.cat([h, logits, entropy, max_prob], dim=1)

        # Reliability score
        reliability_score = self.reliability(rel_input)

        return logits, reliability_score