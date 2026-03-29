import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from metrics import classification_accuracy, reliability_accuracy


def evaluate_model(model):

    transform = transforms.ToTensor()

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=64)

    model.eval()

    total_cls = 0
    total_rel = 0

    correct_scores = []
    wrong_scores = []

    with torch.no_grad():
        for images, labels in test_loader:

            logits, reliability = model(images)

            # Accuracy counts
            total_cls += classification_accuracy(logits, labels)
            total_rel += reliability_accuracy(reliability, logits, labels)

            # Reliability separation analysis
            preds = torch.argmax(logits, dim=1)
            correct_mask = (preds == labels)

            correct_scores.extend(
                reliability[correct_mask].cpu().numpy().flatten()
            )

            wrong_scores.extend(
                reliability[~correct_mask].cpu().numpy().flatten()
            )

    accuracy = 100 * total_cls / len(test_dataset)
    rel_accuracy = 100 * total_rel / len(test_dataset)

    print("\n===== FINAL RESULTS =====")
    print(f"Classification Accuracy: {accuracy:.2f}%")
    print(f"Reliability Accuracy: {rel_accuracy:.2f}%")

    # Important research check
    print("\n===== RELIABILITY ANALYSIS =====")

    if len(correct_scores) > 0:
        print("Avg reliability (correct predictions):",
              round(np.mean(correct_scores), 4))

    if len(wrong_scores) > 0:
        print("Avg reliability (wrong predictions):",
              round(np.mean(wrong_scores), 4))

    print("Number of wrong predictions:",
          len(wrong_scores))