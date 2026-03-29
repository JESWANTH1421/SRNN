import torch

def classification_accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).sum().item()
    return correct

def reliability_accuracy(reliability, logits, labels):
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels)

    reliability_pred = (reliability.squeeze() > 0.5)

    correct_rel = (reliability_pred == correct).sum().item()
    return correct_rel
