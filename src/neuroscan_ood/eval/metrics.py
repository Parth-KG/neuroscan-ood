"""Metrics including calibration. Pure given (logits, labels)."""

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torchmetrics.classification import MulticlassCalibrationError

N_BINS = 15  # fixed so ECE is comparable across R2/R4/R5


def compute_metrics(logits, labels, num_classes: int) -> dict:
    logits = torch.as_tensor(logits, dtype=torch.float32)
    labels = torch.as_tensor(labels, dtype=torch.long)
    probs = F.softmax(logits, dim=1)
    preds = probs.argmax(dim=1)
    y_true, y_pred = labels.numpy(), preds.numpy()

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    ece = float(
        MulticlassCalibrationError(num_classes=num_classes, n_bins=N_BINS, norm="l1")(probs, labels)
    )
    return {
        "accuracy": acc,
        "per_class": {
            "precision": [float(x) for x in p],
            "recall": [float(x) for x in r],
            "f1": [float(x) for x in f1],
            "support": [int(x) for x in support],
        },
        "confusion_matrix": cm.tolist(),
        "ece": ece,
        "n_bins": N_BINS,
        "num_classes": num_classes,
    }
