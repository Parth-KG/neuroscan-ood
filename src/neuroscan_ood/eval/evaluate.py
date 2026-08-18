"""Run a model over a loader and write metrics.json + figures."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from neuroscan_ood.eval.metrics import N_BINS, compute_metrics


@torch.no_grad()
def _collect(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).cpu())
        labels.append(y)
    return torch.cat(logits), torch.cat(labels)


def _plot_confusion(cm, classes, path):
    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    thresh = cm.max() / 2 if cm.max() else 0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _plot_reliability(probs, labels, path, n_bins=N_BINS):
    conf, pred = probs.max(1)
    correct = (pred == labels).float()
    edges = np.linspace(0, 1, n_bins + 1)
    xs, ys = [], []
    for b in range(n_bins):
        m = (conf >= edges[b]) & (conf < edges[b + 1])
        if m.sum() > 0:
            xs.append(conf[m].mean().item())
            ys.append(correct[m].mean().item())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(xs, ys, "o-")
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def evaluate(model, loader, out_dir, device, classes):
    out_dir = Path(out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    logits, labels = _collect(model, loader, device)
    metrics = compute_metrics(logits, labels, num_classes=len(classes))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    _plot_confusion(
        metrics["confusion_matrix"], classes, out_dir / "figures" / "confusion_matrix.png"
    )
    _plot_reliability(F.softmax(logits, dim=1), labels, out_dir / "figures" / "reliability.png")
    return metrics
