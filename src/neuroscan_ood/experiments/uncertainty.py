"""R5: make the model honest about what it does not know.

Two label-free reliability tools, evaluated on a mixed shifted deployment (all corruptions pooled
at a fixed severity), with a patient-disjoint clean split used only for fitting:
  - Temperature scaling: fit one scalar T on in-distribution (clean) calibration logits, then apply
    it to the shifted logits and measure calibration error (ECE) before vs after.
  - Selective prediction (referral): use the top softmax probability as a confidence score, refer
    the least-confident scans to a human, and measure accuracy on the ones kept across coverage
    levels. Reports accuracy at 80% coverage (refer the least-confident 20%).
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassCalibrationError

from neuroscan_ood.data.dataset import MriDataset
from neuroscan_ood.eval.evaluate import _collect
from neuroscan_ood.eval.metrics import N_BINS
from neuroscan_ood.experiments.corruptions import CORRUPTIONS
from neuroscan_ood.experiments.r2_shift import _ensure_model, _test_split
from neuroscan_ood.models.build import build_model
from neuroscan_ood.train.loop import _device
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, runs_root

log = get_logger("r5")
SHIFT_SEVERITY = 3


def _logits_labels(model, df, corruption, img_size, device):
    ds = MriDataset(df, images_root(), img_size, train=False, corruption=corruption)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return _collect(model, ld, device)


def _ece(probs, labels, n_cls):
    return float(
        MulticlassCalibrationError(num_classes=n_cls, n_bins=N_BINS, norm="l1")(probs, labels)
    )


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """1D grid search for the temperature that minimises NLL. Deterministic and robust."""
    best_T, best_nll = 1.0, float("inf")
    for T in np.linspace(0.5, 10.0, 191):
        nll = F.cross_entropy(logits / float(T), labels).item()
        if nll < best_nll:
            best_nll, best_T = nll, float(T)
    return best_T


def referral_curve(probs: torch.Tensor, labels: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Accuracy vs coverage when the least-confident scans are referred away."""
    conf, pred = probs.max(1)
    correct = (pred == labels).numpy().astype(float)
    order = np.argsort(-conf.numpy())
    cs = correct[order]
    n = len(cs)
    coverage = np.arange(1, n + 1) / n
    accuracy = np.cumsum(cs) / np.arange(1, n + 1)
    return coverage, accuracy


def acc_at_coverage(coverage: np.ndarray, accuracy: np.ndarray, c: float = 0.8) -> float:
    idx = min(int(np.searchsorted(coverage, c)), len(accuracy) - 1)
    return float(accuracy[idx])


def _pooled_shifted(model, df, severity, img_size, device):
    logits, labels = [], []
    for fn in CORRUPTIONS.values():
        lo, la = _logits_labels(model, df, lambda a, fn=fn: fn(a, severity), img_size, device)
        logits.append(lo)
        labels.append(la)
    return torch.cat(logits), torch.cat(labels)


def main(base_config, seed=0, severity=SHIFT_SEVERITY):
    base_cfg = load_config(base_config)
    device = _device()
    cfg, ckpt = _ensure_model(base_cfg, seed)
    classes, img_size = cfg["classes"], cfg["train"]["image_size"]
    n_cls = len(classes)
    model = build_model(n_cls, pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_df = _test_split(cfg)

    # patient-disjoint split of the test set: cal (fit T) vs eval (report)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    cal_idx, eval_idx = next(gss.split(test_df, groups=test_df["pid"].values))
    cal_df, eval_df = test_df.iloc[cal_idx].copy(), test_df.iloc[eval_idx].copy()

    cal_logits, cal_labels = _logits_labels(model, cal_df, None, img_size, device)
    T = fit_temperature(cal_logits, cal_labels)
    log.info("fitted temperature T=%.3f", T)

    ext_logits, ext_labels = _pooled_shifted(model, eval_df, severity, img_size, device)
    probs_before = F.softmax(ext_logits, dim=1)
    probs_after = F.softmax(ext_logits / T, dim=1)
    ece_before = _ece(probs_before, ext_labels, n_cls)
    ece_after = _ece(probs_after, ext_labels, n_cls)

    cov, acc = referral_curve(probs_after, ext_labels)
    acc80 = acc_at_coverage(cov, acc, 0.8)
    acc100 = float(acc[-1])

    # clean referral for contrast
    clean_logits, clean_labels = _logits_labels(model, eval_df, None, img_size, device)
    covc, accc = referral_curve(F.softmax(clean_logits / T, dim=1), clean_labels)

    out = runs_root() / "r5"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "calibration.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["temperature", "external_ece_before", "external_ece_after"])
        w.writerow([f"{T:.4f}", f"{ece_before:.4f}", f"{ece_after:.4f}"])
    with open(out / "referral.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["coverage", "accuracy_shifted"])
        for c, a in zip(cov, acc, strict=False):
            w.writerow([f"{c:.4f}", f"{a:.4f}"])

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(covc, accc, color="#2fe3c4", label="clean")
    ax.plot(cov, acc, color="#ff4d9d", label="shifted")
    ax.axvline(0.8, ls="--", color="grey")
    ax.scatter([0.8], [acc80], color="#ff4d9d", zorder=5)
    ax.annotate(
        f"{acc80 * 100:.0f}% @ 80%", (0.8, acc80), textcoords="offset points", xytext=(6, -14)
    )
    ax.set_xlabel("coverage (fraction of scans kept)")
    ax.set_ylabel("accuracy on kept scans")
    ax.set_title("R5: referral trades coverage for accuracy")
    ax.legend()
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(out / "referral_curve.png", dpi=120)
    plt.close(fig)

    print("\nR5 RESULT (uncertainty)")
    print(f" temperature T = {T:.2f}")
    print(f" external ECE: before {ece_before:.3f} -> after {ece_after:.3f}")
    print(f" shifted accuracy: {acc100 * 100:.1f}% keeping all scans")
    print(f" {acc80 * 100:.1f}% keeping the 80% most-confident (refer 20%)")
    print(f" referral gain at 80% coverage: +{(acc80 - acc100) * 100:.1f} points")
    return 0
