"""R4: label-free mitigations vs the unmitigated baseline, under controlled shift.

Four conditions, each needing no labels and no retraining:
  - baseline: the model as-is on shifted scans.
  - arm A (matched): re-apply the training-time per-image min-max intensity convention. This is the
    principled intensity normalisation, matched to what the model learned on.
  - arm A (naive): histogram equalisation. A reasonable-sounding but mismatched normalisation, kept
    to show that a fix which does not match the model's expected input can hurt more than the shift.
  - arm B (AdaBN): recompute the model's BatchNorm running statistics on the shifted scans.

The table reports, per arm: external accuracy and ECE (mean over the corruption families at a fixed
severity) and in-distribution accuracy (on clean scans), so a fix that helps under shift but hurts
clean data is visible.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torch.utils.data import DataLoader

from neuroscan_ood.data.dataset import MriDataset
from neuroscan_ood.data.normalise import to_uint8
from neuroscan_ood.eval.evaluate import _collect
from neuroscan_ood.eval.metrics import compute_metrics
from neuroscan_ood.experiments.corruptions import CORRUPTIONS
from neuroscan_ood.experiments.r2_shift import _ensure_model, _test_split
from neuroscan_ood.models.build import build_model
from neuroscan_ood.train.loop import _device
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, runs_root

log = get_logger("r4")
SHIFT_SEVERITY = 3

# arm keys -> pretty labels used in the printed table and the chart
ARMS = {
    "baseline": "baseline",
    "A_matched": "armA matched",
    "A_equalize": "armA naive",
    "B_adabn": "armB AdaBN",
}


def normalize_intensity_matched(arr: np.ndarray) -> np.ndarray:
    """Arm A (matched): re-apply the model's training-time per-image min-max convention."""
    return to_uint8(arr)


def normalize_intensity(arr: np.ndarray) -> np.ndarray:
    """Arm A (naive): histogram equalisation. Kept as a cautionary comparison."""
    return np.array(ImageOps.equalize(Image.fromarray(arr)))


def adapt_bn(
    model: nn.Module, loader: DataLoader, device: torch.device, passes: int = 1
) -> nn.Module:
    """Arm B (AdaBN): recompute BatchNorm running stats on the shifted data. No labels, no grads."""
    bns = [m for m in model.modules() if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d))]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average over the passes
    model.train()
    with torch.no_grad():
        for _ in range(passes):
            for x, _ in loader:
                model(x.to(device))
    model.eval()
    return model


def _loader(df, corruption, img_size):
    ds = MriDataset(df, images_root(), img_size, train=False, corruption=corruption)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)


def _fresh(ckpt, n_classes, device):
    m = build_model(n_classes, pretrained=False).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    m.eval()
    return m


def _metrics(model, loader, device, n_classes):
    logits, labels = _collect(model, loader, device)
    return compute_metrics(logits, labels, num_classes=n_classes)


def _eval_arm(arm, ckpt, n_cls, device, df, img_size, base_corr):
    """Evaluate one arm on one condition. base_corr is the corruption fn (or None for clean)."""
    if arm == "baseline":
        return _metrics(
            _fresh(ckpt, n_cls, device), _loader(df, base_corr, img_size), device, n_cls
        )
    if arm == "A_matched":
        comp = (
            (lambda a: normalize_intensity_matched(base_corr(a)))
            if base_corr
            else normalize_intensity_matched
        )
        return _metrics(_fresh(ckpt, n_cls, device), _loader(df, comp, img_size), device, n_cls)
    if arm == "A_equalize":
        comp = (lambda a: normalize_intensity(base_corr(a))) if base_corr else normalize_intensity
        return _metrics(_fresh(ckpt, n_cls, device), _loader(df, comp, img_size), device, n_cls)
    if arm == "B_adabn":
        m = adapt_bn(_fresh(ckpt, n_cls, device), _loader(df, base_corr, img_size), device)
        return _metrics(m, _loader(df, base_corr, img_size), device, n_cls)
    raise ValueError(arm)


def main(base_config, seed=0, severity=SHIFT_SEVERITY):
    base_cfg = load_config(base_config)
    device = _device()
    cfg, ckpt = _ensure_model(base_cfg, seed)
    classes, img_size = cfg["classes"], cfg["train"]["image_size"]
    n_cls = len(classes)
    test_df = _test_split(cfg)
    out = runs_root() / "r4"
    out.mkdir(parents=True, exist_ok=True)

    names = list(CORRUPTIONS.keys())
    per = {arm: {} for arm in ARMS}
    id_acc = {}
    for arm in ARMS:
        id_acc[arm] = _eval_arm(arm, ckpt, n_cls, device, test_df, img_size, None)["accuracy"]

    for name in names:
        fn = CORRUPTIONS[name]

        def corr(a, fn=fn):
            return fn(a, severity)

        for arm in ARMS:
            per[arm][name] = _eval_arm(arm, ckpt, n_cls, device, test_df, img_size, corr)
        log.info(
            "%s " + " ".join(a + "=%.3f" for a in ARMS),
            name,
            *[per[a][name]["accuracy"] for a in ARMS],
        )

    def mean_acc(arm):
        return float(np.mean([per[arm][n]["accuracy"] for n in names]))

    def mean_ece(arm):
        return float(np.mean([per[arm][n]["ece"] for n in names]))

    with open(out / "mitigation.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["arm", "external_accuracy", "external_ece", "in_distribution_accuracy"])
        for arm in ARMS:
            w.writerow([arm, f"{mean_acc(arm):.4f}", f"{mean_ece(arm):.4f}", f"{id_acc[arm]:.4f}"])

    with open(out / "mitigation_by_corruption.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption"] + list(ARMS.keys()))
        for n in names:
            w.writerow([n] + [f"{per[a][n]['accuracy']:.4f}" for a in ARMS])

    x = np.arange(len(names))
    colors = {
        "baseline": "#61728c",
        "A_matched": "#2fe3c4",
        "A_equalize": "#ff4d9d",
        "B_adabn": "#8b7bff",
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    k = len(ARMS)
    for i, arm in enumerate(ARMS):
        ax.bar(
            x + (i - (k - 1) / 2) * 0.2,
            [per[arm][n]["accuracy"] for n in names],
            width=0.2,
            label=ARMS[arm],
            color=colors[arm],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("accuracy under shift")
    ax.set_title(f"R4: mitigations vs baseline (severity {severity})")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "mitigation.png", dpi=120)
    plt.close(fig)

    print(f"\nR4 RESULT (mean over shifts at severity {severity})")
    print(f" {'arm':14s} {'external acc':>12s} {'external ECE':>12s} {'clean acc':>10s}")
    for arm in ARMS:
        print(
            f" {ARMS[arm]:14s} {mean_acc(arm) * 100:>11.1f}% {mean_ece(arm):>12.3f} {id_acc[arm] * 100:>9.1f}%"
        )
    fixes = ["A_matched", "A_equalize", "B_adabn"]
    best = max(fixes, key=mean_acc)
    print(
        f" best fix under shift: {ARMS[best]} (+{(mean_acc(best) - mean_acc('baseline')) * 100:.1f} pts over baseline)"
    )
    return 0
