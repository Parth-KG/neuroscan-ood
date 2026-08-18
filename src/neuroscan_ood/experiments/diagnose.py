"""R3: diagnose why the model fails under shift.

Three complementary views on the shift, all using the honest grouped model and its held-out test
patients, with the "shifted" domain produced by the controlled corruptions (Phase 3):
  - Grad-CAM heatmaps: where the model looks, clean vs shifted.
  - A 2D feature embedding: whether clean and shifted scans separate in the model's representation.
  - A domain-classifier AUC: how easily a simple classifier tells clean from shifted using features.
High separability plus drifted attention explains the accuracy drop measured in R2.
"""

import matplotlib
import numpy as np
import torch
import torch.nn as nn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from neuroscan_ood.data.dataset import MriDataset, build_transforms
from neuroscan_ood.experiments.corruptions import CORRUPTIONS
from neuroscan_ood.experiments.r2_shift import _ensure_model, _test_split
from neuroscan_ood.models.build import build_model
from neuroscan_ood.train.loop import _device
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, runs_root

log = get_logger("r3")
SHIFT_SEVERITY = 3


def _loader(df, corruption, img_size):
    ds = MriDataset(df, images_root(), img_size, train=False, corruption=corruption)
    return DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)


def _features(model, loader, device):
    feats = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            f = model.forward_features(x.to(device))
            f = model.forward_head(f, pre_logits=True)
            feats.append(f.cpu())
    return torch.cat(feats).numpy()


def _last_conv(model):
    target = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            target = m
    return target


def gradcam(model, x, device):
    """Grad-CAM map (normalised 0..1) for the top predicted class."""
    target = _last_conv(model)
    store = {}
    fh = target.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    bh = target.register_full_backward_hook(lambda m, gi, go: store.__setitem__("g", go[0]))
    model.zero_grad()
    out = model(x.to(device))
    idx = int(out.argmax(1))
    out[0, idx].backward()
    A, G = store["a"][0].detach(), store["g"][0].detach()
    w = G.mean(dim=(1, 2))
    cam = torch.relu((w[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    fh.remove()
    bh.remove()
    return cam.cpu().numpy(), idx


def _one_input(path, img_size, corruption):
    arr = np.array(Image.open(path).convert("L"))
    if corruption is not None:
        arr = corruption(arr)
    disp = np.array(Image.fromarray(arr).resize((img_size, img_size)))
    x = build_transforms(img_size, train=False)(Image.fromarray(arr).convert("RGB")).unsqueeze(0)
    return x, disp


def _gradcam_figure(model, test_df, img_size, device, out_path, n=3):
    def shift(a):
        return CORRUPTIONS["gaussian_noise"](a, SHIFT_SEVERITY)

    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    for r in range(n):
        path = images_root() / test_df.iloc[r]["filename"]
        for c, (title, corr) in enumerate([("clean", None), ("shifted (noise)", shift)]):
            x, disp = _one_input(path, img_size, corr)
            cam, _ = gradcam(model, x, device)
            cam_r = np.array(
                Image.fromarray((cam * 255).astype("uint8")).resize((img_size, img_size))
            )
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(disp, cmap="gray")
            ax.imshow(cam_r, cmap="jet", alpha=0.45)
            ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle("Grad-CAM: where the model looks (clean vs shifted)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _auc(clean, shifted):
    X = np.concatenate([clean, shifted])
    y = np.concatenate([np.zeros(len(clean)), np.ones(len(shifted))])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    return float(roc_auc_score(yte, clf.predict_proba(sc.transform(Xte))[:, 1]))


def _embedding_figure(clean, shifted, out_path):
    n = min(len(clean), len(shifted), 400)
    X = np.concatenate([clean[:n], shifted[:n]])
    X = PCA(n_components=min(50, X.shape[1]), random_state=0).fit_transform(X)
    perp = min(30, max(2, len(X) - 1))  # perplexity must be < n_samples
    emb = TSNE(n_components=2, random_state=0, init="pca", perplexity=perp).fit_transform(X)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.scatter(emb[:n, 0], emb[:n, 1], s=8, c="#2fe3c4", label="clean", alpha=0.7)
    ax.scatter(emb[n:, 0], emb[n:, 1], s=8, c="#ff4d9d", label="shifted", alpha=0.7)
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Feature embedding: clean vs shifted")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(base_config, seed=0):
    base_cfg = load_config(base_config)
    device = _device()
    cfg, ckpt = _ensure_model(base_cfg, seed)
    classes, img_size = cfg["classes"], cfg["train"]["image_size"]
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_df = _test_split(cfg)

    out = runs_root() / "r3"
    out.mkdir(parents=True, exist_ok=True)

    log.info("rendering Grad-CAM examples")
    _gradcam_figure(model, test_df, img_size, device, out / "gradcam_examples.png")

    log.info("extracting features (clean + shifted)")
    clean = _features(model, _loader(test_df, None, img_size), device)
    saved = {"clean": clean}
    per_auc = {}
    shifted_all = []
    for name, fn in CORRUPTIONS.items():
        sf = _features(
            model, _loader(test_df, lambda a, fn=fn: fn(a, SHIFT_SEVERITY), img_size), device
        )
        saved[f"shift_{name}"] = sf
        shifted_all.append(sf)
        per_auc[name] = _auc(clean, sf)
        log.info("domain AUC clean-vs-%s = %.3f", name, per_auc[name])
    shifted_all = np.concatenate(shifted_all)

    np.savez(out / "features.npz", **saved)
    _embedding_figure(clean, shifted_all, out / "embedding.png")

    headline_auc = _auc(clean, shifted_all)
    import csv

    with open(out / "domain_auc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "auc"])
        for k, v in per_auc.items():
            w.writerow([k, f"{v:.4f}"])
        w.writerow(["pooled", f"{headline_auc:.4f}"])

    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.bar(list(per_auc.keys()), list(per_auc.values()), color="#8b7bff")
    ax.axhline(0.5, ls="--", color="grey")
    ax.set_ylim(0, 1)
    ax.set_ylabel("clean-vs-shifted AUC")
    ax.set_title("How separable is each shift in feature space")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(out / "domain_auc.png", dpi=120)
    plt.close(fig)

    print("\nR3 RESULT (diagnosis)")
    print(f" domain-classifier AUC (clean vs shifted): {headline_auc:.3f}")
    print(" (1.0 = shift is trivially visible in the model's features; 0.5 = invisible)")
    print(
        f" most separable shift: {max(per_auc, key=per_auc.get)} (AUC {max(per_auc.values()):.3f})"
    )
    print(" see runs/r3/ for heatmaps, embedding, and the per-shift table")
    return 0
