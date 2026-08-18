"""R2 (reframed): accuracy of the honest model under controlled acquisition shift.

Loads the patient-grouped model (training it once if no checkpoint is present), then evaluates it
on its held-out test patients under each corruption family across severities 1..5, with severity 0
as the clean baseline. Inference only, so this is fast. Writes a results table and a curve figure.
"""

import copy
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from neuroscan_ood.data.dataset import MriDataset
from neuroscan_ood.data.manifest import load_manifest
from neuroscan_ood.data.splits import make_split
from neuroscan_ood.eval.evaluate import _collect
from neuroscan_ood.eval.metrics import compute_metrics
from neuroscan_ood.experiments.corruptions import CORRUPTIONS, SEVERITIES
from neuroscan_ood.models.build import build_model
from neuroscan_ood.train.loop import _device, _filter_source, train
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, prepared_root, run_dir, runs_root

log = get_logger("r2")


def _ensure_model(base_cfg, seed):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["split"] = "grouped"
    cfg["run_id"] = f"r1_grouped_seed{seed}"
    ckpt = run_dir(cfg["run_id"]) / "checkpoints" / "latest.pt"
    if not ckpt.exists():
        log.info("no checkpoint for %s; training it now", cfg["run_id"])
        train(cfg)
    return cfg, ckpt


def _test_split(cfg):
    df = load_manifest(prepared_root())
    df = _filter_source(df, cfg["source"])
    df = df[df["label_name"].isin(set(cfg["classes"]))]
    _, test_df = make_split(df, "grouped", cfg["seed"])
    return test_df


def _eval(model, test_df, classes, img_size, device, corruption):
    ds = MriDataset(test_df, images_root(), img_size, train=False, corruption=corruption)
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    logits, labels = _collect(model, ld, device)
    return compute_metrics(logits, labels, num_classes=len(classes))


def main(base_config, seed=0):
    base_cfg = load_config(base_config)
    device = _device()
    cfg, ckpt = _ensure_model(base_cfg, seed)
    classes = cfg["classes"]
    img_size = cfg["train"]["image_size"]

    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    test_df = _test_split(cfg)

    out = runs_root() / "r2_shift"
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    clean = _eval(model, test_df, classes, img_size, device, None)
    clean_acc = clean["accuracy"]
    rows.append(("clean", 0, clean_acc, clean["ece"]))
    log.info("clean accuracy=%.4f", clean_acc)

    for name, fn in CORRUPTIONS.items():
        for s in SEVERITIES:
            m = _eval(model, test_df, classes, img_size, device, lambda a, fn=fn, s=s: fn(a, s))
            rows.append((name, s, m["accuracy"], m["ece"]))
            log.info("%s sev%d acc=%.4f ece=%.4f", name, s, m["accuracy"], m["ece"])

    with open(out / "shift_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["corruption", "severity", "accuracy", "ece"])
        w.writerows([(n, s, f"{a:.6f}", f"{e:.6f}") for (n, s, a, e) in rows])

    fig, ax = plt.subplots(figsize=(6, 4))
    per_sev = {s: [] for s in SEVERITIES}
    for name in CORRUPTIONS:
        ys = [clean_acc] + [a for (n, s, a, e) in rows if n == name]
        ax.plot([0] + SEVERITIES, ys, marker="o", label=name)
        for n, s, a, _ in rows:
            if n == name:
                per_sev[s].append(a)
    mean_ys = [clean_acc] + [sum(per_sev[s]) / len(per_sev[s]) for s in SEVERITIES]
    ax.plot([0] + SEVERITIES, mean_ys, marker="s", color="black", lw=2.5, label="mean")
    ax.set_xlabel("corruption severity (0 = clean)")
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title("R2: accuracy under controlled acquisition shift")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "shift_curves.png", dpi=120)
    plt.close(fig)

    mean_at_max = mean_ys[-1]
    max_sev = {n: a for (n, s, a, e) in rows if s == 5}
    most = min(max_sev, key=max_sev.get)
    least = max(max_sev, key=max_sev.get)
    print("\nR2 RESULT (controlled acquisition shift)")
    print(f" clean (no shift): {clean_acc * 100:.1f}%")
    print(
        f" mean at max severity: {mean_at_max * 100:.1f}% (drop of {(clean_acc - mean_at_max) * 100:.1f} pts)"
    )
    print(f" most sensitive to: {most} ({max_sev[most] * 100:.1f}% at severity 5)")
    print(f" least sensitive to: {least} ({max_sev[least] * 100:.1f}% at severity 5)")
    return 0
