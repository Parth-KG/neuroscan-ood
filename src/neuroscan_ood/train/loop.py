"""Training loop with per-epoch checkpointing and resume."""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuroscan_ood.data.dataset import MriDataset
from neuroscan_ood.data.manifest import load_manifest
from neuroscan_ood.data.splits import make_split
from neuroscan_ood.eval.evaluate import evaluate
from neuroscan_ood.models.build import build_model
from neuroscan_ood.utils.config import save_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, prepared_root, run_dir
from neuroscan_ood.utils.seed import loader_generator, seed_worker, set_seed

log = get_logger("train")


def _device():
    dev = os.environ.get("NEUROSCAN_DEVICE")
    if dev:
        return torch.device(dev)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _filter_source(df, source):
    if isinstance(source, list):
        return df[df["source"].isin(source)]
    return df[df["source"] == source]


def train(cfg: dict) -> str:
    set_seed(cfg["seed"])
    device = _device()
    classes = cfg["classes"]
    out = run_dir(cfg["run_id"])
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    df = load_manifest(prepared_root())
    df = _filter_source(df, cfg["source"])
    df = df[df["label_name"].isin(set(classes))]  # supports the 2-class clean subset later
    train_df, test_df = make_split(df, cfg["split"], cfg["seed"])

    tcfg = cfg["train"]
    img_size, bs = tcfg["image_size"], tcfg["batch_size"]
    workers = tcfg.get("num_workers", 2)
    g = loader_generator(cfg["seed"])
    train_ld = DataLoader(
        MriDataset(train_df, images_root(), img_size, train=True),
        batch_size=bs,
        shuffle=True,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=g,
    )
    test_ld = DataLoader(
        MriDataset(test_df, images_root(), img_size, train=False),
        batch_size=bs,
        shuffle=False,
        num_workers=workers,
    )

    model = build_model(num_classes=len(classes), pretrained=cfg["model"]["pretrained"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    lossf = nn.CrossEntropyLoss()

    start_epoch = 0
    latest = out / "checkpoints" / "latest.pt"
    if cfg.get("_resume") and latest.exists():
        ck = torch.load(latest, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_epoch = ck["epoch"] + 1
        log.info("resumed from epoch %d", start_epoch)

    n_train = len(train_df)
    for epoch in range(start_epoch, tcfg["epochs"]):
        model.train()
        running = 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = lossf(model(x), y)
            loss.backward()
            opt.step()
            running += loss.item() * x.size(0)
        log.info("epoch %d/%d loss=%.4f", epoch + 1, tcfg["epochs"], running / max(n_train, 1))
        ck = {
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "epoch": epoch,
            "seed": cfg["seed"],
        }
        torch.save(ck, out / "checkpoints" / f"epoch_{epoch:03d}.pt")
        torch.save(ck, latest)

    metrics = evaluate(model, test_ld, out, device, classes)
    log.info("test accuracy=%.4f ece=%.4f", metrics["accuracy"], metrics["ece"])
    return cfg["run_id"]
