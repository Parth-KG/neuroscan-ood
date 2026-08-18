"""CLI: evaluate a saved checkpoint on its config's test split."""

import argparse

import torch
from torch.utils.data import DataLoader

from neuroscan_ood.data.dataset import MriDataset
from neuroscan_ood.data.manifest import load_manifest
from neuroscan_ood.data.splits import make_split
from neuroscan_ood.eval.evaluate import evaluate
from neuroscan_ood.models.build import build_model
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.paths import images_root, prepared_root, run_dir


def _filter_source(df, source):
    if isinstance(source, list):
        return df[df["source"].isin(source)]
    return df[df["source"] == source]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_manifest(prepared_root())
    df = _filter_source(df, cfg["source"])
    df = df[df["label_name"].isin(set(cfg["classes"]))]
    _, test_df = make_split(df, cfg["split"], cfg["seed"])
    ld = DataLoader(
        MriDataset(test_df, images_root(), cfg["train"]["image_size"], train=False),
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
    )
    model = build_model(len(cfg["classes"]), pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device)["model"])
    print(evaluate(model, ld, run_dir(cfg["run_id"]), device, cfg["classes"])["accuracy"])
