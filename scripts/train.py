"""CLI: train one model from a config."""

import argparse

from neuroscan_ood.train.loop import train
from neuroscan_ood.utils.config import load_config

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["_resume"] = args.resume
    print("done:", train(cfg))
