"""CLI: source-independence audit between Figshare and SARTAJ."""

import argparse
import sys

import yaml

from neuroscan_ood.experiments.audit import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/audit.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}
    sys.exit(main(threshold=int(cfg.get("hamming_threshold", 5))))
