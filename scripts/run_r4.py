"""CLI: R4 - compare two mitigations against the baseline under shift."""

import argparse
import sys

from neuroscan_ood.experiments.mitigate import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--severity", type=int, default=3)
    args = ap.parse_args()
    sys.exit(main(args.config, args.seed, args.severity))
