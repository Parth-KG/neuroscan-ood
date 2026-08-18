"""CLI: R2 - controlled acquisition-shift robustness of the grouped model."""

import argparse
import sys

from neuroscan_ood.experiments.r2_shift import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(main(args.config, args.seed))
