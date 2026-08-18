"""CLI: Result R1 - random vs patient-grouped split over seeds."""

import argparse
import sys

from neuroscan_ood.experiments.r1 import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()
    sys.exit(main(args.config, args.seeds))
