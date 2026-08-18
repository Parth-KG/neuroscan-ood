"""CLI: R3 - diagnose the shift (Grad-CAM, embedding, domain AUC)."""

import argparse
import sys

from neuroscan_ood.experiments.diagnose import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r1.yaml")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.exit(main(args.config, args.seed))
