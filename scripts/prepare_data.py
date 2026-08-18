"""CLI: prepare raw data into manifest + PNGs."""

import argparse
import sys
from pathlib import Path

from neuroscan_ood.data.prepare import main

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    sys.exit(main(Path(args.raw_root), Path(args.out_root)))
