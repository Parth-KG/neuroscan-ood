"""Download the raw datasets needed to reproduce the experiments.

The Figshare (Cheng) brain-tumour dataset is fetched automatically from Figshare's public
API (no account needed). SARTAJ is used only for the source-independence audit and requires
a Kaggle account, so this script prints instructions for it rather than requiring credentials.

Usage:
    python scripts/download_data.py --out-root $NEUROSCAN_ROOT/data/raw
"""

import argparse
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

FIGSHARE_ARTICLE = 1512427  # doi.org/10.6084/m9.figshare.1512427
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"

SARTAJ_MESSAGE = """
SARTAJ (needed only for the source-independence audit) requires a Kaggle account.
Download it manually:
  1. https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri
  2. Unzip so the layout is: <out-root>/sartaj/{Training,Testing}/<class>/*.jpg
Or, with the Kaggle CLI (after placing your kaggle.json):
  kaggle datasets download -d sartajbhuvaji/brain-tumor-classification-mri \\
      -p <out-root>/sartaj --unzip
"""


def _read(url: str) -> bytes:
    with urllib.request.urlopen(url) as response:
        return response.read()


def download_figshare(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_read(FIGSHARE_API))
    files = meta.get("files", [])
    if not files:
        raise RuntimeError(
            "Figshare API returned no files; check the article id or your connection."
        )
    for entry in files:
        dest = out_dir / entry["name"]
        if dest.exists():
            print(f"  have {entry['name']}")
            continue
        print(f"  downloading {entry['name']} ...")
        dest.write_bytes(_read(entry["download_url"]))
    for archive in sorted(out_dir.glob("*.zip")):
        print(f"  extracting {archive.name} ...")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(out_dir)
    n_mat = len(list(out_dir.rglob("*.mat")))
    print(f"Figshare ready: {n_mat} .mat files under {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root", required=True, help="raw data root, e.g. $NEUROSCAN_ROOT/data/raw"
    )
    parser.add_argument("--skip-figshare", action="store_true", help="do not download Figshare")
    args = parser.parse_args()
    out_root = Path(args.out_root)
    if not args.skip_figshare:
        print("Figshare (Cheng brain-tumour dataset):")
        download_figshare(out_root / "figshare")
    print(SARTAJ_MESSAGE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
