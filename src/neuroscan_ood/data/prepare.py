"""Read raw Figshare (.mat) and SARTAJ (JPG) data into prepared 8-bit PNGs + manifest.

Figshare .mat files are v7.3/HDF5; read with h5py. Each has a `cjdata` group with
`image`, `label`, and `PID`. Known-bad indices are excluded; native image size is recorded and
handled by the resize in the transform pipeline. SARTAJ is optional here: if its raw folder is
absent, only Figshare is processed (Phase 1 is Figshare-only).
"""

import argparse
import csv
import sys
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from neuroscan_ood.data.normalise import to_uint8
from neuroscan_ood.utils.logging import get_logger

log = get_logger("prepare")

# Canonical label encoding used throughout the pipeline.
CANON = {"meningioma": 0, "glioma": 1, "pituitary": 2}
# Figshare on-disk integer labels -> canonical name.
FIGSHARE_LABEL_TO_NAME = {1: "meningioma", 2: "glioma", 3: "pituitary"}
# SARTAJ folder names -> canonical name. SARTAJ 'notumor' is not part of the 3-class study.
SARTAJ_DIR_TO_NAME = {
    "glioma_tumor": "glioma",
    "meningioma_tumor": "meningioma",
    "pituitary_tumor": "pituitary",
    "glioma": "glioma",
    "meningioma": "meningioma",
    "pituitary": "pituitary",
}
# Figshare indices documented as corrupt in the dataset README.
KNOWN_BAD_FIGSHARE = set(range(954, 957)) | set(range(1069, 1076)) | set(range(1202, 1207))


def _decode_pid(pid_ds) -> str:
    codes = np.array(pid_ds).ravel()
    return "".join(chr(int(c)) for c in codes)


def prepare_figshare(raw_dir: Path, images_out: Path, rows: list, excluded: list) -> None:
    def _key(p):
        return int(p.stem) if p.stem.isdigit() else 10**9

    mat_files = sorted(raw_dir.glob("*.mat"), key=_key)
    if not mat_files:
        log.warning("no .mat files under %s; skipping Figshare", raw_dir)
        return
    for path in mat_files:
        idx = int(path.stem) if path.stem.isdigit() else -1
        if idx in KNOWN_BAD_FIGSHARE:
            excluded.append((str(path), "figshare", "corrupt_known_index"))
            continue
        try:
            with h5py.File(path, "r") as f:
                cj = f["cjdata"]
                image = np.array(cj["image"]).T  # HDF5 stores transposed vs MATLAB
                label_int = int(np.array(cj["label"]).ravel()[0])
                pid = _decode_pid(cj["PID"])
        except Exception:
            excluded.append((str(path), "figshare", "unreadable"))
            continue
        if label_int not in FIGSHARE_LABEL_TO_NAME:
            excluded.append((str(path), "figshare", "unmapped_label"))  # never coerce
            continue
        name = FIGSHARE_LABEL_TO_NAME[label_int]
        dtype_min, dtype_max = int(image.min()), int(image.max())
        img8 = to_uint8(image)
        h, w = img8.shape
        fname = f"figshare_{path.stem}.png"
        Image.fromarray(img8).save(images_out / fname)
        rows.append(
            {
                "filename": fname,
                "source": "figshare",
                "label": CANON[name],
                "label_name": name,
                "pid": pid,
                "height": h,
                "width": w,
                "dtype_min": dtype_min,
                "dtype_max": dtype_max,
                "glioma_flag": 0,
            }
        )


def prepare_sartaj(raw_dir: Path, images_out: Path, rows: list, excluded: list) -> None:
    if not raw_dir.exists():
        log.info("SARTAJ raw dir %s absent; skipping (Phase 1 is Figshare-only)", raw_dir)
        return
    exts = {".jpg", ".jpeg", ".png"}
    count = 0
    for img_path in sorted(raw_dir.rglob("*")):
        if img_path.suffix.lower() not in exts:
            continue
        name = SARTAJ_DIR_TO_NAME.get(img_path.parent.name.lower())
        if name is None:
            excluded.append((str(img_path), "sartaj", "unmapped_label"))
            continue
        try:
            arr = np.array(Image.open(img_path).convert("L"))
        except Exception:
            excluded.append((str(img_path), "sartaj", "unreadable"))
            continue
        img8 = to_uint8(arr)
        h, w = img8.shape
        fname = f"sartaj_{count:05d}.png"
        Image.fromarray(img8).save(images_out / fname)
        rows.append(
            {
                "filename": fname,
                "source": "sartaj",
                "label": CANON[name],
                "label_name": name,
                "pid": f"sartaj_{count:05d}",
                "height": h,
                "width": w,
                "dtype_min": int(arr.min()),
                "dtype_max": int(arr.max()),
                "glioma_flag": 1 if name == "glioma" else 0,
            }
        )
        count += 1


def main(raw_root: Path, out_root: Path, expect_figshare_counts: bool = True) -> int:
    raw_root, out_root = Path(raw_root), Path(out_root)
    images_out = out_root / "images"
    images_out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    excluded: list[list] = []
    prepare_figshare(raw_root / "figshare", images_out, rows, excluded)
    prepare_sartaj(raw_root / "sartaj", images_out, rows, excluded)

    fieldnames = [
        "filename",
        "source",
        "label",
        "label_name",
        "pid",
        "height",
        "width",
        "dtype_min",
        "dtype_max",
        "glioma_flag",
    ]
    with open(out_root / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    with open(out_root / "excluded.csv", "w", newline="") as f:
        ew = csv.writer(f)
        ew.writerow(["raw_path", "source", "reason"])
        ew.writerows(excluded)

    fig_rows = [r for r in rows if r["source"] == "figshare"]
    fig_excl = [e for e in excluded if e[1] == "figshare"]
    n_pid = len({r["pid"] for r in fig_rows})
    log.info(
        "Figshare: %d prepared, %d excluded, %d unique PIDs", len(fig_rows), len(fig_excl), n_pid
    )
    log.info("SARTAJ: %d prepared", len([r for r in rows if r["source"] == "sartaj"]))
    if expect_figshare_counts and fig_rows:
        total = len(fig_rows) + len(fig_excl)
        if total != 3064:
            log.error("Figshare prepared+excluded=%d, expected 3064", total)
            return 1
        if n_pid != 233:
            log.error("Figshare unique PIDs=%d, expected 233", n_pid)
            return 1
    log.info("wrote manifest.csv (%d rows), excluded.csv (%d rows)", len(rows), len(excluded))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--out-root", required=True)
    args = ap.parse_args()
    sys.exit(main(Path(args.raw_root), Path(args.out_root)))
