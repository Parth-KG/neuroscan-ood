"""Source-independence audit via perceptual hashing.

Computes a perceptual hash (pHash) for every prepared image and, for each SARTAJ image, its
smallest Hamming distance to any Figshare image. If any cross-source pair is within the documented
threshold, the sources are not independent (FAIL): near-duplicate images shared across sources
would leak between the training source and the external-test source and invalidate R2/R3/R4.
"""

import csv

import matplotlib

matplotlib.use("Agg")
import imagehash
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from neuroscan_ood.data.manifest import load_manifest
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import images_root, prepared_root, runs_root

log = get_logger("audit")

HAMMING_THRESHOLD = 5  # <=5 of 64 bits differing counts as a near-duplicate (documented choice)


def _phash_bits(path):
    return imagehash.phash(Image.open(path)).hash.flatten()  # bool array length 64


def compute_min_distances(fig_paths, sar_paths):
    """For each SARTAJ image, the smallest pHash Hamming distance to any Figshare image."""
    F = np.stack([_phash_bits(p) for p in fig_paths])
    S = np.stack([_phash_bits(p) for p in sar_paths])
    min_d = np.empty(len(S), dtype=int)
    argmin = np.empty(len(S), dtype=int)
    for i in range(len(S)):
        d = (F != S[i]).sum(axis=1)
        j = int(d.argmin())
        min_d[i], argmin[i] = int(d[j]), j
    return min_d, argmin


def main(threshold: int = HAMMING_THRESHOLD):
    df = load_manifest(prepared_root())
    root = images_root()
    fig = df[df["source"] == "figshare"]
    sar = df[df["source"] == "sartaj"]
    if len(sar) == 0:
        log.error("no SARTAJ images in manifest; add the SARTAJ data and re-run prepare_data.py")
        return 2
    fig_paths = [root / f for f in fig["filename"]]
    sar_paths = [root / f for f in sar["filename"]]
    log.info("hashing %d figshare and %d sartaj images", len(fig_paths), len(sar_paths))
    min_d, argmin = compute_min_distances(fig_paths, sar_paths)

    out = runs_root() / "audit"
    out.mkdir(parents=True, exist_ok=True)

    fig_, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(min_d, bins=range(0, 34), color="#4f81bd")
    ax.axvline(threshold + 0.5, color="red", ls="--", label=f"threshold = {threshold}")
    ax.set_xlabel("min pHash Hamming distance to any Figshare image")
    ax.set_ylabel("SARTAJ image count")
    ax.legend()
    fig_.tight_layout()
    fig_.savefig(out / "distances_hist.png", dpi=120)
    plt.close(fig_)

    dup_idx = np.where(min_d <= threshold)[0]
    with open(out / "near_duplicates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sartaj_file", "nearest_figshare_file", "hamming"])
        for i in dup_idx:
            w.writerow([sar_paths[i].name, fig_paths[argmin[i]].name, int(min_d[i])])

    verdict = "PASS" if len(dup_idx) == 0 else "FAIL"
    log.info(
        "min distance across sources: min=%d median=%d", int(min_d.min()), int(np.median(min_d))
    )
    log.info("cross-source near-duplicates (<=%d): %d", threshold, len(dup_idx))
    print(f"\nSOURCE-INDEPENDENCE AUDIT: {verdict}")
    print(f" near-duplicate pairs within Hamming {threshold}: {len(dup_idx)}")
    print(f" closest cross-source distance found: {int(min_d.min())}")
    if verdict == "FAIL":
        print(" see runs/audit/near_duplicates.csv for the offending pairs")
    return 0
