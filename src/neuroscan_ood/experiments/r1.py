"""Result R1 - patient-level leakage. Random vs patient-grouped split over seeds.

Trains the same EfficientNet-B0 on Figshare under each split strategy and each seed, then reports
each split's test accuracy as mean +/- spread and the gap between them. Runs are idempotent: a run
whose metrics.json already exists is skipped, so an interrupted sweep can simply be re-run.
"""

import copy
import csv
import json
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from neuroscan_ood.train.loop import train
from neuroscan_ood.utils.config import load_config
from neuroscan_ood.utils.logging import get_logger
from neuroscan_ood.utils.paths import run_dir, runs_root

log = get_logger("r1")


def _run_one(base_cfg, split, seed):
    cfg = copy.deepcopy(base_cfg)
    cfg["seed"] = seed
    cfg["split"] = split
    cfg["run_id"] = f"r1_{split}_seed{seed}"
    metrics_path = run_dir(cfg["run_id"]) / "metrics.json"
    if metrics_path.exists():
        log.info("skip %s (already done)", cfg["run_id"])
    else:
        log.info("train %s", cfg["run_id"])
        train(cfg)
    return json.loads(metrics_path.read_text())["accuracy"]


def summarise(results: dict) -> dict:
    """Pure aggregation: {split: [acc,...]} -> means, spreads, and the random-grouped gap."""

    def stats(xs):
        return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)

    rm, rs = stats(results["random"])
    gm, gs = stats(results["grouped"])
    return {
        "random_mean": rm,
        "random_std": rs,
        "grouped_mean": gm,
        "grouped_std": gs,
        "gap": rm - gm,
    }


def main(base_config, seeds):
    base_cfg = load_config(base_config)
    results = {"random": [], "grouped": []}
    for split in ["random", "grouped"]:
        for seed in seeds:
            acc = _run_one(base_cfg, split, seed)
            results[split].append(acc)
            log.info("%s seed%d accuracy=%.4f", split, seed, acc)

    s = summarise(results)
    out = runs_root() / "r1"
    out.mkdir(parents=True, exist_ok=True)

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "seed", "accuracy"])
        for split in ["random", "grouped"]:
            for seed, acc in zip(seeds, results[split], strict=False):
                w.writerow([split, seed, f"{acc:.6f}"])
        w.writerow([])
        w.writerow(["split", "mean", "stdev"])
        w.writerow(["random", f"{s['random_mean']:.6f}", f"{s['random_std']:.6f}"])
        w.writerow(["grouped", f"{s['grouped_mean']:.6f}", f"{s['grouped_std']:.6f}"])
        w.writerow(["gap_random_minus_grouped", f"{s['gap']:.6f}", ""])

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.bar(
        ["random\n(leaky)", "patient-grouped\n(honest)"],
        [s["random_mean"], s["grouped_mean"]],
        yerr=[s["random_std"], s["grouped_std"]],
        capsize=6,
        color=["#c0504d", "#4f81bd"],
    )
    ax.set_ylabel("test accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(f"R1: leakage inflates accuracy by {s['gap'] * 100:.1f} pts")
    for i, (m, _sd) in enumerate(
        [(s["random_mean"], s["random_std"]), (s["grouped_mean"], s["grouped_std"])]
    ):
        ax.text(i, m + 0.02, f"{m * 100:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out / "r1_accuracy.png", dpi=120)
    plt.close(fig)

    log.info("R1 random = %.4f +/- %.4f", s["random_mean"], s["random_std"])
    log.info("R1 grouped = %.4f +/- %.4f", s["grouped_mean"], s["grouped_std"])
    log.info("R1 gap = %.4f", s["gap"])
    print("\nR1 RESULT")
    print(f" random-split (leaky): {s['random_mean'] * 100:.1f}% +/- {s['random_std'] * 100:.1f}")
    print(
        f" patient-grouped (honest): {s['grouped_mean'] * 100:.1f}% +/- {s['grouped_std'] * 100:.1f}"
    )
    print(f" leakage inflation: {s['gap'] * 100:.1f} percentage points")
    return 0
