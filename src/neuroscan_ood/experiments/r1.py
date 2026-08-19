"""Result R1 - patient-level leakage. Random vs patient-grouped split over seeds.

Trains the same EfficientNet-B0 on Figshare under each split strategy and each seed, then reports
each split's test accuracy with a 95% confidence interval and a paired t-test on the per-seed gap.
Runs are idempotent: a run whose metrics.json already exists is skipped, so an interrupted sweep
can simply be re-run.
"""

import copy
import csv
import json
import math
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sstats

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


def _ci(xs):
    """Mean, sample std, and 95% t-interval for a small sample."""
    n = len(xs)
    m = statistics.mean(xs)
    if n < 2:
        return m, 0.0, m, m
    sd = statistics.stdev(xs)  # sample std (ddof=1)
    half = sstats.t.ppf(0.975, n - 1) * sd / math.sqrt(n)
    return m, sd, m - half, m + half


def summarise(results: dict) -> dict:
    """{split: [acc per seed]} -> means, 95% CIs, the paired gap, and a paired t-test."""
    r, g = results["random"], results["grouped"]
    rm, rsd, rlo, rhi = _ci(r)
    gm, gsd, glo, ghi = _ci(g)
    diffs = [a - b for a, b in zip(r, g, strict=False)]
    _dm, _dsd, dlo, dhi = _ci(diffs)
    n = len(r)
    if n >= 2:
        t, p = sstats.ttest_rel(r, g)
        t, p = float(t), float(p)
    else:
        t, p = float("nan"), float("nan")
    return {
        "random_mean": rm,
        "random_std": rsd,
        "random_ci": (rlo, rhi),
        "grouped_mean": gm,
        "grouped_std": gsd,
        "grouped_ci": (glo, ghi),
        "gap": rm - gm,
        "gap_ci": (dlo, dhi),
        "t": t,
        "p": p,
        "n": n,
        "all_positive": all(d > 0 for d in diffs) if diffs else False,
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
        w.writerow(["split", "mean", "sample_stdev", "ci95_low", "ci95_high"])
        w.writerow(
            [
                "random",
                f"{s['random_mean']:.6f}",
                f"{s['random_std']:.6f}",
                f"{s['random_ci'][0]:.6f}",
                f"{s['random_ci'][1]:.6f}",
            ]
        )
        w.writerow(
            [
                "grouped",
                f"{s['grouped_mean']:.6f}",
                f"{s['grouped_std']:.6f}",
                f"{s['grouped_ci'][0]:.6f}",
                f"{s['grouped_ci'][1]:.6f}",
            ]
        )
        w.writerow([])
        w.writerow(["gap", f"{s['gap']:.6f}", "", f"{s['gap_ci'][0]:.6f}", f"{s['gap_ci'][1]:.6f}"])
        w.writerow(["paired_t_test", f"t={s['t']:.4f}", f"p={s['p']:.4f}", f"n={s['n']}", ""])

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
    ax.set_title(f"R1: leakage gap {s['gap'] * 100:.1f} pts (p={s['p']:.3f}, n={s['n']})")
    for i, m in enumerate([s["random_mean"], s["grouped_mean"]]):
        ax.text(i, m + 0.02, f"{m * 100:.1f}%", ha="center")
    fig.tight_layout()
    fig.savefig(out / "r1_accuracy.png", dpi=120)
    plt.close(fig)

    def clip(x):
        return min(max(x, 0.0), 1.0)

    sig = "significant" if s["p"] < 0.05 else "NOT significant"
    print("\nR1 RESULT")
    print(
        f"  random (leaky):   {s['random_mean'] * 100:.1f}%"
        f"  95% CI [{clip(s['random_ci'][0]) * 100:.1f}, {clip(s['random_ci'][1]) * 100:.1f}]"
    )
    print(
        f"  grouped (honest): {s['grouped_mean'] * 100:.1f}%"
        f"  95% CI [{clip(s['grouped_ci'][0]) * 100:.1f}, {clip(s['grouped_ci'][1]) * 100:.1f}]"
    )
    print(
        f"  leakage gap:      {s['gap'] * 100:.1f} pts"
        f"  95% CI [{s['gap_ci'][0] * 100:.1f}, {s['gap_ci'][1] * 100:.1f}]"
    )
    print(f"  paired t-test:    t={s['t']:.2f}, p={s['p']:.3f} (n={s['n']})  ->  {sig} at 0.05")
    print(f"  all seeds show inflation: {s['all_positive']}")
    return 0
