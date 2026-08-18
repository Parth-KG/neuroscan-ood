"""Load and validate manifest.csv."""

import os

import pandas as pd

CANON_NAMES = {0: "meningioma", 1: "glioma", 2: "pituitary"}
REQUIRED_COLS = [
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


def load_manifest(prepared_root) -> pd.DataFrame:
    path = os.path.join(str(prepared_root), "manifest.csv")
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns: {missing}")
    bad = df["label"].map(CANON_NAMES) != df["label_name"]
    if bad.any():
        raise ValueError(f"label/label_name mismatch in {int(bad.sum())} rows")
    return df


def assert_figshare_counts(
    df, n_excluded_figshare: int, expected_rows: int = 3064, expected_pids: int = 233
) -> None:
    fig = df[df["source"] == "figshare"]
    total = len(fig) + n_excluded_figshare
    if total != expected_rows:
        raise ValueError(f"Figshare rows+excluded={total}, expected {expected_rows}")
    n_pid = fig["pid"].nunique()
    if n_pid != expected_pids:
        raise ValueError(f"Figshare unique PIDs={n_pid}, expected {expected_pids}")
