"""Train/test splits. Random and patient-grouped, both seed-reproducible."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split

if TYPE_CHECKING:
    import pandas as pd


def make_split(
    manifest: pd.DataFrame, strategy: str, seed: int, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = manifest.reset_index(drop=True)
    if strategy == "random":
        tr_idx, te_idx = train_test_split(
            np.arange(len(df)),
            test_size=test_size,
            random_state=seed,
            stratify=df["label"].values,
        )
    elif strategy == "grouped":
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(df, df["label"].values, groups=df["pid"].values))
    else:
        raise ValueError(f"unknown split strategy: {strategy}")
    train, test = df.iloc[tr_idx].copy(), df.iloc[te_idx].copy()
    if strategy == "grouped":
        shared = set(train["pid"]) & set(test["pid"])
        if shared:
            raise ValueError(
                f"grouped split has shared pid across train/test: {sorted(shared)[:5]}"
            )
    return train, test
