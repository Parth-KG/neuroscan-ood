import pandas as pd

from neuroscan_ood.data.splits import make_split

NAMES = ["meningioma", "glioma", "pituitary"]


def _manifest(n_patients=12, per=8):
    rows = []
    for p in range(n_patients):
        for k in range(per):
            rows.append(
                {
                    "filename": f"f_{p}_{k}.png",
                    "source": "figshare",
                    "label": p % 3,
                    "label_name": NAMES[p % 3],
                    "pid": f"P{p:03d}",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_split_has_no_shared_pid():
    tr, te = make_split(_manifest(), "grouped", seed=0)
    assert set(tr["pid"]).isdisjoint(set(te["pid"]))


def test_grouped_split_is_reproducible():
    a = make_split(_manifest(), "grouped", 0)[0]["filename"].tolist()
    b = make_split(_manifest(), "grouped", 0)[0]["filename"].tolist()
    assert a == b


def test_random_split_is_reproducible():
    a = make_split(_manifest(), "random", 0)[1]["filename"].tolist()
    b = make_split(_manifest(), "random", 0)[1]["filename"].tolist()
    assert a == b
