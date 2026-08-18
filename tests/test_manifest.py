import pandas as pd
import pytest

from neuroscan_ood.data.manifest import assert_figshare_counts, load_manifest


def _df(pids):
    return pd.DataFrame(
        [{"source": "figshare", "pid": p, "label": 0, "label_name": "meningioma"} for p in pids]
    )


def test_counts_pass():
    pids = [f"P{i}" for i in range(233) for _ in range(2)]  # 233 unique, 466 rows
    assert_figshare_counts(_df(pids), n_excluded_figshare=3064 - 466)


def test_counts_fail_on_wrong_pid_total():
    pids = [f"P{i}" for i in range(10)]
    with pytest.raises(ValueError):
        assert_figshare_counts(_df(pids), n_excluded_figshare=3064 - 10)


def test_load_manifest_rejects_label_mismatch(tmp_path):
    row = {
        "filename": "a.png",
        "source": "figshare",
        "label": 0,
        "label_name": "glioma",
        "pid": "P1",
        "height": 1,
        "width": 1,
        "dtype_min": 0,
        "dtype_max": 1,
        "glioma_flag": 0,
    }
    pd.DataFrame([row]).to_csv(tmp_path / "manifest.csv", index=False)
    with pytest.raises(ValueError):
        load_manifest(tmp_path)
