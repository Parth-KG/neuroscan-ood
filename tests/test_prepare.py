import h5py
import numpy as np

from neuroscan_ood.data.prepare import prepare_figshare


def _write_mat(path, image, label, pid):
    with h5py.File(path, "w") as f:
        g = f.create_group("cjdata")
        g.create_dataset("image", data=image.T.astype(np.float64))  # store MATLAB-style
        g.create_dataset("label", data=np.array([[float(label)]]))
        g.create_dataset("PID", data=np.array([[float(ord(c))] for c in pid]))


def test_reads_mat_and_maps_labels(tmp_path):
    raw = tmp_path / "figshare"
    raw.mkdir()
    out = tmp_path / "prepared" / "images"
    out.mkdir(parents=True)
    _write_mat(raw / "1.mat", np.random.randint(0, 4096, (64, 64)), 1, "P001")
    _write_mat(raw / "2.mat", np.random.randint(0, 4096, (64, 64)), 3, "P002")
    rows, excluded = [], []
    prepare_figshare(raw, out, rows, excluded)
    assert len(rows) == 2
    assert {r["label_name"] for r in rows} == {"meningioma", "pituitary"}
    assert rows[0]["pid"] == "P001"
    assert (out / rows[0]["filename"]).exists()


def test_excludes_known_bad_index(tmp_path):
    raw = tmp_path / "figshare"
    raw.mkdir()
    out = tmp_path / "prepared" / "images"
    out.mkdir(parents=True)
    _write_mat(raw / "955.mat", np.random.randint(0, 4096, (64, 64)), 2, "P900")
    rows, excluded = [], []
    prepare_figshare(raw, out, rows, excluded)
    assert rows == []
    assert excluded and excluded[0][2] == "corrupt_known_index"
