import numpy as np
from PIL import Image

from neuroscan_ood.experiments.audit import compute_min_distances


def _save(path, arr):
    Image.fromarray(arr.astype("uint8")).save(path)


def test_identical_image_has_distance_zero(tmp_path):
    a = np.random.RandomState(0).rand(64, 64) * 255
    p = tmp_path / "a.png"
    _save(p, a)
    min_d, _ = compute_min_distances([p], [p])
    assert min_d[0] == 0  # a duplicate across sources must register as distance 0


def test_different_images_have_positive_distance(tmp_path):
    a = np.tile(np.linspace(0, 255, 64), (64, 1))  # horizontal gradient
    b = np.random.RandomState(1).rand(64, 64) * 255  # noise
    pa, pb = tmp_path / "a.png", tmp_path / "b.png"
    _save(pa, a)
    _save(pb, b)
    min_d, _ = compute_min_distances([pa], [pb])
    assert min_d[0] > 0
