import numpy as np

from neuroscan_ood.experiments.corruptions import CORRUPTIONS, SEVERITIES


def _img():
    return (np.random.RandomState(0).rand(32, 32) * 255).astype("uint8")


def test_all_preserve_shape_and_dtype():
    a = _img()
    for name, fn in CORRUPTIONS.items():
        for s in SEVERITIES:
            out = fn(a, s)
            assert out.shape == a.shape and out.dtype == np.uint8, name


def test_noise_is_deterministic():
    a = _img()
    f = CORRUPTIONS["gaussian_noise"]
    assert np.array_equal(f(a, 3), f(a, 3))


def test_higher_severity_changes_more():
    a = _img()
    f = CORRUPTIONS["gaussian_noise"]
    d1 = np.abs(f(a, 1).astype(int) - a).mean()
    d5 = np.abs(f(a, 5).astype(int) - a).mean()
    assert d5 > d1
