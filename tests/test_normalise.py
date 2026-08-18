import numpy as np

from neuroscan_ood.data.normalise import to_uint8


def test_scales_to_full_range():
    out = to_uint8(np.array([[0, 100], [200, 400]], dtype=np.uint16))
    assert out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255
    assert out[1, 0] == 128  # 200/400 * 255 = 127.5 -> 128


def test_flat_image_is_zero():
    assert (to_uint8(np.full((3, 3), 7, dtype=np.uint16)) == 0).all()
