"""The single 16-bit -> 8-bit conversion rule.

This conversion is itself an intensity normalisation. Per-image min-max scaling makes each
image independently full-range, which removes part of the absolute-intensity difference
between sources. That is a deliberate, fixed, documented convention here; the explicit
intensity intervention for the study is arm A (histogram matching to the training source),
not this function.
"""

import numpy as np


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(img.min()), float(img.max())
    if hi <= lo:
        return np.zeros(img.shape, dtype=np.uint8)
    scaled = (img - lo) / (hi - lo) * 255.0
    return np.rint(scaled).astype(np.uint8)
