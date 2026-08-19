"""Controlled acquisition-shift corruptions for R2.

Each function maps a uint8 grayscale image and a severity level (1..5) to a corrupted uint8
image of the same shape. All are deterministic given (image, severity), so an evaluation sweep
is reproducible. These mimic scanner/acquisition differences rather than adversarial noise.
"""

import numpy as np
from PIL import Image, ImageFilter


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0, 255).astype(np.uint8)


def brightness(a: np.ndarray, s: int) -> np.ndarray:
    add = [0.10, 0.20, 0.30, 0.40, 0.50][s - 1] * 255
    return _clip(a.astype(np.float32) + add)


def contrast(a: np.ndarray, s: int) -> np.ndarray:
    f = [0.80, 0.65, 0.50, 0.35, 0.25][s - 1]
    m = float(a.mean())
    return _clip((a.astype(np.float32) - m) * f + m)


def gaussian_noise(a: np.ndarray, s: int) -> np.ndarray:
    std = [0.04, 0.08, 0.12, 0.18, 0.26][s - 1] * 255
    rng = np.random.RandomState(1000 + s)  # fixed realisation -> reproducible
    return _clip(a.astype(np.float32) + rng.normal(0.0, std, a.shape))


def blur(a: np.ndarray, s: int) -> np.ndarray:
    radius = [0.5, 1.0, 1.5, 2.5, 3.5][s - 1]
    return np.array(Image.fromarray(a).filter(ImageFilter.GaussianBlur(radius)))


def downsample(a: np.ndarray, s: int) -> np.ndarray:
    scale = [0.75, 0.60, 0.50, 0.40, 0.30][s - 1]
    h, w = a.shape
    small = Image.fromarray(a).resize(
        (max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR
    )
    return np.array(small.resize((w, h), Image.Resampling.BILINEAR))


def bias_field(a: np.ndarray, s: int) -> np.ndarray:
    # Smooth low-frequency multiplicative field: the classic MRI intensity inhomogeneity artifact.
    strength = [0.10, 0.20, 0.30, 0.45, 0.60][s - 1]
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = np.sin(np.pi * xx / max(w - 1, 1)) * np.sin(np.pi * yy / max(h - 1, 1))
    return _clip(a.astype(np.float32) * (1.0 + strength * (field - float(field.mean()))))


CORRUPTIONS = {
    "brightness": brightness,
    "contrast": contrast,
    "gaussian_noise": gaussian_noise,
    "blur": blur,
    "downsample": downsample,
    "bias_field": bias_field,
}
SEVERITIES = [1, 2, 3, 4, 5]
