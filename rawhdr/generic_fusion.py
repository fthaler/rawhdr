"""Generic image fusion."""

import numpy as np
import pywt


def fuse_wavelets(first, second, *, levels=None):
    wavelet = 'db6'
    first = pywt.wavedec2(first, wavelet, axes=(0, 1), level=levels)
    second = pywt.wavedec2(second, wavelet, axes=(0, 1), level=levels)

    first[0] = 0.5 * (first[0] + second[0])
    for first_cs, second_cs in zip(first[1:], second[1:]):
        mask = sum(c**2 for c in second_cs) > sum(c**2 for c in first_cs)
        if mask.ndim == 3:
            mask = np.broadcast_to(
                np.median(mask, axis=2, keepdims=True).astype(bool),
                first_cs[0].shape)
        for first_c, second_c in zip(first_cs, second_cs):
            first_c[mask] = second_c[mask]
    return pywt.waverec2(first, wavelet, axes=(0, 1))
