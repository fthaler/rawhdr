"""Generic image fusion."""

import pywt

from .common import reduce_color_dimension


def fuse_wavelets(first, second, *, levels=None, pca=False):
    wavelet = 'db6'
    first = pywt.wavedec2(first, wavelet, axes=(0, 1), level=levels)
    second = pywt.wavedec2(second, wavelet, axes=(0, 1), level=levels)

    first[0] = 0.5 * (first[0] + second[0])
    for first_cs, second_cs in zip(first[1:], second[1:]):
        mask = (sum(reduce_color_dimension(c, pca=pca)**2 for c in second_cs) >
                sum(reduce_color_dimension(c, pca=pca)**2 for c in first_cs))
        for first_c, second_c in zip(first_cs, second_cs):
            first_c[mask, ...] = second_c[mask, ...]
    return pywt.waverec2(first, wavelet, axes=(0, 1))
