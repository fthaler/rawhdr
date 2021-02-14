"""Generic image fusion."""

import math

import numpy as np
import pywt

from .common import reduce_color_dimension


def _coeff_strength(cs, pca):
    return sum(reduce_color_dimension(c, pca=pca)**2 for c in cs)


def fuse_wavelets(first, second, *, levels=None, pca=False, wavelet=None):
    if wavelet is None:
        wavelet = 'sym4'
    first = pywt.wavedec2(first, wavelet, axes=(0, 1), level=levels)
    second = pywt.wavedec2(second, wavelet, axes=(0, 1), level=levels)

    first[0] = 0.5 * (first[0] + second[0])
    for first_cs, second_cs in zip(first[1:], second[1:]):
        mask = _coeff_strength(second_cs, pca) > _coeff_strength(first_cs, pca)
        for first_c, second_c in zip(first_cs, second_cs):
            first_c[mask, ...] = second_c[mask, ...]
    return pywt.waverec2(first, wavelet, axes=(0, 1))


def _pad_func(shape, divisor):
    padded_shape = [((s + divisor - 1) // divisor) * divisor
                    for s in shape[:2]]
    pad_width = [((p - s) // 2, (p - s) - (p - s) // 2)
                 for p, s in zip(padded_shape, shape)]
    while len(shape) > len(pad_width):
        pad_width += ((0, 0), )

    def pad(image):
        return np.pad(image, pad_width, mode='symmetric')

    def unpad(image):
        slices = tuple(slice(s, -e if e else None) for s, e in pad_width)
        return image[slices]

    return pad, unpad


def fuse_stationary_wavelets(first,
                             second,
                             *,
                             levels=None,
                             pca=False,
                             wavelet=None):
    if levels is None:
        levels = 6
    if wavelet is None:
        wavelet = 'sym4'

    pad, unpad = _pad_func(first.shape, 2**levels)
    first = pad(first)
    second = pad(second)

    first = pywt.swtn(first,
                      wavelet,
                      level=levels,
                      axes=(0, 1),
                      norm=True,
                      trim_approx=True)
    second = pywt.swtn(second,
                       wavelet,
                       level=levels,
                       axes=(0, 1),
                       norm=True,
                       trim_approx=True)

    first[0] = (first[0] + second[0]) / 2
    for first_cs, second_cs in zip(first[1:], second[1:]):
        mask = _coeff_strength(second_cs.values(), pca) > _coeff_strength(
            first_cs.values(), pca)
        for k, v in first_cs.items():
            v[mask, ...] = second_cs[k][mask, ...]
    del second
    first = pywt.iswtn(first, wavelet, axes=(0, 1), norm=True)
    return unpad(first)
