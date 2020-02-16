"""Functions for focus stacking of images."""

import numpy as np
from scipy import ndimage
import pywt
import dtcwt


def merge_dtcwt(first, second, levels=None, sigma=None):
    """Focus-stack two images using the dual-tree complex wavelet
       transform (DTCWT).

    Parameters
    ----------
    first : array_like
        First input image.
    second : array_like
        Second input image.
    levels : int
        Number of levels to use in the DTCWT.
    sigma : float
        Standard deviation of lowpass filter used to smooth the mixing weights.

    Returns
    -------
    stacked : numpy.ndarray
        Focus-stacked result.
    """
    if levels is None:
        levels = 3
    if sigma is None:
        sigma = 1.5

    transform = dtcwt.Transform2d()

    def merge_channel(fc, sc):
        fc = transform.forward(fc, nlevels=levels)
        sc = transform.forward(sc, nlevels=levels)
        fc.lowpass[:] = 0.5 * (fc.lowpass + sc.lowpass)
        for f, s in zip(fc.highpasses, sc.highpasses):
            mix = 1.0 * (np.abs(f) > np.abs(s))
            mix = np.median(mix, axis=2)
            mix = ndimage.gaussian_filter(mix, sigma=sigma)
            mix = mix[:, :, np.newaxis]
            f[:] = mix * f + (1 - mix) * s
        return transform.inverse(fc)

    if first.ndim == 2:
        return merge_channel(first, second)

    return np.dstack([
        merge_channel(first[:, :, c], second[:, :, c])
        for c in range(first.ndim)
    ])


def merge_highpass(first, second, sigma=None):
    """Focus-stack two images using a Gaussian highpass filter.

    Parameters
    ----------
    first : array_like
        First input image.
    second : array_like
        Second input image.
    sigma : float
        Standard deviation of the highpass filter.

    Returns
    -------
    stacked : numpy.ndarray
        Focus-stacked result.
    """
    if sigma is None:
        sigma = 1.5

    def sharpness(image):
        bw = image
        if image.ndim == 3:
            bw = np.mean(bw, axis=2)
        lowpass = ndimage.gaussian_filter(bw, sigma=sigma)
        highpass = bw - lowpass
        return ndimage.gaussian_filter(np.abs(highpass), sigma=sigma)

    first_sharpness = sharpness(first)
    second_sharpness = sharpness(second)
    diff = first_sharpness - second_sharpness
    mix = 1 / (1 + np.exp(-100 * diff))
    if first.ndim == 3:
        mix = mix[:, :, np.newaxis]
    return mix * first + (1 - mix) * second


def merge_waveletes(first, second, levels=None, sigma=None):
    """Focus-stack two images using a wavelet transform.

    Parameters
    ----------
    first : array_like
        First input image.
    second : array_like
        Second input image.
    sigma : float
        Standard deviation of lowpass filter used to smooth the mixing weights.

    Returns
    -------
    stacked : numpy.ndarray
        Focus-stacked result.
    """
    if levels is None:
        levels = 4
    if sigma is None:
        sigma = 1.5

    rgb = first.ndim == 3
    wavelet = 'db6'
    first = pywt.wavedec2(first, wavelet, axes=(0, 1))
    second = pywt.wavedec2(second, wavelet, axes=(0, 1))

    coeffs = [0.5 * (first[0] + second[0])]
    for first_cs, second_cs in zip(first[1:], second[1:]):
        merged_cs = []
        for first_c, second_c in zip(first_cs, second_cs):
            mix = 1.0 * (np.abs(first_c) > np.abs(second_c))
            if rgb:
                mix = np.mean(mix, axis=2)
            mix = ndimage.gaussian_filter(mix, sigma=sigma)
            if rgb:
                mix = mix[:, :, np.newaxis]
            merged_cs.append(mix * first_c + (1 - mix) * second_c)
        coeffs.append(merged_cs)

    return np.maximum(pywt.waverec2(coeffs, wavelet, axes=(0, 1)), 0)


def merge_waveletes2(first, second, levels=None):
    """Focus-stack two images using a wavelet transform.

    Parameters
    ----------
    first : array_like
        First input image.
    second : array_like
        Second input image.
    sigma : float
        Standard deviation of lowpass filter used to smooth the mixing weight.

    Returns
    -------
    stacked : numpy.ndarray
        Focus-stacked result.
    """
    if levels is None:
        levels = 4

    wavelet = 'db6'
    rgb = first.ndim == 3

    first_coeffs = pywt.wavedec2(np.mean(first, axis=2) if rgb else first,
                                 wavelet,
                                 level=levels,
                                 axes=(0, 1))[1:]
    second_coeffs = pywt.wavedec2(np.mean(second, axis=2) if rgb else second,
                                  wavelet,
                                  level=levels,
                                  axes=(0, 1))[1:]

    mix = np.full(first.shape[:2], 0.5, dtype='float32')

    for f, s in zip(reversed(first_coeffs), reversed(second_coeffs)):
        f = sum(np.abs(fi) for fi in f)
        s = sum(np.abs(si) for si in s)
        f = ndimage.zoom(
            f, (first.shape[0] / f.shape[0], first.shape[1] / f.shape[1]))
        s = ndimage.zoom(
            s, (first.shape[0] / s.shape[0], first.shape[1] / s.shape[1]))
        undefined = mix == 0.5

        fbig = np.bitwise_and(undefined, f > 2 * s)
        sbig = np.bitwise_and(undefined, s > 2 * f)
        mix[fbig] = 1
        mix[sbig] = 0

    if rgb:
        mix = mix[:, :, np.newaxis]
    return mix * first + (1.0 - mix) * second
