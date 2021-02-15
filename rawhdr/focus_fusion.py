"""Focus stacking functions."""

import numpy as np
import pywt
from scipy import ndimage

from .common import reduce_color_dimension, temporary_array_list
from .generic_fusion import swt_pad_funcs


def sharpness(image, sigma, *, pca):
    bw = reduce_color_dimension(image, pca=pca)
    bw_mean = ndimage.gaussian_filter(bw, sigma)
    return ndimage.gaussian_filter((bw - bw_mean)**2, sigma)


def max_sharpness_depth(sharpnesses):
    max_sharpness = np.copy(sharpnesses[0])
    depth = np.zeros_like(sharpnesses[0], dtype=np.uint16)
    for d, sharpness in enumerate(sharpnesses[1:]):
        mask = sharpness > max_sharpness
        max_sharpness[mask] = sharpness[mask]
        depth[mask] = d
    return max_sharpness, depth


def kmax_sharpnesses(sharpnesses, k):
    assert k <= len(sharpnesses)
    max_sharpnesses = np.empty_like(sharpnesses[0],
                                    shape=sharpnesses[0].shape + (k, ))
    for i, sharpness in enumerate(sharpnesses[:k]):
        max_sharpnesses[:, :, i] = sharpness
    max_sharpnesses.sort(axis=-1)
    for sharpness in sharpnesses[k:]:
        mask = sharpness > max_sharpnesses[:, :, 0]
        max_sharpnesses[:, :, 0][mask] = sharpness[mask]
        max_sharpnesses.sort(axis=-1)
    return max_sharpnesses


def weighted_depth(sharpnesses, depth, n):
    dtot = np.zeros_like(sharpnesses[0])
    dsum = np.zeros_like(sharpnesses[0])
    for d, sharpness in enumerate(sharpnesses):
        mask = np.abs(d - depth) <= n
        dtot[mask] += d * sharpness[mask]
        dsum[mask] += sharpness[mask]
    return dtot / dsum


def depth_sigma(sharpnesses, depth):
    stot = np.zeros_like(sharpnesses[0])
    ssum = np.zeros_like(sharpnesses[0])
    for d, sharpness in enumerate(sharpnesses):
        stot += (d - depth)**2 * sharpness
        ssum += sharpness
    return np.sqrt(stot / ssum)


def depth_rms(sharpnesses, max_sharpness, depth, sigma):
    error = np.zeros_like(max_sharpness)
    for d, sharpness in enumerate(sharpnesses):
        g = max_sharpness * np.exp(-(d - depth)**2 / (2 * sigma**2))
        error += (g - sharpness)**2
    return np.sqrt(error / len(sharpnesses))


def gaussian_weights(sharpnesses,
                     max_sharpness,
                     depth,
                     sigma,
                     *,
                     smooth,
                     in_memory=False):
    weights = temporary_array_list(in_memory=in_memory)
    for d in range(len(sharpnesses)):
        weights.append(
            ndimage.gaussian_filter(
                max_sharpness * np.exp(-(d - depth)**2 / (2 * sigma**2)),
                smooth))
    weights_sum = sum(weights)
    for w in weights:
        w /= weights_sum
    return weights


def fuse_images(images, weights):
    result = np.zeros_like(images[0])
    for image, weight in zip(images, weights):
        if weight.ndim == 2:
            weight = weight[..., np.newaxis]
        result += image * weight
    return result


def fuse_focal_stack(images,
                     *,
                     pca=None,
                     in_memory=None,
                     sharpness_sigma=None,
                     weighted_depth_n=None,
                     error_weight=None,
                     sigma_weight=None,
                     weights_smoothing=None):
    if pca is None:
        pca = True
    if in_memory is None:
        in_memory = False
    if sharpness_sigma is None:
        sharpness_sigma = 3
    if weighted_depth_n is None:
        weighted_depth_n = 10
    if error_weight is None:
        error_weight = 0.01
    if sigma_weight is None:
        sigma_weight = 0.05
    if weights_smoothing is None:
        weights_smoothing = 3

    sharpnesses = temporary_array_list(
        (sharpness(image, sharpness_sigma, pca=pca) for image in images),
        in_memory=in_memory)
    max_sharpness, raw_depth = max_sharpness_depth(sharpnesses)
    depth = weighted_depth(sharpnesses, raw_depth, weighted_depth_n)
    sigma = depth_sigma(sharpnesses, depth)
    relrms = depth_rms(sharpnesses, max_sharpness, depth,
                       sigma) / max_sharpness
    reconstruction_sigma = (error_weight * relrms * len(images) +
                            sigma_weight * sigma)
    weights = gaussian_weights(sharpnesses,
                               max_sharpness,
                               depth,
                               reconstruction_sigma,
                               smooth=weights_smoothing,
                               in_memory=in_memory)
    return fuse_images(images, weights)


def fuse_focal_stack_kmax(images,
                          *,
                          k=None,
                          levels=None,
                          wavelet=None,
                          pca=None,
                          in_memory=None,
                          sharpness_sigma=None):
    if k is None:
        k = 3
    if levels is None:
        levels = 3
    if wavelet is None:
        wavelet = 'sym4'
    if pca is None:
        pca = True
    if in_memory is None:
        in_memory = False
    if sharpness_sigma is None:
        sharpness_sigma = 3

    k = min(k, len(images))

    pad, unpad = swt_pad_funcs(images[0].shape, levels)
    sharpnesses = temporary_array_list(
        (sharpness(pad(image), sharpness_sigma, pca=pca) for image in images),
        in_memory=in_memory)
    kmax = kmax_sharpnesses(sharpnesses, k)

    bases = temporary_array_list()
    ads = [temporary_array_list() for _ in range(levels)]
    das = [temporary_array_list() for _ in range(levels)]
    dds = [temporary_array_list() for _ in range(levels)]

    for image in images:
        image = pad(image)
        shape = image.shape
        coeffs = pywt.swtn(image,
                           wavelet,
                           level=levels,
                           axes=(0, 1),
                           norm=True,
                           trim_approx=True)
        bases.append(coeffs[0])
        for l in range(levels):
            cl = coeffs[l + 1]
            ads[l].append(cl['ad'])
            das[l].append(cl['da'])
            dds[l].append(cl['dd'])

    base = np.empty_like(bases[0])
    ad = [np.empty_like(ads[l][0]) for l in range(levels)]
    da = [np.empty_like(das[l][0]) for l in range(levels)]
    dd = [np.empty_like(dds[l][0]) for l in range(levels)]
    first = np.full(shape[:2], True)

    for i, s in enumerate(sharpnesses):
        mask = np.any(s[:, :, np.newaxis] == kmax, axis=2)
        mask_and_first = mask & first
        base[mask_and_first, ...] = bases[i][mask_and_first, ...]
        for l in range(levels):
            ad[l][mask_and_first, ...] = ads[l][i][mask_and_first, ...]
            da[l][mask_and_first, ...] = das[l][i][mask_and_first, ...]
            dd[l][mask_and_first, ...] = dds[l][i][mask_and_first, ...]
        del mask_and_first
        mask_and_not_first = mask & ~first
        base[mask_and_not_first, ...] += bases[i][mask_and_not_first, ...]
        for l in range(levels):
            cmask = (
                (reduce_color_dimension(ads[l][i]**2) + reduce_color_dimension(
                    das[l][i]**2) + reduce_color_dimension(dds[l][i]**2)) >
                (reduce_color_dimension(ad[l]**2) +
                 reduce_color_dimension(da[l]**2) +
                 reduce_color_dimension(dd[l]**2))) & mask_and_not_first
            ad[l][cmask, ...] = ads[l][i][cmask, ...]
            da[l][cmask, ...] = das[l][i][cmask, ...]
            dd[l][cmask, ...] = dds[l][i][cmask, ...]
        first[mask] = False

    base /= k
    coeffs = [base
              ] + [dict(ad=ad[l], da=da[l], dd=dd[l]) for l in range(levels)]
    return unpad(pywt.iswtn(coeffs, wavelet, axes=(0, 1), norm=True))
