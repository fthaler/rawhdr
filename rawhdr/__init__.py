# -*- coding: utf-8 -*-

import numpy as np
import rawpy


from .version import __version__


def load_image(path):
    """Loads a raw image file.

    Parameters
    ----------

    path : str
        Path to the image file.

    Returns
    -------

    image : ndarray
        Loaded image data in linear color space.
    """
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(gamma=(1, 1), no_auto_bright=True,
                              use_camera_wb=True, output_bps=16) / 2.0**16
    return rgb


def merge_exposures(base, other, mask_width=0.8, blend_width=None,
                    target_gamma=2.2):
    """Merges an LDR image into a (possibly HDR) base image.

    Parameters
    ----------

    base : array_like
        Floating point input image in linear color space and desired exposure.
    other : array_like
        Floating point input image that will be merged into `base`.
    mask_width : float
        Width of the mask (in gamma corrected color space) that defines the
        well-working range of the sensor (low noise, not saturated).
        Default: 0.8.
    blend_width : float
        Width of the blending region where the `other` in gamma corrected
        color space. Defaults to 1 - `mask_width`.
    target_gamma : float
        Gamma of color space to perform the blending. Only used for mask
        and blending computation. Default: 2.2.

    Returns
    -------

    merged : ndarray
        Merged HDR image of `base` and `other` with same exposure as `base`.
    """
    base = np.asarray(base)
    other = np.asarray(other)

    if mask_width <= 0 or mask_width > 1:
        raise ValueError('Mask width must be positive and at most 1')

    # Compute mask where all image data is in reasonable sensor range
    mask_min = 0.5 - 0.5 * mask_width
    mask_max = 0.5 + 0.5 * mask_width
    invg = 1 / target_gamma
    mask = np.logical_and(np.logical_and(base**invg >= mask_min,
                                         base**invg <= mask_max),
                          np.logical_and(other**invg >= mask_min,
                                         other**invg <= mask_max))

    if np.sum(mask) == 0:
        raise RuntimeError('No values to match exposure, '
                           'try increasing the mask width')

    # Compute scaling of `other` image to match exposure of `base`
    scale = np.mean(base[mask] / other[mask])

    if blend_width is None:
        blend_width = 1 - mask_width
    if blend_width <= 0 or blend_width > 0.5:
        raise ValueError('Invalid blend width, must be in range [0, 0.5]')
    eps = 1e-6
    # Compute blending weight
    weight = np.minimum(np.minimum(other**invg,
                                   1 - other**invg) / blend_width + eps,
            1)

    # Blend scaled `other` image into `base` image
    return (1 - weight) * base + weight * scale * other


def merge_multiple_exposures(base, others, **kwargs):
    """Merges multiple LDR images into a (possibly HDR) base image.

    Parameters
    ----------

    base : array_like
        Floating point input image in linear color space and desired exposure.
    others : list of array_like
        Floating point input images that will be merged into `base`.
    kwargs : arguments forwarded to `merge_exposures`

    Returns
    -------

    merged : ndarray
        Merged HDR image of `base` and `others` with same exposure as `base`.
    """
    for other in others:
        base = merge_exposures(base, other, **kwargs)
    return base
