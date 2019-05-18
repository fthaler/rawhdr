# -*- coding: utf-8 -*-
"""RawHdr merges multiple RAW images into an floating point HDR image."""

import numpy as np

__version__ = '0.2.0'


def compute_scaling(image, base_image, *, mask_width=None, target_gamma=None):
    """Exposure scaling computation.

    Computes the scaling required for `image` such that its
    brightness/exposure is equal to `base_image`.

    Only the central part of the brightness distribution of both images is
    used to perform the calculation. Very bright (possibly saturated) and
    dark (possibly noisy) pixels are ignored.

    Masking of the center part can be controlled by `mask_width` and
    `target_gamma`, in most cases the default parameters should work
    reasonaly well.

    Parameters
    ----------
    image : array_like
        Image for which the scaling factor is computed.
    base_image : array_like
        Image with reference brightness/exposure.
    mask_width : float
        Size of the mask in gamma-corrected color space.
    target_gamma : float
        Target gamma used for the masking computations.

    Returns
    -------
    scaling : float
        Scaling factor such that `scaling` * `image` is approximately
        equal to `base_image`.
    """
    if mask_width is None:
        mask_width = 0.8
    if target_gamma is None:
        target_gamma = 2.2

    if mask_width <= 0 or mask_width > 1:
        raise ValueError('Maske width must be positive and at most 1.0')
    if target_gamma <= 0:
        raise ValueError('Invalid value for `target_gamma`. Must be positive')

    if base_image is image:
        return 1.0

    base_image = np.asarray(base_image)
    image = np.asarray(image)

    # Compute mask where all image data is in reasonable sensor range
    mask_min = 0.5 - 0.5 * mask_width
    mask_max = 0.5 + 0.5 * mask_width
    image_gammac = image**(1.0 / target_gamma)
    base_image_gammac = base_image**(1.0 / target_gamma)
    mask = ((base_image_gammac >= mask_min)
            & (base_image_gammac <= mask_max)
            & (image_gammac >= mask_min)
            & (image_gammac <= mask_max))

    if np.sum(mask) == 0:
        raise RuntimeError('No values to match exposure, '
                           'try increasing the mask width')

    # Compute scaling of `image` image to match exposure of `base_image`
    return np.mean(base_image[mask] / image[mask])


def compute_weight(image,
                   *,
                   blend_low=True,
                   blend_high=True,
                   blend_width=None,
                   blend_cap=None,
                   target_gamma=None):
    """Compute per-pixel blending weights.

    Parameters
    ----------
    image : array_like
        Image for which the blending weights are computed.
    blend_low : bool
        Enables blending of the lower (dark) part of the image brightness.
    blend_high : bool
        Enabled blending of the higher (dark) part of the image brightness.
    blend_width : float
        Width of the blended regions at both ends of the brightness range.
    blend_cap : float
        Cap of dark and bright regions.
    target_gamma : float
        Gamma at which the blending is performed.

    Returns
    -------
    mask : array_like
        Mask with the same shape as `image`.
    """
    if blend_width is None:
        blend_width = 0.2
    if blend_cap is None:
        blend_cap = 0.1
    if target_gamma is None:
        target_gamma = 2.2

    if not 0 <= blend_width <= 0.5:
        raise ValueError(
            'Invalid value for `blend_width`. Must be in range [0, 0.5]')
    if not 0 <= blend_cap <= 0.5:
        raise ValueError(
            'Invalid value for `blend_cap`. Must be in range [0, 0.5]')
    if blend_width + blend_cap > 0.5:
        raise ValueError('Invalid value for `blend_width` + `blend_cap`.'
                         'Sum must be less than 0.5')
    if target_gamma <= 0:
        raise ValueError('Invalid value for `target_gamma`. Must be positive')

    image_gammac = image**(1.0 / target_gamma)
    mask = np.ones_like(image)
    if blend_low:
        mask = np.minimum(np.clip((image_gammac - blend_cap) / blend_width, 0,
                                  1),
                          mask,
                          out=mask)
    if blend_high:
        mask = np.minimum(np.clip((1 - image_gammac - blend_cap) / blend_width,
                                  0, 1),
                          mask,
                          out=mask)

    return mask


def merge_exposures(exposures,
                    mask_width=None,
                    blend_width=None,
                    blend_cap=None,
                    target_gamma=None):
    """Merge multiple LDR images into a HDR image.

    Parameters
    ----------
    exposures : list of array_like
        Floating point input images in linear color space.
    mask_width : float
        Size of the mask in gamma-corrected color space.
    blend_width : float
        Width of the blended regions at both ends of the brightness range.
    blend_cap : float
        Cap of dark and bright regions.
    target_gamma : float
        Target gamma used for the masking computations.

    Returns
    -------
    merged : ndarray
        Merged HDR image width same exposure as the first image in the
        `exposures` input list.
    """
    scalings = [
        compute_scaling(exposure,
                        base_image=exposures[0],
                        mask_width=mask_width,
                        target_gamma=target_gamma) for exposure in exposures
    ]
    min_scaling, max_scaling = min(scalings), max(scalings)

    weights = [
        compute_weight(exposure,
                       blend_low=(scaling != min_scaling),
                       blend_high=(scaling != max_scaling),
                       blend_width=blend_width,
                       blend_cap=blend_cap,
                       target_gamma=target_gamma)
        for exposure, scaling in zip(exposures, scalings)
    ]

    total_weight = sum(weights)
    for weight in weights:
        weight /= total_weight

    return sum(
        exposure * scaling * weight
        for exposure, scaling, weight in zip(exposures, scalings, weights))
