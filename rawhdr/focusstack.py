# -*- coding: utf-8 -*-
"""Functions for focus stacking of images."""

import numpy as np
from scipy import ndimage


def compute_sharpness(image, sigma=None, power=None):
    if sigma is None:
        sigma = 1.5
    if power is None:
        power = 5
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    lowpass = ndimage.gaussian_filter(image, sigma=sigma)
    highpass = image - lowpass
    return ndimage.gaussian_filter(np.abs(highpass / (lowpass + 1e-8)),
                                   sigma=sigma)**power


def stack_images(images, sigma=None, power=None):
    images = list(images)
    if not images:
        raise ValueError('At least one input image is required')

    weights = [compute_sharpness(image, sigma, power) for image in images]
    total_weight = sum(weights)
    for weight in weights:
        weight[total_weight < 1e-8] = 1.0
    total_weight = sum(weights)
    for weight in weights:
        weight /= total_weight
    return sum(image * weight[:, :, np.newaxis]
               for image, weight in zip(images, weights))
