"""Unit tests of rawhdr.focusstack."""

import pytest

import numpy as np
from scipy import ndimage

from rawhdr import focusstack

rng = np.random.default_rng(42)


def sharpness(image):
    bw = image
    if image.ndim == 3:
        bw = np.mean(bw, axis=2)
    lowpass = ndimage.gaussian_filter(bw, sigma=1.5)
    highpass = bw - lowpass
    return ndimage.gaussian_filter(np.abs(highpass), sigma=1.5)


@pytest.fixture
def image_gen():
    def gen():
        n = 200
        sharp = rng.uniform(size=(n, 300, 3))
        image0 = np.copy(sharp)
        image1 = np.copy(sharp)
        image0[n // 2:, :, :] = 0.5
        image1[:n // 2, :, :] = 0.5
        return sharp, image0, image1

    return gen


@pytest.fixture(params=[
    focusstack.merge_dtcwt, focusstack.merge_highpass,
    focusstack.merge_waveletes, focusstack.merge_waveletes2
])
def merger(request):
    return request.param


def test_equal(merger, image_gen):
    """Test stacking of images with equal sharpness."""
    sharp, _, _ = image_gen()
    stacked = merger(sharp, sharp)
    assert np.allclose(stacked, sharp)


def test_improved_sharpness(merger, image_gen):
    """Test sharpness improvement of stacking."""
    _, image0, image1 = image_gen()
    stacked = merger(image0, image1)

    assert np.mean(sharpness(stacked)) > np.mean(sharpness(image0))
    assert np.mean(sharpness(stacked)) > np.mean(sharpness(image1))


def test_close_to_sharp(merger, image_gen):
    """Check that stacked image sharpness is close to perfectly sharp."""
    sharp, image0, image1 = image_gen()
    stacked = merger(image0, image1)
    assert np.mean(sharpness(stacked)) / np.mean(sharpness(sharp)) > 0.9


def test_bw_input(merger, image_gen):
    _, image0, image1 = image_gen()
    stacked = merger(image0[:, :, 0], image1[:, :, 0])
    assert stacked.shape == image0.shape[:2]
