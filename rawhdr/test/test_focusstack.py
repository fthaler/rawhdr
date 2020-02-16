"""Unit tests of rawhdr.focusstack."""

import unittest

import numpy as np
from scipy import ndimage

from rawhdr import focusstack


def _generate_images():
    n = 200
    sharp = np.random.RandomState(42).uniform(size=(n, 300, 3))
    image0 = np.copy(sharp)
    image1 = np.copy(sharp)
    image0[n // 2:, :, :] = 0.5
    image1[:n // 2, :, :] = 0.5
    return sharp, image0, image1


_MERGERS = [
    focusstack.merge_dtcwt, focusstack.merge_highpass,
    focusstack.merge_waveletes, focusstack.merge_waveletes2
]


def _sharpness(image):
    bw = image
    if image.ndim == 3:
        bw = np.mean(bw, axis=2)
    lowpass = ndimage.gaussian_filter(bw, sigma=1.5)
    highpass = bw - lowpass
    return ndimage.gaussian_filter(np.abs(highpass), sigma=1.5)


class TestStackImages(unittest.TestCase):
    def mergers(self):
        for merge in _MERGERS:
            with self.subTest(merge=merge.__name__):
                yield merge

    """Tests for focusstack.stack_images()."""

    def test_equal(self):
        """Test stacking of images with equal sharpness."""
        for merge in self.mergers():
            sharp, _, _ = _generate_images()
            stacked = merge(sharp, sharp)
            self.assertTrue(np.allclose(stacked, sharp))

    def test_improved_sharpness(self):
        """Test sharpness improvement of stacking."""
        _, image0, image1 = _generate_images()
        for merge in self.mergers():
            stacked = merge(image0, image1)

            self.assertGreater(np.mean(_sharpness(stacked)),
                               np.mean(_sharpness(image0)))
            self.assertGreater(np.mean(_sharpness(stacked)),
                               np.mean(_sharpness(image1)))

    def test_close_to_sharp(self):
        """Check that stacked image sharpness is close to perfectly sharp."""
        for merge in self.mergers():
            sharp, image0, image1 = _generate_images()
            stacked = merge(image0, image1)
            self.assertGreater(
                np.mean(_sharpness(stacked)) / np.mean(_sharpness(sharp)), 0.9)

    def test_bw_input(self):
        for merge in self.mergers():
            rng = np.random.RandomState(42)
            first = rng.uniform(size=(200, 300))
            second = rng.uniform(size=(200, 300))
            stacked = merge(first, second)
            self.assertEqual(stacked.shape, (200, 300))
