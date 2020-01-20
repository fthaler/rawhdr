# -*- coding: utf-8 -*-
"""Unit tests of rawhdr.focusstack."""

import unittest

import numpy as np

from rawhdr import focusstack


def _generate_images():
    np.random.seed(42)
    n = 20
    sharp = np.random.uniform(size=(n, 10, 3))
    image0 = np.copy(sharp)
    image1 = np.copy(sharp)
    image0[n // 2:, :, :] = 0.5
    image1[:n // 2, :, :] = 0.5
    return sharp, image0, image1


class TestComputeSharpness(unittest.TestCase):
    """Tests for focusstack.compute_sharpness()."""
    def test_uniform(self):
        """Higher sharpness for random image than for uniform."""
        np.random.seed(42)
        ones = np.full((15, 10, 3), 0.5)
        random = np.random.uniform(size=(15, 10, 3))
        ones_sharpness = focusstack.compute_sharpness(ones)
        random_sharpness = focusstack.compute_sharpness(random)
        self.assertTrue(np.all(random_sharpness > ones_sharpness))

    def test_relative(self):
        """Relative sharpness test."""
        _, image0, image1 = _generate_images()
        sharpness0 = focusstack.compute_sharpness(image0)
        sharpness1 = focusstack.compute_sharpness(image1)
        n = image0.shape[0]
        axis_sharpness0 = np.mean(sharpness0, axis=1)
        axis_sharpness1 = np.mean(sharpness1, axis=1)
        relative_sharpness = axis_sharpness0 / (axis_sharpness0 +
                                                axis_sharpness1)
        self.assertTrue(np.all(relative_sharpness[:n // 2] >= 0.5))
        self.assertTrue(np.all(relative_sharpness[n // 2:] <= 0.5))


class TestStackImages(unittest.TestCase):
    """Tests for focusstack.stack_images()."""
    def test_equal(self):
        """Test stacking of images with equal sharpness."""
        sharp, _, _ = _generate_images()
        stacked = focusstack.stack_images([sharp, sharp, sharp])
        self.assertTrue(np.allclose(stacked, sharp))

    def test_improved_sharpness(self):
        """Test sharpness improvement of stacking."""
        _, image0, image1 = _generate_images()
        stacked = focusstack.stack_images([image0, image1])

        self.assertGreater(np.mean(focusstack.compute_sharpness(stacked)),
                           np.mean(focusstack.compute_sharpness(image0)))
        self.assertGreater(np.mean(focusstack.compute_sharpness(stacked)),
                           np.mean(focusstack.compute_sharpness(image1)))

    def test_close_to_sharp(self):
        """Check that stacked image sharpness is close to perfectly sharp."""
        sharp, image0, image1 = _generate_images()
        stacked = focusstack.stack_images([image0, image1])
        self.assertGreater(
            np.mean(focusstack.compute_sharpness(stacked)) /
            np.mean(focusstack.compute_sharpness(sharp)), 0.95)
