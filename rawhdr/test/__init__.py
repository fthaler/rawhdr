# -*- coding: utf-8 -*-
"""Unit tests of rawhdr."""

import unittest

import numpy as np

import rawhdr


def _generate_image():
    np.random.seed(42)
    return np.random.uniform(size=(15, 10, 3))


class TestComputeScaling(unittest.TestCase):
    """Tests for rawhdr.compute_scaling()."""

    def test_unscaled(self):
        """Same brightness for both input images."""
        base_image = _generate_image()
        image = np.copy(base_image)

        result = rawhdr.compute_scaling(image, base_image)
        self.assertTrue(np.isclose(result, 1))

    def test_same_image(self):
        """id(image) == id(base_image)."""
        image = _generate_image()

        result = rawhdr.compute_scaling(image, image)
        self.assertTrue(np.isclose(result, 1))

    def test_scaled(self):
        """Basic functinality test."""
        base_image = _generate_image()

        image = base_image / 1.5
        result = rawhdr.compute_scaling(image, base_image)
        self.assertTrue(np.isclose(result, 1.5))

        image = base_image / 0.5
        result = rawhdr.compute_scaling(image, base_image)
        self.assertTrue(np.isclose(result, 0.5))

    def test_argument_check(self):
        """Test checking of function argument values."""
        base_image = _generate_image()
        image = base_image / 1.5

        rawhdr.compute_scaling(image,
                               base_image,
                               mask_width=0.3,
                               target_gamma=1.0)

        with self.assertRaises(ValueError):
            rawhdr.compute_scaling(image, base_image, mask_width=-0.01)
        with self.assertRaises(ValueError):
            rawhdr.compute_scaling(image, base_image, mask_width=1.01)
        with self.assertRaises(ValueError):
            rawhdr.compute_scaling(image, base_image, target_gamma=-1)

    def test_large_factor_fails(self):
        """Test raising of RuntimeError for very large scaling factors."""
        base_image = _generate_image()

        with self.assertRaises(RuntimeError):
            rawhdr.compute_scaling(base_image / 1000, base_image)
        with self.assertRaises(RuntimeError):
            rawhdr.compute_scaling(base_image / 1e-3, base_image)


class TestComputeWeight(unittest.TestCase):
    """Tests for rawhdr.compute_weight()."""

    def test_no_blending(self):
        """Test disabled blending."""
        image = _generate_image()

        result = rawhdr.compute_weight(image,
                                       blend_low=False,
                                       blend_high=False)
        self.assertTrue(np.all(np.isclose(result, 1)))

    def test_blending(self):
        """Test full blending."""
        linear_image = np.linspace(0, 1, 100)
        expected = np.clip((linear_image - 0.1) / 0.3, 0, 1)

        result = rawhdr.compute_weight(linear_image**2.0,
                                       blend_low=True,
                                       blend_high=False,
                                       blend_width=0.3,
                                       blend_cap=0.1,
                                       target_gamma=2.0)

        self.assertTrue(np.all(np.isclose(result, expected)))
        result = rawhdr.compute_weight(linear_image**2.0,
                                       blend_low=False,
                                       blend_high=True,
                                       blend_width=0.3,
                                       blend_cap=0.1,
                                       target_gamma=2.0)
        self.assertTrue(np.all(np.isclose(result, expected[::-1])))
        result = rawhdr.compute_weight(linear_image**2.0,
                                       blend_low=True,
                                       blend_high=True,
                                       blend_width=0.3,
                                       blend_cap=0.1,
                                       target_gamma=2.0)
        self.assertTrue(
            np.all(np.isclose(result, np.minimum(expected, expected[::-1]))))

    def test_argument_check(self):
        """Test checking of function argument values."""
        image = _generate_image()

        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, blend_width=-0.01)
        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, blend_width=0.51)
        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, blend_cap=-0.01)
        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, blend_cap=0.51)
        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, blend_width=0.3, blend_cap=0.3)
        with self.assertRaises(ValueError):
            rawhdr.compute_weight(image, target_gamma=-0.01)


class TestMergeExposures(unittest.TestCase):
    """Tests for rawhdr.merge_exposures()."""

    def test_merging(self):
        """Test merging of LDR images."""
        # Generate HDR image
        image = 2 * _generate_image()

        # Generate clipped LDR images
        scalings = [1, 0.4, 0.7, 1 / 0.7, 1 / 0.4]
        images = [np.clip(image * scaling, 1e-3, 0.95) for scaling in scalings]

        result = rawhdr.merge_exposures(images)
        self.assertTrue(np.all(np.isclose(image, result)))

    def test_argument_check(self):
        """Test checking of function argument values."""
        with self.assertRaises(ValueError):
            rawhdr.merge_exposures([])
