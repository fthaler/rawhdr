"""Unit tests of rawhdr.merge."""

import pytest

import numpy as np

from rawhdr import merge

rng = np.random.default_rng(42)


@pytest.fixture
def image_gen():
    return lambda: rng.uniform(size=(15, 10, 3))


def test_unscaled(image_gen):
    """Same brightness for both input images."""
    base_image = image_gen()
    image = np.copy(base_image)

    result = merge.compute_scaling(image, base_image)
    assert np.isclose(result, 1)


def test_same_image(image_gen):
    """id(image) == id(base_image)."""
    image = image_gen()

    result = merge.compute_scaling(image, image)
    assert np.isclose(result, 1)


def test_scaled(image_gen):
    """Basic functionality test."""
    base_image = image_gen()

    image = base_image / 1.5
    result = merge.compute_scaling(image, base_image)
    assert np.isclose(result, 1.5)

    image = base_image / 0.5
    result = merge.compute_scaling(image, base_image)
    assert np.isclose(result, 0.5)


def test_argument_check(image_gen):
    """Test checking of function argument values."""
    base_image = image_gen()
    image = base_image / 1.5

    merge.compute_scaling(image, base_image, mask_width=0.3, target_gamma=1.0)

    with pytest.raises(ValueError):
        merge.compute_scaling(image, base_image, mask_width=-0.01)
    with pytest.raises(ValueError):
        merge.compute_scaling(image, base_image, mask_width=1.01)
    with pytest.raises(ValueError):
        merge.compute_scaling(image, base_image, target_gamma=-1)


def test_large_factor_fails(image_gen):
    """Test raising of RuntimeError for very large scaling factors."""
    base_image = image_gen()

    with pytest.raises(RuntimeError):
        merge.compute_scaling(base_image / 1000, base_image)
    with pytest.raises(RuntimeError):
        merge.compute_scaling(base_image / 1e-3, base_image)


def test_no_blending(image_gen):
    """Test disabled blending."""
    image = image_gen()

    result = merge.compute_weight(image, blend_low=False, blend_high=False)
    assert np.allclose(result, 1)


def test_blending():
    """Test full blending."""
    linear_image = np.linspace(0, 1, 100)
    expected = np.clip((linear_image - 0.1) / 0.3, 0, 1)

    result = merge.compute_weight(linear_image**2.0,
                                  blend_low=True,
                                  blend_high=False,
                                  blend_width=0.3,
                                  blend_cap=0.1,
                                  target_gamma=2.0)

    assert np.allclose(result, expected)
    result = merge.compute_weight(linear_image**2.0,
                                  blend_low=False,
                                  blend_high=True,
                                  blend_width=0.3,
                                  blend_cap=0.1,
                                  target_gamma=2.0)
    assert np.allclose(result, expected[::-1])
    result = merge.compute_weight(linear_image**2.0,
                                  blend_low=True,
                                  blend_high=True,
                                  blend_width=0.3,
                                  blend_cap=0.1,
                                  target_gamma=2.0)
    assert np.allclose(result, np.minimum(expected, expected[::-1]))


def test_weight_argument_check(image_gen):
    """Test checking of function argument values."""
    image = image_gen()

    with pytest.raises(ValueError):
        merge.compute_weight(image, blend_width=-0.01)
    with pytest.raises(ValueError):
        merge.compute_weight(image, blend_width=0.51)
    with pytest.raises(ValueError):
        merge.compute_weight(image, blend_cap=-0.01)
    with pytest.raises(ValueError):
        merge.compute_weight(image, blend_cap=0.51)
    with pytest.raises(ValueError):
        merge.compute_weight(image, blend_width=0.3, blend_cap=0.3)
    with pytest.raises(ValueError):
        merge.compute_weight(image, target_gamma=-0.01)


def test_merging(image_gen):
    """Test merging of LDR images."""
    # Generate HDR image
    image = 2 * image_gen()

    # Generate clipped LDR images
    scalings = [1, 0.4, 0.7, 1 / 0.7, 1 / 0.4]
    images = [np.clip(image * scaling, 1e-3, 0.95) for scaling in scalings]

    result = merge.merge_exposures(images)
    assert np.allclose(image, result)


def test_merging_without_weight_first(image_gen):
    """Test merging of LDR images with weight_first=False."""
    # Generate HDR image
    image = 2 * image_gen()

    # Generate clipped LDR images
    scalings = [1, 0.4, 0.7, 1 / 0.7, 1 / 0.4]
    images = [np.clip(image * scaling, 1e-3, 0.95) for scaling in scalings]

    result = merge.merge_exposures(images, weight_first=False)
    assert np.allclose(image, result)


def test_merge_argument_check():
    """Test checking of function argument values."""
    with pytest.raises(ValueError):
        merge.merge_exposures([])
