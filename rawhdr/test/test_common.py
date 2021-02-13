import pytest

import numpy as np

from rawhdr import common

rng = np.random.default_rng(42)


@pytest.fixture
def image_gen():
    return lambda: rng.uniform(size=(5, 7, 3))


def test_principal_component(image_gen):
    image = image_gen()

    # set two channels to zero, remaining channel to zero mean
    image[:, :, 1:] = 0
    image[:, :, 0] -= np.mean(image[:, :, 0])

    # rotate with some rotation matrix
    rot = rng.normal(size=(3, 3))
    rot = rot.T @ rot
    rot /= np.sqrt(np.sum(rot**2, axis=0, keepdims=True))
    rotated = np.einsum("ik,...k->...i", rot, image, optimize=True)

    # check that PCA recovers (possibly negative) unrotated result
    pc = common.principal_component(rotated)
    assert np.allclose(pc, image[..., 0]) or np.allclose(-pc, image[..., 0])


def test_reduce_color_dimension(image_gen):
    image = image_gen()

    assert common.reduce_color_dimension(image,
                                         pca=False).shape == image.shape[:2]
    assert common.reduce_color_dimension(image,
                                         pca=True).shape == image.shape[:2]
    assert common.reduce_color_dimension(image[...,
                                               0]).shape == image.shape[:2]


@pytest.fixture(params=[True, False])
def temporary_array_list(request):
    return common.temporary_array_list(in_memory=request.param)


def test_temporary_array_list(temporary_array_list, image_gen):
    assert len(temporary_array_list) == 0

    reference = []
    for _ in range(3):
        reference.append(image_gen())
        temporary_array_list.append(reference[-1])
    assert len(temporary_array_list) == 3

    assert np.all(temporary_array_list[0] == reference[0])
    assert np.all(temporary_array_list[1] == reference[1])
    assert np.all(temporary_array_list[2] == reference[2])

    for i, r in zip(temporary_array_list, reference):
        assert np.all(i == r)

    for i in range(3):
        reference[i] = image_gen()
        temporary_array_list[i] = reference[i]
        assert np.all(temporary_array_list[i] == reference[i])

    for i in range(-3, 0):
        reference[i] = image_gen()
        temporary_array_list[i] = reference[i]
        assert np.all(temporary_array_list[i] == reference[i])
