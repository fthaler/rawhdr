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
    def create(*args, **kwargs):
        return common.temporary_array_list(*args,
                                           in_memory=request.param,
                                           **kwargs)

    return create


def test_temporary_array_list(temporary_array_list, image_gen):
    tal = temporary_array_list()
    assert len(tal) == 0

    reference = []
    for _ in range(3):
        reference.append(image_gen())
        tal.append(reference[-1])
    assert len(tal) == 3

    assert np.all(tal[0] == reference[0])
    assert np.all(tal[1] == reference[1])
    assert np.all(tal[2] == reference[2])

    for i, r in zip(tal, reference):
        assert np.all(i == r)

    for i in range(3):
        reference[i] = image_gen()
        tal[i] = reference[i]
        assert np.all(tal[i] == reference[i])

    for i in range(-3, 0):
        reference[i] = image_gen()
        tal[i] = reference[i]
        assert np.all(tal[i] == reference[i])

    for i, r in zip(tal[1:], reference[1:]):
        assert np.all(i == r)

    for i, r in zip(tal[::2], reference[::2]):
        assert np.all(i == r)

    for i, r in zip(tal[2:0:-1], reference[2:0:-1]):
        assert np.all(i == r)


def test_temporary_array_list_from_iterable(temporary_array_list, image_gen):
    reference = [image_gen() for i in range(10)]
    tal = temporary_array_list(reference)
    assert len(tal) == len(reference)
    for i, r in zip(tal, reference):
        assert np.all(i == r)
