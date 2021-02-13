import pytest

import numpy as np

from rawhdr import common

rng = np.random.default_rng(42)


@pytest.fixture
def image_gen():
    return lambda: rng.uniform(size=(5, 7, 3))


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
